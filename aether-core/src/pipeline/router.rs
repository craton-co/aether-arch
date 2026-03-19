//! Adaptive routing: decides which compression method to use per chunk
//! and dispatches to the appropriate compressor.
//!
//! The routing cascade for the PredictorRans path is:
//!   1. Try BWT+MTF → predictor+range coding  (BwtPredictorRans)
//!   2. Try LZ77 → predictor+range coding     (Lz77PredictorRans)
//!   3. Try plain predictor+range coding       (PredictorRans)
//!   4. Fall back to Zstd
//!   5. Fall back to Store
//!
//! Since encode_block resets the predictor at the start of each call,
//! we can try multiple transforms and pick the smallest result.

use crate::analyzer::{self, RecommendedMethod};
use crate::chunker::Chunk;
#[cfg(feature = "lz4")]
use crate::coding::lz_preprocess;
use crate::coding::{bwt_preprocess, lz77_preprocess, rans, zstd_fallback};
use crate::entropy::{NeuralSsmPredictor, ProbabilityPredictor};
use crate::error::{AetherError, Result};
use crate::format::{CompressionMethod, BWT_DECISIVE_RATIO, MAX_DECOMPRESSED_BLOCK_SIZE};

/// Result of compressing a single chunk via the adaptive routing cascade.
///
/// Contains the winning compression method, compressed payload, and metadata
/// needed to write the block header and trailer.
#[derive(Debug)]
pub struct CompressedChunk {
    /// Which compression method produced the smallest output.
    pub method: CompressionMethod,
    /// Compressed payload bytes (smallest of all tried methods).
    pub data: Vec<u8>,
    /// Original uncompressed size in bytes.
    pub original_size: usize,
    /// BLAKE3 hash of the original uncompressed data.
    pub blake3_hash: [u8; 32],
    /// Whether `sync_predictor` was called during compression.
    /// When `false`, the decompressor must also skip syncing to match.
    pub predictor_synced: bool,
}

/// Compress a chunk using the best method based on its entropy.
///
/// Tries multiple transforms and picks the smallest result.
/// Since `encode_block` calls `predictor.reset()` at the start,
/// each attempt starts with clean predictor state.
pub fn compress_chunk(
    chunk: &Chunk,
    predictor: &mut dyn ProbabilityPredictor,
    content_type: crate::format::ContentType,
) -> Result<CompressedChunk> {
    let method = analyzer::recommend_method_for(chunk.entropy, content_type);

    let mut predictor_synced = true;

    let (compression_method, compressed_data) = match method {
        RecommendedMethod::PredictorRans => {
            let mut best: Option<(CompressionMethod, Vec<u8>)> = None;

            // ── Try BWT+MTF+RLE → predictor + range coding ────────────
            // BWT clusters context; MTF converts to small integers; RLE
            // compacts zero runs using bijective base-2 (RUNA/RUNB).
            // The predictor sees the RLE stream, not the raw MTF stream.
            // bwt_mtf_encode_parts returns Err if input exceeds MAX_BWT_INPUT_SIZE;
            // we treat that as "BWT not applicable" and fall through to LZ77/plain.
            if chunk.data.len() >= 8 {
                if let Ok((primary_index, mtf_data)) =
                    bwt_preprocess::bwt_mtf_encode_parts(&chunk.data)
                {
                    // Try RLE first (much more compact); fall back to raw MTF
                    let (encode_data, rle_applied) =
                        if let Some(rle) = bwt_preprocess::rle_encode(&mtf_data) {
                            (rle, true)
                        } else {
                            (mtf_data, false)
                        };

                    let mut bwt_predictor = NeuralSsmPredictor::new();
                    if let Ok(rc_bytes) = rans::encode_block(&encode_data, &mut bwt_predictor) {
                        // Payload: [flags: u8] [primary_index: u32] [encoded_len: u32] [RC bytes]
                        let flags: u8 = if rle_applied { 1 } else { 0 };
                        let encoded_len = encode_data.len() as u32;
                        let mut payload = Vec::with_capacity(1 + 4 + 4 + rc_bytes.len());
                        payload.push(flags);
                        payload.extend_from_slice(&primary_index.to_le_bytes());
                        payload.extend_from_slice(&encoded_len.to_le_bytes());
                        payload.extend_from_slice(&rc_bytes);

                        if payload.len() < chunk.data.len() {
                            best = Some((CompressionMethod::BwtPredictorRans, payload));
                        }
                    }
                }
            }

            // ── Try LZ77 → predictor + range coding ──────────────────
            // Skip if BWT already compressed below BWT_DECISIVE_RATIO —
            // LZ77 won't beat it on text, and we can skip predictor sync.
            // Use division instead of multiplication to avoid overflow for
            // large chunks (b.len() * 100 could overflow usize).
            let bwt_decisive = best.as_ref().is_some_and(|(_, b)| {
                !chunk.data.is_empty() && b.len() < chunk.data.len() / 100 * BWT_DECISIVE_RATIO
            });
            if !bwt_decisive {
                if let Some(lz_bytes) = lz77_preprocess::lz77_encode(&chunk.data) {
                    // Use a fresh predictor for the LZ77 trial so that a
                    // failed attempt does not corrupt the group predictor's
                    // state. The winning path syncs via sync_predictor below.
                    let mut lz_predictor = NeuralSsmPredictor::new();
                    if let Ok(rc_bytes) = rans::encode_block(&lz_bytes, &mut lz_predictor) {
                        let lz_len = lz_bytes.len() as u32;

                        let mut payload = Vec::with_capacity(4 + rc_bytes.len());
                        payload.extend_from_slice(&lz_len.to_le_bytes());
                        payload.extend_from_slice(&rc_bytes);

                        if payload.len() < chunk.data.len() {
                            let is_better =
                                best.as_ref().is_none_or(|(_, b)| payload.len() < b.len());
                            if is_better {
                                best = Some((CompressionMethod::Lz77PredictorRans, payload));
                            }
                        }
                    }
                }
            }

            // ── Try plain predictor + range coding ───────────────────
            // Use a fresh predictor so a failed attempt does not corrupt
            // the group predictor's state (encode_block calls reset()).
            if best.is_none() {
                let mut plain_predictor = NeuralSsmPredictor::new();
                if let Ok(rc_bytes) = rans::encode_block(&chunk.data, &mut plain_predictor) {
                    if rc_bytes.len() < chunk.data.len() {
                        best = Some((CompressionMethod::PredictorRans, rc_bytes));
                    }
                }
            }

            // ── Sync predictor to the winning path ───────────────────
            //
            // **Design note**: `encode_block` / `decode_block` both call
            // `predictor.reset()` at the start of every block.  Cross-block
            // predictor state is therefore NOT consumed by predictor-based
            // paths (PredictorRans, LZ77, BWT).  The `sync_predictor` call
            // below maintains advisory state for Zstd/Store paths and
            // forward-compatible use.
            //
            // Previous implementation re-encoded with the winning path
            // (calling `rans::encode_block` a second time), which:
            //   1. Wasted CPU on a redundant encode pass.
            //   2. Reset the group predictor via `encode_block().reset()`,
            //      destroying accumulated cross-block state and causing
            //      compressor/decompressor sync_predictor divergence.
            //   3. Introduced a theoretical floating-point non-determinism
            //      risk if the predictor ran different code paths on
            //      different platforms.
            //
            // The fix: use `sync_predictor` for all winning paths.  This
            // feeds the original chunk data through predict+update WITHOUT
            // resetting, preserving cross-block state symmetrically with
            // the decompressor.
            if let Some((method, payload)) = best {
                match method {
                    CompressionMethod::BwtPredictorRans => {
                        // BWT uses its own predictor internally; sync the group
                        // predictor on the original data for cross-block state.
                        // Skip when BWT won decisively: subsequent chunks of the
                        // same content type will also use BWT, so this state is
                        // unlikely to be consumed by a LZ77/plain path block.
                        if !bwt_decisive {
                            sync_predictor(predictor, &chunk.data);
                        } else {
                            predictor_synced = false;
                        }
                    }
                    CompressionMethod::Lz77PredictorRans | CompressionMethod::PredictorRans => {
                        // Feed original data (not the LZ77/RC encoded form)
                        // through the predictor to maintain cross-block state.
                        // This matches the decompressor path which also calls
                        // sync_predictor on the decompressed original data.
                        sync_predictor(predictor, &chunk.data);
                    }
                    _ => {}
                }
                (method, payload)
            } else {
                // Nothing helped — fall back to zstd/store
                sync_predictor(predictor, &chunk.data);
                try_zstd_or_store(chunk)
            }
        }
        RecommendedMethod::Zstd => {
            let compressed = zstd_fallback::compress(&chunk.data)?;
            sync_predictor(predictor, &chunk.data);
            if compressed.len() >= chunk.data.len() {
                (CompressionMethod::Store, chunk.data.clone())
            } else {
                (CompressionMethod::Zstd, compressed)
            }
        }
        RecommendedMethod::Store => {
            sync_predictor(predictor, &chunk.data);
            (CompressionMethod::Store, chunk.data.clone())
        }
    };

    Ok(CompressedChunk {
        method: compression_method,
        data: compressed_data,
        original_size: chunk.length,
        blake3_hash: chunk.blake3_hash,
        predictor_synced,
    })
}

/// Decompress a chunk based on its stored compression method.
///
/// `predictor_synced`: if `true`, call `sync_predictor` after decompression
/// to advance the group predictor state. If `false`, skip sync (matches the
/// compressor's decision to skip when BWT won decisively).
///
/// # Safety Limits
///
/// Rejects `uncompressed_size` exceeding [`MAX_DECOMPRESSED_BLOCK_SIZE`] (64 MiB)
/// to prevent out-of-memory from crafted archives.
pub fn decompress_chunk(
    compressed_data: &[u8],
    method: CompressionMethod,
    uncompressed_size: usize,
    predictor: &mut dyn ProbabilityPredictor,
    predictor_synced: bool,
) -> Result<Vec<u8>> {
    // Bounds check: reject implausibly large decompressed sizes
    if uncompressed_size > MAX_DECOMPRESSED_BLOCK_SIZE {
        return Err(AetherError::ResourceLimitExceeded(format!(
            "Decompressed block size {} exceeds maximum {} bytes",
            uncompressed_size, MAX_DECOMPRESSED_BLOCK_SIZE,
        )));
    }

    match method {
        CompressionMethod::BwtPredictorRans => {
            if compressed_data.len() < 9 {
                return Err(crate::error::AetherError::Decompression(format!(
                    "BwtPredictorRans payload too short: {} bytes (need ≥9, uncompressed_size={})",
                    compressed_data.len(),
                    uncompressed_size,
                )));
            }
            let flags = compressed_data[0];
            let rle_applied = (flags & 1) != 0;
            let primary_index =
                u32::from_le_bytes(compressed_data[1..5].try_into().map_err(|_| {
                    AetherError::Decompression(
                        "BwtPredictorRans: truncated primary_index field".into(),
                    )
                })?);
            let encoded_len =
                u32::from_le_bytes(compressed_data[5..9].try_into().map_err(|_| {
                    AetherError::Decompression(
                        "BwtPredictorRans: truncated encoded_len field".into(),
                    )
                })?) as usize;

            // Bounds check: encoded_len must be ≤ MAX_DECOMPRESSED_BLOCK_SIZE.
            // BWT doesn't expand data (MTF output = input length), and RLE can
            // only shrink it, so encoded_len should be ≤ uncompressed_size.
            if encoded_len > MAX_DECOMPRESSED_BLOCK_SIZE {
                return Err(AetherError::ResourceLimitExceeded(format!(
                    "BWT encoded_len {} exceeds safety limit {} (uncompressed_size={})",
                    encoded_len, MAX_DECOMPRESSED_BLOCK_SIZE, uncompressed_size,
                )));
            }

            let rc_bytes = &compressed_data[9..];

            let mut bwt_predictor = NeuralSsmPredictor::new();
            let encode_data = rans::decode_block(rc_bytes, encoded_len, &mut bwt_predictor)?;

            // Undo RLE if applied, then undo BWT+MTF
            let mtf_data = if rle_applied {
                bwt_preprocess::rle_decode(&encode_data, uncompressed_size)?
            } else {
                encode_data
            };

            let original =
                bwt_preprocess::bwt_mtf_decode_parts(primary_index, &mtf_data, uncompressed_size)?;
            if predictor_synced {
                sync_predictor(predictor, &original);
            }
            Ok(original)
        }
        CompressionMethod::Lz77PredictorRans => {
            if compressed_data.len() < 4 {
                return Err(crate::error::AetherError::Decompression(format!(
                    "Lz77PredictorRans payload too short: {} bytes (need ≥4, uncompressed_size={})",
                    compressed_data.len(),
                    uncompressed_size,
                )));
            }
            let lz_len = u32::from_le_bytes(compressed_data[..4].try_into().map_err(|_| {
                AetherError::Decompression("Lz77PredictorRans: truncated lz_len field".into())
            })?) as usize;

            // Bounds check on lz_len from untrusted archive data
            if lz_len > MAX_DECOMPRESSED_BLOCK_SIZE {
                return Err(AetherError::ResourceLimitExceeded(format!(
                    "LZ77 intermediate len {} exceeds safety limit {} (uncompressed_size={})",
                    lz_len, MAX_DECOMPRESSED_BLOCK_SIZE, uncompressed_size,
                )));
            }

            let rc_bytes = &compressed_data[4..];

            let lz_bytes = rans::decode_block(rc_bytes, lz_len, predictor)?;
            let original = lz77_preprocess::lz77_decode(&lz_bytes, uncompressed_size)?;

            // Sync group predictor on the decompressed original data to match
            // the compressor's sync_predictor(predictor, &chunk.data) call.
            // decode_block above reset the predictor; this re-establishes
            // cross-block state symmetry.
            if predictor_synced {
                sync_predictor(predictor, &original);
            }
            Ok(original)
        }
        CompressionMethod::LzPredictorRans => {
            #[cfg(feature = "lz4")]
            {
                if compressed_data.len() < 4 {
                    return Err(crate::error::AetherError::Decompression(format!(
                        "LzPredictorRans payload too short: {} bytes (need ≥4, uncompressed_size={})",
                        compressed_data.len(), uncompressed_size,
                    )));
                }
                let lz_len = u32::from_le_bytes(compressed_data[..4].try_into().map_err(|_| {
                    AetherError::Decompression("LzPredictorRans: truncated lz_len field".into())
                })?) as usize;

                // Bounds check on lz_len from untrusted archive data
                if lz_len > MAX_DECOMPRESSED_BLOCK_SIZE {
                    return Err(AetherError::ResourceLimitExceeded(format!(
                        "LZ4 intermediate len {} exceeds safety limit {} (uncompressed_size={})",
                        lz_len, MAX_DECOMPRESSED_BLOCK_SIZE, uncompressed_size,
                    )));
                }

                let rc_bytes = &compressed_data[4..];

                let lz_bytes = rans::decode_block(rc_bytes, lz_len, predictor)?;
                let original = lz_preprocess::lz_decode(&lz_bytes, uncompressed_size)?;

                // Sync group predictor on decompressed data (see LZ77 note above).
                if predictor_synced {
                    sync_predictor(predictor, &original);
                }
                Ok(original)
            }
            #[cfg(not(feature = "lz4"))]
            {
                Err(AetherError::Decompression(
                    "LzPredictorRans blocks require the 'lz4' feature (disabled at compile time)"
                        .into(),
                ))
            }
        }
        CompressionMethod::PredictorRans => {
            let original = rans::decode_block(compressed_data, uncompressed_size, predictor)?;

            // Sync group predictor on decompressed data (see LZ77 note above).
            if predictor_synced {
                sync_predictor(predictor, &original);
            }
            Ok(original)
        }
        CompressionMethod::Zstd => {
            let data = zstd_fallback::decompress(compressed_data, uncompressed_size)?;
            if predictor_synced {
                sync_predictor(predictor, &data);
            }
            Ok(data)
        }
        CompressionMethod::Store => {
            if compressed_data.len() != uncompressed_size {
                return Err(AetherError::Decompression(format!(
                    "Store block size mismatch: payload is {} bytes but uncompressed_size is {}",
                    compressed_data.len(),
                    uncompressed_size,
                )));
            }
            let data = compressed_data.to_vec();
            if predictor_synced {
                sync_predictor(predictor, &data);
            }
            Ok(data)
        }
    }
}

/// Try Zstd, then Store — used when predictor paths expanded the data.
fn try_zstd_or_store(chunk: &Chunk) -> (CompressionMethod, Vec<u8>) {
    if let Ok(zstd_bytes) = zstd_fallback::compress(&chunk.data) {
        if zstd_bytes.len() < chunk.data.len() {
            return (CompressionMethod::Zstd, zstd_bytes);
        }
    }
    (CompressionMethod::Store, chunk.data.clone())
}

/// Feed data through the predictor to keep cross-block state in sync.
fn sync_predictor(predictor: &mut dyn ProbabilityPredictor, data: &[u8]) {
    for &byte in data {
        predictor.predict();
        predictor.update(byte);
    }
}
