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
//!
//! # Group predictor state
//!
//! Every predictor-backed payload — BWT, LZ77, plain and byte-plane — is
//! range-coded by a **scratch predictor owned by this module**, not by the
//! caller's group predictor, and both `rans::encode_block` and
//! `rans::decode_block` call `reset()` before the first symbol. Cross-block
//! group state is therefore never consumed on either side: a block's coding
//! depends only on that block's bytes and (optionally) the dictionary
//! baseline. The parallel compression path relies on the same property — it
//! builds a fresh predictor per chunk and still produces byte-identical
//! archives.
//!
//! Earlier revisions nonetheless ran a full `predict`/`update` pass over
//! every chunk's plaintext on both sides ("sync_predictor") to keep that
//! unread state symmetric. On the decode side that pass cost one NeuralSSM
//! step per output byte — for `Zstd`, `Store` and `BcjZstd` blocks it was the
//! entire cost of decompression, dwarfing the actual codec. It is gone.
//!
//! [`CompressedChunk::predictor_synced`] and the block header's
//! `predictor_state_flag` are still written and read unchanged, so the
//! on-disk format is untouched and a future predictor that genuinely carries
//! state across blocks can reintroduce the pass without a format break.

use crate::analyzer::{self, RecommendedMethod};
use crate::chunker::ChunkRef;
use crate::coding::byteplane_preprocess;
#[cfg(feature = "lz4")]
use crate::coding::lz_preprocess;
use crate::coding::{bcj, bwt_preprocess, lz77_preprocess, rans, zstd_fallback};
use crate::entropy::{NeuralSsmPredictor, ProbabilityPredictor};
use crate::error::{AetherError, Result};
#[cfg(feature = "bwt-encode")]
use crate::format::BWT_ENTROPY_SKIP;
use crate::format::{
    CompressionMethod, ContentType, BWT_DECISIVE_RATIO, MAX_DECOMPRESSED_BLOCK_SIZE,
};
use crate::pipeline::compress::CompressionProfile;

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
    /// Recorded per-block predictor-sync disposition, written to the block
    /// header as `predictor_state_flag`.
    ///
    /// `false` for methods that code through their own internal predictor
    /// (BWT, byte-plane, BCJ+Zstd). No decode path consumes it today — see
    /// the module-level "Group predictor state" note — but it is preserved
    /// so the archive format is unchanged.
    pub predictor_synced: bool,
}

/// Compress a chunk using the best method based on its entropy.
///
/// Tries multiple transforms and picks the smallest result.
/// Since `encode_block` calls `predictor.reset()` at the start,
/// each attempt starts with clean predictor state.
pub fn compress_chunk(
    chunk: &ChunkRef<'_>,
    predictor: &mut dyn ProbabilityPredictor,
    content_type: crate::format::ContentType,
    profile: CompressionProfile,
) -> Result<CompressedChunk> {
    let fast_route = profile == CompressionProfile::Fast
        || (profile == CompressionProfile::Balanced
            && matches!(
                content_type,
                ContentType::BinaryStructured
                    | ContentType::BinaryRandom
                    | ContentType::Image
                    | ContentType::Executable
            ));

    if fast_route {
        let zstd = zstd_fallback::compress(chunk.data)?;
        let mut best = if zstd.len() < chunk.data.len() {
            (CompressionMethod::Zstd, zstd)
        } else {
            (CompressionMethod::Store, chunk.data.to_vec())
        };
        if content_type == ContentType::Executable {
            if let Some(transformed) = bcj::encode_x86(chunk.data) {
                let compressed = zstd_fallback::compress(&transformed)?;
                if compressed.len() < best.1.len() {
                    best = (CompressionMethod::BcjZstd, compressed);
                }
            }
        }
        return Ok(CompressedChunk {
            method: best.0,
            data: best.1,
            original_size: chunk.length,
            blake3_hash: chunk.blake3_hash,
            predictor_synced: false,
        });
    }

    let method = analyzer::recommend_method_for(chunk.entropy, content_type);
    let bcj_candidate = if content_type == ContentType::Executable {
        bcj::encode_x86(chunk.data).and_then(|transformed| {
            zstd_fallback::compress(&transformed)
                .ok()
                .filter(|compressed| compressed.len() < chunk.data.len())
                .map(|compressed| (CompressionMethod::BcjZstd, compressed))
        })
    } else {
        None
    };

    let mut predictor_synced = true;

    let (compression_method, compressed_data) = match method {
        RecommendedMethod::PredictorRans => {
            let mut best: Option<(CompressionMethod, Vec<u8>)> = None;

            // Single scratch predictor reused across all trial paths.
            // encode_block() calls predictor.reset() at the start of every
            // call, so state from a failed trial never bleeds into the next.
            // Saves 2 heap allocations (~25 KiB each) per chunk.
            let mut scratch = NeuralSsmPredictor::new();
            // Stage A: if the configured predictor carries a dictionary
            // baseline (only NeuralSSM does), propagate it so the BWT/LZ77/
            // plain coding starts each block from the pretrained distribution.
            // decode mirrors this in decompress_chunk via the same dictionary.
            if let Some(baseline) = predictor.coding_baseline() {
                scratch.set_dict_baseline(baseline);
            }

            // ── Try byte-plane splitting (numeric data fast path) ────
            // For NumericData content, try byte-plane first and skip
            // BWT/LZ77 if it wins decisively. For other content types,
            // still try byte-plane as a competing method after BWT/LZ77.
            let is_numeric = content_type == ContentType::NumericData;
            if is_numeric {
                if let Some(width) = byteplane_preprocess::detect_numeric_width(chunk.data) {
                    if let Some(payload) = byteplane_preprocess::byteplane_encode(chunk.data, width)
                    {
                        if payload.len() < chunk.data.len() {
                            best = Some((CompressionMethod::BytePlanePredictorRans, payload));
                        }
                    }
                } else {
                    // Auto-detect failed; try both widths
                    for &width in &[
                        byteplane_preprocess::BytePlaneWidth::Two,
                        byteplane_preprocess::BytePlaneWidth::Four,
                    ] {
                        if byteplane_preprocess::is_byteplane_beneficial(chunk.data, width) {
                            if let Some(payload) =
                                byteplane_preprocess::byteplane_encode(chunk.data, width)
                            {
                                let is_better =
                                    best.as_ref().is_none_or(|(_, b)| payload.len() < b.len());
                                if payload.len() < chunk.data.len() && is_better {
                                    best =
                                        Some((CompressionMethod::BytePlanePredictorRans, payload));
                                }
                            }
                        }
                    }
                }
            }

            // ── Try BWT+MTF+RLE → predictor + range coding ────────────
            // BWT clusters context; MTF converts to small integers; RLE
            // compacts zero runs using bijective base-2 (RUNA/RUNB).
            // The predictor sees the RLE stream, not the raw MTF stream.
            // bwt_mtf_encode_parts returns Err if input exceeds MAX_BWT_INPUT_SIZE;
            // we treat that as "BWT not applicable" and fall through to LZ77/plain.
            //
            // Skip BWT for high-entropy chunks — suffix array construction
            // is expensive and BWT clustering provides minimal benefit on
            // near-random data.  Text is typically 4-5 bps.
            // The shared threshold is also used by transformed dictionary
            // training so the two paths cannot silently diverge.
            //
            // Gated on `bwt-encode`: the forward transform needs libsais.
            // A decompress-only build (wasm) still *decodes* BwtPredictorRans
            // blocks — only the encode-side trial is unavailable, so the
            // cascade falls through to LZ77/plain/Zstd.
            #[cfg(feature = "bwt-encode")]
            if chunk.data.len() >= 8 && chunk.entropy < BWT_ENTROPY_SKIP {
                if let Ok((primary_index, mtf_data)) =
                    bwt_preprocess::bwt_mtf_encode_parts(chunk.data)
                {
                    // Try RLE first (much more compact); fall back to raw MTF
                    let (encode_data, rle_applied) =
                        if let Some(rle) = bwt_preprocess::rle_encode(&mtf_data) {
                            (rle, true)
                        } else {
                            (mtf_data, false)
                        };

                    if let Ok(rc_bytes) = rans::encode_block(&encode_data, &mut scratch) {
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
                if let Some(lz_bytes) = lz77_preprocess::lz77_encode(chunk.data) {
                    // Reuse scratch predictor (encode_block resets it first).
                    if let Ok(rc_bytes) = rans::encode_block(&lz_bytes, &mut scratch) {
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

            // ── Try byte-plane splitting (non-numeric fallback) ─────
            // For non-text, non-numeric content, try byte-plane as a
            // competing method. Executables and structured binary can
            // contain embedded float tables that benefit from splitting.
            if !is_numeric && content_type != ContentType::Text {
                if let Some(width) = byteplane_preprocess::detect_numeric_width(chunk.data) {
                    if let Some(payload) = byteplane_preprocess::byteplane_encode(chunk.data, width)
                    {
                        let is_better = best.as_ref().is_none_or(|(_, b)| payload.len() < b.len());
                        if payload.len() < chunk.data.len() && is_better {
                            best = Some((CompressionMethod::BytePlanePredictorRans, payload));
                        }
                    }
                }
            }

            // ── Try plain predictor + range coding ───────────────────
            // Reuse scratch predictor (encode_block resets it first).
            if best.is_none() {
                if let Ok(rc_bytes) = rans::encode_block(chunk.data, &mut scratch) {
                    if rc_bytes.len() < chunk.data.len() {
                        best = Some((CompressionMethod::PredictorRans, rc_bytes));
                    }
                }
            }

            // ── Record the winning path's sync disposition ────────────
            //
            // See the module-level "Group predictor state" note: every
            // predictor-backed payload is coded by a scratch predictor that
            // `encode_block`/`decode_block` reset per block, so no cross-block
            // group state is ever consumed. `predictor_synced` is still
            // recorded (and written to the block header as
            // `predictor_state_flag`) so the on-disk format is unchanged and a
            // future predictor that *does* carry state across blocks can be
            // reintroduced without a format break.
            if let Some((method, payload)) = best {
                if matches!(
                    method,
                    CompressionMethod::BwtPredictorRans | CompressionMethod::BytePlanePredictorRans
                ) {
                    // BWT and byte-plane both code through their own internal
                    // predictors, so the group predictor's state is not
                    // meaningful for them.
                    predictor_synced = false;
                }
                (method, payload)
            } else {
                // Nothing helped — fall back to zstd/store
                try_zstd_or_store(chunk)
            }
        }
        RecommendedMethod::Zstd => {
            let compressed = zstd_fallback::compress(chunk.data)?;
            if compressed.len() >= chunk.data.len() {
                (CompressionMethod::Store, chunk.data.to_vec())
            } else {
                (CompressionMethod::Zstd, compressed)
            }
        }
        RecommendedMethod::Store => (CompressionMethod::Store, chunk.data.to_vec()),
    };

    let (compression_method, compressed_data) = if let Some((method, data)) = bcj_candidate {
        if data.len() < compressed_data.len() {
            predictor_synced = false;
            (method, data)
        } else {
            (compression_method, compressed_data)
        }
    } else {
        (compression_method, compressed_data)
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
/// `predictor` supplies only the dictionary coding baseline; the payload
/// itself is decoded by a scratch predictor built to match the encoder. See
/// the module-level "Group predictor state" note. Callers that already know
/// the baseline should use [`decompress_chunk_with_baseline`] instead.
///
/// `predictor_synced` mirrors the block header's `predictor_state_flag`. No
/// decode path consumes it today — it is accepted so callers keep passing the
/// archive's value through, which keeps the door open for a future predictor
/// with genuine cross-block state.
pub fn decompress_chunk(
    compressed_data: &[u8],
    method: CompressionMethod,
    uncompressed_size: usize,
    predictor: &mut dyn ProbabilityPredictor,
    _predictor_synced: bool,
) -> Result<Vec<u8>> {
    decompress_chunk_with_baseline(
        compressed_data,
        method,
        uncompressed_size,
        predictor.coding_baseline(),
    )
}

/// Decompress a chunk, given the dictionary coding baseline directly.
///
/// This is what [`decompress_chunk`] does after reading the one thing it
/// needs from the predictor it is handed. Decoding depends on the payload,
/// the method, the expected size and the baseline — nothing else — so
/// callers that decode many blocks (in parallel, or across solid groups) can
/// resolve the baseline once instead of constructing a predictor per worker.
/// That matters: a `ContextMixer` instance is ~100 MiB.
///
/// The baseline is `predictor.coding_baseline()`, i.e. `Some(dict.state)`
/// only when a dictionary is configured *and* the archive's predictor type
/// installs one (today: NeuralSSM). [`Decompressor::decode_baseline`] computes
/// it once.
///
/// # Safety Limits
///
/// Rejects `uncompressed_size` exceeding [`MAX_DECOMPRESSED_BLOCK_SIZE`] (64 MiB)
/// to prevent out-of-memory from crafted archives.
///
/// [`Decompressor::decode_baseline`]: crate::pipeline::decompress::Decompressor
pub fn decompress_chunk_with_baseline(
    compressed_data: &[u8],
    method: CompressionMethod,
    uncompressed_size: usize,
    dict_baseline: Option<&[u8]>,
) -> Result<Vec<u8>> {
    // Bounds check: reject implausibly large decompressed sizes
    if uncompressed_size > MAX_DECOMPRESSED_BLOCK_SIZE {
        return Err(AetherError::ResourceLimitExceeded(format!(
            "Decompressed block size {} exceeds maximum {} bytes",
            uncompressed_size, MAX_DECOMPRESSED_BLOCK_SIZE,
        )));
    }

    match method {
        CompressionMethod::BcjZstd => {
            let mut original = zstd_fallback::decompress(compressed_data, uncompressed_size)?;
            bcj::decode_x86(&mut original);
            Ok(original)
        }
        CompressionMethod::BytePlanePredictorRans => {
            let original =
                byteplane_preprocess::byteplane_decode(compressed_data, uncompressed_size)?;
            Ok(original)
        }
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

            // Stage A: mirror the encoder — seed the BWT decode predictor with
            // the same dictionary baseline the encoder used, so reset() (called
            // inside decode_block) restores the identical starting state.
            let mut bwt_predictor = coding_predictor(dict_baseline);
            let encode_data = rans::decode_block(rc_bytes, encoded_len, &mut bwt_predictor)?;

            // Undo RLE if applied, then undo BWT+MTF
            let mtf_data = if rle_applied {
                bwt_preprocess::rle_decode(&encode_data, uncompressed_size)?
            } else {
                encode_data
            };

            let original =
                bwt_preprocess::bwt_mtf_decode_parts(primary_index, &mtf_data, uncompressed_size)?;
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

            let mut lz_predictor = coding_predictor(dict_baseline);
            let lz_bytes = rans::decode_block(rc_bytes, lz_len, &mut lz_predictor)?;
            let original = lz77_preprocess::lz77_decode(&lz_bytes, uncompressed_size)?;
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

                let mut lz_predictor = coding_predictor(dict_baseline);
                let lz_bytes = rans::decode_block(rc_bytes, lz_len, &mut lz_predictor)?;
                let original = lz_preprocess::lz_decode(&lz_bytes, uncompressed_size)?;
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
            let mut plain_predictor = coding_predictor(dict_baseline);
            let original =
                rans::decode_block(compressed_data, uncompressed_size, &mut plain_predictor)?;
            Ok(original)
        }
        CompressionMethod::Zstd => zstd_fallback::decompress(compressed_data, uncompressed_size),
        CompressionMethod::Store => {
            if compressed_data.len() != uncompressed_size {
                return Err(AetherError::Decompression(format!(
                    "Store block size mismatch: payload is {} bytes but uncompressed_size is {}",
                    compressed_data.len(),
                    uncompressed_size,
                )));
            }
            Ok(compressed_data.to_vec())
        }
    }
}

/// Try Zstd, then Store — used when predictor paths expanded the data.
fn try_zstd_or_store(chunk: &ChunkRef<'_>) -> (CompressionMethod, Vec<u8>) {
    if let Ok(zstd_bytes) = zstd_fallback::compress(chunk.data) {
        if zstd_bytes.len() < chunk.data.len() {
            return (CompressionMethod::Zstd, zstd_bytes);
        }
    }
    (CompressionMethod::Store, chunk.data.to_vec())
}

/// Build the coding predictor the *encoder* used for a predictor-backed
/// block payload.
///
/// [`compress_chunk`] range-codes every predictor path (BWT, LZ77, plain,
/// byte-plane) with a scratch [`NeuralSsmPredictor`] rather than with the
/// group predictor — it has always done so, since the first release. The
/// decoder must therefore build the *same* predictor, not the group
/// predictor, or the two sides disagree on the CDF for every symbol.
///
/// `dict_baseline` mirrors `compress_chunk`'s `scratch.set_dict_baseline(...)`,
/// so `reset()` inside `decode_block` restores the identical starting state.
fn coding_predictor(dict_baseline: Option<&[u8]>) -> NeuralSsmPredictor {
    let mut predictor = NeuralSsmPredictor::new();
    if let Some(baseline) = dict_baseline {
        predictor.set_dict_baseline(baseline);
    }
    predictor
}
