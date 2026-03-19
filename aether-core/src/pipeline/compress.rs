//! High-level compression pipeline: files → .aet archive.

use std::borrow::Cow;
use std::collections::HashMap;
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

#[cfg(feature = "threading")]
use rayon::prelude::*;

use crate::analyzer;
use crate::block::{BlockHeader, BlockIndexEntry, BlockTrailer};
use crate::chunker;
use crate::entropy::ProbabilityPredictor;
use crate::error::{AetherError, Result};
use crate::format::*;
use crate::grouper::{self, FileInfo};
use crate::header::{ArchiveFooter, ArchiveHeader, FileEntry, SolidGroupEntry};
use crate::pipeline::router;

#[cfg(feature = "enterprise")]
use crate::crypto;

/// Compressed blocks for one file, produced by the parallel compression phase.
struct CompressedFileBlocks {
    file_idx: usize,
    chunk_count: usize,
    blocks: Vec<router::CompressedChunk>,
}

/// Compresses files into an AetherArch archive.
///
/// # Memory Backpressure
///
/// Parallel compression across solid groups can use significant memory:
/// each group creates its own predictor instance, and all compressed
/// blocks are buffered before writing. The `max_threads` parameter
/// limits how many groups are compressed concurrently, capping peak
/// memory at roughly `max_threads × (predictor_size + group_data)`.
///
/// Default is 4 concurrent groups. Use [`Compressor::with_max_threads`]
/// to tune for available memory:
/// - Low memory (< 2 GiB): `max_threads = 1-2`
/// - Default (2-8 GiB): `max_threads = 4`
/// - High memory (> 8 GiB): `max_threads = 8+` or `0` for unlimited
///
/// Encryption configuration for the compressor (enterprise feature).
///
/// Uses `zeroize::Zeroizing` to securely erase the password from memory on drop,
/// preventing recovery via crash dumps or memory disclosure.
#[cfg(feature = "enterprise")]
struct EncryptionConfig {
    password: zeroize::Zeroizing<String>,
    cipher_id: crypto::CipherId,
}

pub struct Compressor {
    predictor_factory: Box<dyn Fn() -> Box<dyn ProbabilityPredictor> + Send + Sync>,
    predictor_id: PredictorId,
    /// Maximum number of concurrent compression threads.
    /// 0 means use all available cores (rayon default).
    max_threads: usize,
    /// Encryption configuration (enterprise feature).
    #[cfg(feature = "enterprise")]
    encryption_config: Option<EncryptionConfig>,
    /// Dictionary for predictor pretraining.
    dictionary: Option<crate::dictionary::Dictionary>,
}

/// Default maximum concurrent groups for memory backpressure.
const DEFAULT_MAX_THREADS: usize = 4;

impl Compressor {
    /// Create a new compressor with the given predictor factory.
    ///
    /// The factory is called to create fresh predictor instances for each block.
    /// It must be `Send + Sync` so it can be shared across rayon worker threads
    /// for parallel group compression.
    ///
    /// Uses `DEFAULT_MAX_THREADS` (4) concurrent groups by default.
    pub fn new<F>(factory: F) -> Self
    where
        F: Fn() -> Box<dyn ProbabilityPredictor> + Send + Sync + 'static,
    {
        let sample = factory();
        let pid = sample.predictor_id();
        Self {
            predictor_factory: Box::new(factory),
            predictor_id: pid,
            max_threads: DEFAULT_MAX_THREADS,
            #[cfg(feature = "enterprise")]
            encryption_config: None,
            dictionary: None,
        }
    }

    /// Set a pretrained dictionary to initialize predictors from.
    ///
    /// Each predictor instance will be initialized with the dictionary's
    /// learned state before compressing its solid group. The dictionary's
    /// BLAKE3 hash is stored in the archive header so the decompressor
    /// can verify it has the matching dictionary.
    pub fn with_dictionary(mut self, dict: crate::dictionary::Dictionary) -> Self {
        self.dictionary = Some(dict);
        self
    }

    /// Set the maximum number of concurrent compression threads (builder pattern).
    ///
    /// Limits peak memory by restricting how many solid groups compress
    /// simultaneously. Set to 0 for unlimited (uses all available cores).
    pub fn with_max_threads(mut self, max_threads: usize) -> Self {
        self.max_threads = max_threads;
        self
    }

    /// Set the maximum number of concurrent compression threads (mutable reference).
    ///
    /// Same as [`with_max_threads`](Self::with_max_threads) but takes `&mut self`
    /// for use when the `Compressor` is already constructed (e.g. behind an FFI handle).
    pub fn set_max_threads(&mut self, max_threads: usize) {
        self.max_threads = max_threads;
    }

    /// Enable encryption for the output archive (enterprise feature).
    ///
    /// Each compressed block is encrypted with the specified cipher after
    /// compression, preserving random-access extraction capability.
    /// The password is stretched with Argon2id (64 MiB, 3 iterations).
    #[cfg(feature = "enterprise")]
    pub fn with_encryption(mut self, password: &str, cipher_id: crypto::CipherId) -> Self {
        self.encryption_config = Some(EncryptionConfig {
            password: zeroize::Zeroizing::new(password.to_string()),
            cipher_id,
        });
        self
    }

    /// Compress a list of file paths into an archive written to `output`.
    ///
    /// Returns basic [`CompressionStats`] and optional [`CompressionAnalytics`]
    /// with per-method and per-group breakdowns.
    pub fn compress_to_archive<W: Write + Seek>(
        &self,
        base_dir: &Path,
        file_paths: &[PathBuf],
        output: &mut W,
    ) -> Result<(CompressionStats, CompressionAnalytics)> {
        let mut stats = CompressionStats::default();

        // ── Step 1: Scan files ───────────────────────────────────────────
        let mut file_entries = Vec::new();
        let mut file_infos = Vec::new();
        let mut file_datas: Vec<Vec<u8>> = Vec::new();

        let mut cumulative_input_size: u64 = 0;

        for path in file_paths {
            let data = std::fs::read(path)?;

            // Track cumulative input to prevent unbounded memory consumption
            cumulative_input_size = cumulative_input_size.saturating_add(data.len() as u64);
            if cumulative_input_size > MAX_TOTAL_INPUT_SIZE {
                return Err(AetherError::ResourceLimitExceeded(format!(
                    "Total input size {} exceeds limit of {} bytes",
                    cumulative_input_size, MAX_TOTAL_INPUT_SIZE,
                )));
            }

            let rel_path = path
                .strip_prefix(base_dir)
                .unwrap_or(path)
                .to_str()
                .ok_or(AetherError::InvalidUtf8Path)?
                .replace('\\', "/");

            let hash = blake3::hash(&data);
            let content_type = analyzer::detect_content_type(&rel_path, &data);
            let entropy = crate::format::shannon_entropy(&data);

            // Read mtime from filesystem metadata
            let mtime = std::fs::metadata(path)
                .and_then(|m| m.modified())
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_secs() as i64)
                .unwrap_or(0);

            file_infos.push(FileInfo {
                path: rel_path.clone(),
                size: data.len() as u64,
                content_type,
                mean_entropy: entropy,
            });

            // Read actual file permissions on Unix; default to 0o644 elsewhere.
            let permissions = {
                #[cfg(unix)]
                {
                    use std::os::unix::fs::PermissionsExt;
                    std::fs::metadata(path)
                        .map(|m| m.permissions().mode() & SAFE_PERMISSION_MASK)
                        .unwrap_or(0o644)
                }
                #[cfg(not(unix))]
                {
                    0o644
                }
            };

            file_entries.push(FileEntry {
                path: rel_path,
                original_size: data.len() as u64,
                blake3_hash: *hash.as_bytes(),
                solid_group_id: 0, // Will be set later
                chunk_start_idx: 0,
                chunk_count: 0,
                permissions,
                mtime,
            });

            stats.original_size += data.len() as u64;
            file_datas.push(data);
        }

        // ── Step 2: Group files ──────────────────────────────────────────
        let groups = grouper::group_files(&file_infos);

        // Update file entries with group assignments
        for group in &groups {
            for &file_idx in &group.file_indices {
                file_entries[file_idx].solid_group_id = group.group_id;
            }
        }

        // ── Step 3: Write placeholder header ─────────────────────────────
        let mut flags = FLAG_SOLID_ARCHIVE;

        if self.dictionary.is_some() {
            flags |= FLAG_HAS_DICTIONARY;
        }

        // Set up encryption state if configured (enterprise feature)
        #[cfg(feature = "enterprise")]
        let encryption_state: Option<(crypto::EncryptionHeader, crypto::DerivedKey)> =
            if let Some(ref config) = self.encryption_config {
                flags |= FLAG_ENCRYPTED;
                crypto::validate_encryption_password(config.password.as_bytes())?;
                let salt = crypto::generate_salt();
                let nonce = crypto::generate_nonce();
                let key = crypto::derive_key(
                    config.password.as_bytes(),
                    &salt,
                    crypto::DEFAULT_ARGON2_M_COST,
                    crypto::DEFAULT_ARGON2_T_COST,
                    crypto::DEFAULT_ARGON2_P_COST,
                )?;
                let verification_tag = crypto::compute_verification_tag(&key);
                let enc_header = crypto::EncryptionHeader {
                    version: crypto::HEADER_VERSION,
                    cipher_id: config.cipher_id,
                    salt,
                    m_cost: crypto::DEFAULT_ARGON2_M_COST,
                    t_cost: crypto::DEFAULT_ARGON2_T_COST,
                    p_cost: crypto::DEFAULT_ARGON2_P_COST,
                    master_nonce: nonce,
                    verification_tag,
                };
                Some((enc_header, key))
            } else {
                None
            };

        let file_count_u32 = u32::try_from(file_entries.len()).map_err(|_| {
            AetherError::Compression(format!(
                "File count {} exceeds u32::MAX",
                file_entries.len(),
            ))
        })?;
        let group_count_u32 = u32::try_from(groups.len()).map_err(|_| {
            AetherError::Compression(format!(
                "Solid group count {} exceeds u32::MAX",
                groups.len(),
            ))
        })?;

        let placeholder_header = ArchiveHeader {
            flags,
            predictor_id: self.predictor_id,
            file_count: file_count_u32,
            solid_group_count: group_count_u32,
            block_count: 0,        // patched later
            file_table_offset: 0,  // patched later
            block_index_offset: 0, // patched later
        };
        placeholder_header.write_to(output)?;

        // Write encryption header if encrypted (after the 48-byte archive header)
        #[cfg(feature = "enterprise")]
        if let Some((ref enc_header, _)) = encryption_state {
            enc_header.write_to(output)?;
        }

        // Write dictionary hash if using dictionary (32 bytes)
        if let Some(ref dict) = self.dictionary {
            output.write_all(&dict.hash)?;
        }

        // ── Step 4: Write file table ─────────────────────────────────────
        let file_table_offset = output.stream_position().map_err(AetherError::Io)?;

        // We'll need to patch chunk_start_idx and chunk_count later
        let file_table_pos = file_table_offset;
        for entry in &file_entries {
            entry.write_to(output)?;
        }

        // ── Step 5: Write solid group table ──────────────────────────────
        let mut solid_group_entries: Vec<SolidGroupEntry> = groups
            .iter()
            .map(|g| SolidGroupEntry {
                group_id: g.group_id,
                content_type: g.content_type,
                compression_method: match g.recommended_method {
                    crate::analyzer::RecommendedMethod::PredictorRans => {
                        CompressionMethod::PredictorRans
                    }
                    crate::analyzer::RecommendedMethod::Zstd => CompressionMethod::Zstd,
                    crate::analyzer::RecommendedMethod::Store => CompressionMethod::Store,
                },
                first_block_idx: 0, // patched later
                block_count: 0,     // patched later
                file_count: g.file_indices.len() as u32,
            })
            .collect();

        let solid_group_table_pos = output.stream_position().map_err(AetherError::Io)?;
        for entry in &solid_group_entries {
            entry.write_to(output)?;
        }

        // ── Step 6: Compress blocks (two-phase parallel) ─────────────────
        //
        // Phase A — parallel compression across groups:
        //   Each solid group has an independent predictor, so groups can be
        //   compressed concurrently via rayon without any shared mutable state.
        //   Results are buffered in memory before writing.
        //
        // Phase B — sequential write:
        //   Blocks are written to the archive in deterministic group order so
        //   the byte layout is reproducible regardless of thread scheduling.
        let mut all_block_indices = Vec::new();
        let mut block_id_counter = 0u32;
        let mut global_chunk_idx = 0u32;

        // ── Phase A: parallel compression ────────────────────────────────
        //
        // Memory backpressure: use a scoped thread pool with bounded threads
        // to limit how many groups compress simultaneously.  Each group
        // creates its own predictor + buffers compressed blocks in memory,
        // so bounding concurrency caps peak memory at roughly
        // max_threads × (predictor_size + group_data).
        let compression_start = Instant::now();
        let compress_group =
            |group: &crate::grouper::SolidGroup| -> Result<Vec<CompressedFileBlocks>> {
                // One predictor per group — created inside the closure so each
                // rayon worker thread gets its own instance (no sharing).
                let mut predictor = (self.predictor_factory)();
                // Apply dictionary state if configured — propagate failure to
                // prevent silently compressing with wrong predictor state.
                if let Some(ref dict) = self.dictionary {
                    dict.apply(predictor.as_mut()).map_err(|e| {
                        AetherError::Compression(format!(
                            "Failed to apply dictionary to predictor: {e}"
                        ))
                    })?;
                }
                let mut results = Vec::with_capacity(group.file_indices.len());

                for &file_idx in &group.file_indices {
                    let file_data = &file_datas[file_idx];

                    let chunks = if file_data.len() < chunker::MIN_CHUNK_SIZE as usize {
                        chunker::chunk_fixed(file_data, chunker::AVG_CHUNK_SIZE as usize)
                    } else {
                        chunker::chunk_data(file_data)
                    };
                    let chunk_count = chunks.len();

                    let mut blocks = Vec::with_capacity(chunk_count);
                    for chunk in &chunks {
                        blocks.push(router::compress_chunk(
                            chunk,
                            predictor.as_mut(),
                            group.content_type,
                        )?);
                    }

                    results.push(CompressedFileBlocks {
                        file_idx,
                        chunk_count,
                        blocks,
                    });
                }
                Ok(results)
            };

        #[cfg(feature = "threading")]
        let compressed_groups: Vec<Vec<CompressedFileBlocks>> = if self.max_threads == 0 {
            // Unlimited: use the global rayon pool (all cores)
            groups
                .par_iter()
                .map(compress_group)
                .collect::<Result<Vec<_>>>()?
        } else {
            // Bounded: create a scoped thread pool with limited threads
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(self.max_threads)
                .build()
                .map_err(|e| {
                    AetherError::Compression(format!(
                        "Failed to create thread pool with {} threads: {e}",
                        self.max_threads,
                    ))
                })?;
            pool.install(|| {
                groups
                    .par_iter()
                    .map(compress_group)
                    .collect::<Result<Vec<_>>>()
            })?
        };
        #[cfg(not(feature = "threading"))]
        let compressed_groups: Vec<Vec<CompressedFileBlocks>> = groups
            .iter()
            .map(compress_group)
            .collect::<Result<Vec<_>>>()?;

        let compression_time = compression_start.elapsed();

        // ── Phase B: sequential write ─────────────────────────────────────
        let write_start = Instant::now();
        let mut analytics_method_counts: HashMap<CompressionMethod, u32> = HashMap::new();
        let mut analytics_method_orig: HashMap<CompressionMethod, u64> = HashMap::new();
        let mut analytics_method_comp: HashMap<CompressionMethod, u64> = HashMap::new();
        let mut analytics_groups: Vec<GroupAnalytics> = Vec::with_capacity(groups.len());

        for (group_idx, file_results) in compressed_groups.iter().enumerate() {
            let group_first_block = block_id_counter;
            let mut group_orig_size = 0u64;
            let mut group_comp_size = 0u64;
            let mut group_method_counts: HashMap<CompressionMethod, u32> = HashMap::new();
            let mut group_block_count = 0u32;

            for fb in file_results {
                // Assign chunk range to this file (must be sequential)
                file_entries[fb.file_idx].chunk_start_idx = global_chunk_idx;
                file_entries[fb.file_idx].chunk_count =
                    u32::try_from(fb.chunk_count).map_err(|_| {
                        AetherError::Compression(format!(
                            "File {} chunk count {} exceeds u32::MAX",
                            fb.file_idx, fb.chunk_count,
                        ))
                    })?;

                for compressed in &fb.blocks {
                    let block_offset = output.stream_position().map_err(AetherError::Io)?;

                    // Encrypt block data if encryption is enabled
                    let block_data: Cow<[u8]> = {
                        #[cfg(feature = "enterprise")]
                        {
                            if let Some((ref enc_header, ref derived_key)) = encryption_state {
                                Cow::Owned(crypto::encrypt_block(
                                    enc_header.cipher_id,
                                    derived_key.as_bytes(),
                                    &enc_header.master_nonce,
                                    block_id_counter,
                                    &compressed.data,
                                )?)
                            } else {
                                Cow::Borrowed(&compressed.data)
                            }
                        }
                        #[cfg(not(feature = "enterprise"))]
                        {
                            Cow::Borrowed(&compressed.data)
                        }
                    };

                    let block_compressed_size = u32::try_from(block_data.len()).map_err(|_| {
                        AetherError::Compression(format!(
                            "Block {} compressed size {} exceeds u32::MAX",
                            block_id_counter,
                            block_data.len(),
                        ))
                    })?;
                    let block_uncompressed_size =
                        u32::try_from(compressed.original_size).map_err(|_| {
                            AetherError::Compression(format!(
                                "Block {} uncompressed size {} exceeds u32::MAX",
                                block_id_counter, compressed.original_size,
                            ))
                        })?;

                    let block_header = BlockHeader {
                        block_id: block_id_counter,
                        solid_group_id: groups[group_idx].group_id,
                        compression_method: compressed.method,
                        predictor_state_flag: !compressed.predictor_synced,
                        compressed_size: block_compressed_size,
                        uncompressed_size: block_uncompressed_size,
                    };

                    let block_trailer = BlockTrailer {
                        content_blake3: compressed.blake3_hash,
                    };

                    block_header.write_to(output)?;
                    output.write_all(&block_data)?;
                    block_trailer.write_to(output)?;

                    let total_block_size = BLOCK_HEADER_SIZE as u64
                        + block_data.len() as u64
                        + BLOCK_TRAILER_SIZE as u64;

                    all_block_indices.push(BlockIndexEntry {
                        block_id: block_id_counter,
                        archive_offset: block_offset,
                        compressed_size: u32::try_from(total_block_size).map_err(|_| {
                            AetherError::Compression(format!(
                                "Block {} total size {} exceeds u32::MAX",
                                block_id_counter, total_block_size,
                            ))
                        })?,
                        uncompressed_size: block_uncompressed_size,
                        solid_group_id: groups[group_idx].group_id,
                    });

                    stats.compressed_size += block_data.len() as u64;

                    // Analytics: track per-method and per-group stats
                    *analytics_method_counts
                        .entry(compressed.method)
                        .or_insert(0) += 1;
                    *analytics_method_orig.entry(compressed.method).or_insert(0) +=
                        compressed.original_size as u64;
                    *analytics_method_comp.entry(compressed.method).or_insert(0) +=
                        block_data.len() as u64;
                    *group_method_counts.entry(compressed.method).or_insert(0) += 1;
                    group_orig_size += compressed.original_size as u64;
                    group_comp_size += block_data.len() as u64;
                    group_block_count += 1;

                    block_id_counter = block_id_counter.checked_add(1).ok_or_else(|| {
                        AetherError::ResourceLimitExceeded(
                            "Block ID counter overflow (u32::MAX blocks)".into(),
                        )
                    })?;
                }

                global_chunk_idx += file_entries[fb.file_idx].chunk_count;
            }

            // Update solid group entry
            solid_group_entries[group_idx].first_block_idx = group_first_block;
            solid_group_entries[group_idx].block_count = block_id_counter - group_first_block;

            analytics_groups.push(GroupAnalytics {
                group_id: groups[group_idx].group_id,
                content_type: groups[group_idx].content_type,
                block_count: group_block_count,
                original_size: group_orig_size,
                compressed_size: group_comp_size,
                method_counts: group_method_counts,
            });
        }

        // ── Step 7: Write block index ────────────────────────────────────
        let block_index_offset = output.stream_position().map_err(AetherError::Io)?;

        for entry in &all_block_indices {
            entry.write_to(output)?;
        }

        // ── Step 8: Write footer ─────────────────────────────────────────
        let footer = ArchiveFooter {
            block_index_offset,
            file_table_offset,
            block_count: block_id_counter,
            file_count: file_count_u32,
        };
        footer.write_to(output)?;

        // ── Step 9: Patch header with final offsets ──────────────────────
        let final_header = ArchiveHeader {
            flags,
            predictor_id: self.predictor_id,
            file_count: file_count_u32,
            solid_group_count: group_count_u32,
            block_count: block_id_counter,
            file_table_offset,
            block_index_offset,
        };

        output.seek(SeekFrom::Start(0)).map_err(AetherError::Io)?;
        final_header.write_to(output)?;

        // ── Step 10: Re-write file table with correct chunk indices ──────
        output
            .seek(SeekFrom::Start(file_table_pos))
            .map_err(AetherError::Io)?;
        for entry in &file_entries {
            entry.write_to(output)?;
        }

        // Re-write solid group table with correct block indices
        output
            .seek(SeekFrom::Start(solid_group_table_pos))
            .map_err(AetherError::Io)?;
        for entry in &solid_group_entries {
            entry.write_to(output)?;
        }

        let write_time = write_start.elapsed();

        stats.block_count = block_id_counter;
        stats.file_count = file_count_u32;
        stats.group_count = group_count_u32;

        let analytics = CompressionAnalytics {
            method_counts: analytics_method_counts,
            method_bytes_original: analytics_method_orig,
            method_bytes_compressed: analytics_method_comp,
            group_stats: analytics_groups,
            compression_time,
            write_time,
        };

        Ok((stats, analytics))
    }
}

/// Statistics from a compression operation.
#[derive(Debug, Default)]
pub struct CompressionStats {
    pub original_size: u64,
    pub compressed_size: u64,
    pub block_count: u32,
    pub file_count: u32,
    pub group_count: u32,
}

impl CompressionStats {
    pub fn ratio(&self) -> f64 {
        if self.original_size == 0 {
            return 1.0;
        }
        self.compressed_size as f64 / self.original_size as f64
    }

    pub fn bits_per_byte(&self) -> f64 {
        self.ratio() * 8.0
    }
}

/// Per-group analytics collected during compression.
#[derive(Debug, Clone)]
pub struct GroupAnalytics {
    /// Solid group ID.
    pub group_id: u32,
    /// Content type of the group.
    pub content_type: ContentType,
    /// Number of blocks in this group.
    pub block_count: u32,
    /// Total original (uncompressed) size of all blocks in this group.
    pub original_size: u64,
    /// Total compressed size of all blocks in this group.
    pub compressed_size: u64,
    /// Per-method block counts within this group.
    pub method_counts: HashMap<CompressionMethod, u32>,
}

/// Detailed analytics from a compression operation.
///
/// Provides per-method and per-group breakdowns beyond what
/// [`CompressionStats`] offers. Enable with `--analytics` in the CLI.
#[derive(Debug, Clone)]
pub struct CompressionAnalytics {
    /// Number of blocks compressed with each method.
    pub method_counts: HashMap<CompressionMethod, u32>,
    /// Total original bytes routed to each method.
    pub method_bytes_original: HashMap<CompressionMethod, u64>,
    /// Total compressed bytes produced by each method.
    pub method_bytes_compressed: HashMap<CompressionMethod, u64>,
    /// Per-group analytics.
    pub group_stats: Vec<GroupAnalytics>,
    /// Time spent in Phase A (parallel compression of all groups).
    pub compression_time: Duration,
    /// Time spent in Phase B (sequential archive writing).
    pub write_time: Duration,
}
