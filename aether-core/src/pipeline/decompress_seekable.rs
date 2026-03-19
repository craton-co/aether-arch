//! Seekable decompression: `Read + Seek` path for random-access extraction.
//!
//! Uses the footer + block index to seek directly to any block. Supports
//! single-file extraction (`extract_file`).

use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use crate::block::{BlockHeader, BlockIndexEntry, BlockTrailer};
use crate::entropy::ProbabilityPredictor;
use crate::error::{AetherError, Result};
use crate::format::*;
use crate::header::{ArchiveFooter, FileEntry, SolidGroupEntry};
use crate::pipeline::router;

use super::decompress::{
    derive_decrypt_key, hex_str, maybe_decrypt_payload, reassemble_file_from_blocks,
    write_validated_file, ArchiveMetadata, Decompressor, VerificationResult,
};

#[cfg(feature = "enterprise")]
use crate::crypto;
#[cfg(feature = "enterprise")]
use std::collections::HashMap;

// ── Parallel decompression types (enterprise) ─────────────────────────────────

/// A block's raw data read from disk before decompression.
///
/// Used by the parallel decompression path to separate sequential I/O
/// from CPU-bound decompression (which can be parallelized across groups).
#[cfg(feature = "enterprise")]
struct RawBlock {
    /// Position in the global `decompressed_blocks` array.
    global_idx: usize,
    /// Compressed (and decrypted, if applicable) payload bytes.
    payload: Vec<u8>,
    /// Compression method from the block header.
    compression_method: CompressionMethod,
    /// Original uncompressed size in bytes.
    uncompressed_size: usize,
    /// Whether predictor sync was skipped during compression.
    predictor_state_flag: bool,
    /// BLAKE3 hash from the block trailer, for integrity verification.
    content_blake3: [u8; 32],
    /// Block ID (for error reporting and ordering within a group).
    block_id: u32,
    /// Archive byte offset (for error reporting).
    archive_offset: u64,
    /// Solid group this block belongs to.
    solid_group_id: u32,
}

/// Decompress a group of blocks sequentially with a shared predictor.
///
/// Blocks must already be sorted by `block_id` so the predictor state
/// evolves in the same order as during compression.
#[cfg(feature = "enterprise")]
fn decompress_group(
    blocks: &[RawBlock],
    predictor: &mut dyn ProbabilityPredictor,
) -> Result<Vec<(usize, Vec<u8>)>> {
    let mut results = Vec::with_capacity(blocks.len());
    for block in blocks {
        let decompressed = router::decompress_chunk(
            &block.payload,
            block.compression_method,
            block.uncompressed_size,
            predictor,
            !block.predictor_state_flag,
        )
        .map_err(|e| {
            AetherError::Decompression(format!(
                "Block {} (offset {:#x}, group {}, method {:?}): {}",
                block.block_id,
                block.archive_offset,
                block.solid_group_id,
                block.compression_method,
                e,
            ))
        })?;

        // Validate that actual decompressed size matches the claimed size.
        // A crafted archive could set a small uncompressed_size in the header
        // to pass the pre-check, while the decompression codec produces a
        // much larger output, causing OOM in the parallel path.
        if decompressed.len() != block.uncompressed_size {
            return Err(AetherError::Decompression(format!(
                "Block {} (group {}): actual decompressed size {} differs from claimed {}",
                block.block_id,
                block.solid_group_id,
                decompressed.len(),
                block.uncompressed_size,
            )));
        }

        // Verify content hash
        let computed_hash = blake3::hash(&decompressed);
        if *computed_hash.as_bytes() != block.content_blake3 {
            return Err(AetherError::ChecksumMismatch {
                block_id: block.block_id,
                expected: hex_str(&block.content_blake3),
                actual: hex_str(computed_hash.as_bytes()),
            });
        }

        results.push((block.global_idx, decompressed));
    }
    Ok(results)
}

/// Seekable methods on `Decompressor`.
impl Decompressor {
    /// Read the archive metadata (header, file table, block index) without
    /// decompressing any blocks.
    ///
    /// Validates that file/block/group counts do not exceed safety limits
    /// to prevent resource exhaustion from crafted archives.
    pub fn read_metadata<R: Read + Seek>(&self, archive: &mut R) -> Result<ArchiveMetadata> {
        // Read footer from the end
        archive.seek(SeekFrom::End(-(ARCHIVE_FOOTER_SIZE as i64)))?;
        let footer = ArchiveFooter::read_from(archive)?;

        // Bounds checks on counts from untrusted archive data
        if footer.file_count > MAX_FILE_COUNT {
            return Err(AetherError::ResourceLimitExceeded(format!(
                "Archive claims {} files, exceeding limit of {}",
                footer.file_count, MAX_FILE_COUNT,
            )));
        }
        if footer.block_count > MAX_BLOCK_COUNT {
            return Err(AetherError::ResourceLimitExceeded(format!(
                "Archive claims {} blocks, exceeding limit of {}",
                footer.block_count, MAX_BLOCK_COUNT,
            )));
        }

        // Read header from the start
        archive.seek(SeekFrom::Start(0))?;
        let header = crate::header::ArchiveHeader::read_from(archive)?;

        // Read encryption header if archive is encrypted
        #[cfg(feature = "enterprise")]
        let encryption_header = if header.flags & FLAG_ENCRYPTED != 0 {
            Some(crypto::EncryptionHeader::read_from(archive)?)
        } else {
            None
        };
        #[cfg(not(feature = "enterprise"))]
        if header.flags & FLAG_ENCRYPTED != 0 {
            return Err(AetherError::Compression(
                "This archive is encrypted. Enable the 'enterprise' feature to decrypt.".into(),
            ));
        }

        // Read dictionary hash if present (32 bytes after encryption header)
        let dictionary_hash = if header.flags & FLAG_HAS_DICTIONARY != 0 {
            let mut hash = [0u8; 32];
            archive.read_exact(&mut hash)?;
            Some(hash)
        } else {
            None
        };

        if header.solid_group_count > MAX_SOLID_GROUP_COUNT {
            return Err(AetherError::ResourceLimitExceeded(format!(
                "Archive claims {} solid groups, exceeding limit of {}",
                header.solid_group_count, MAX_SOLID_GROUP_COUNT,
            )));
        }

        // Cross-validate header and footer redundant fields to detect
        // crafted archives with inconsistent metadata.
        if header.file_count != footer.file_count {
            return Err(AetherError::Decompression(format!(
                "Header/footer file_count mismatch: header={}, footer={}",
                header.file_count, footer.file_count,
            )));
        }
        if header.block_count != footer.block_count {
            return Err(AetherError::Decompression(format!(
                "Header/footer block_count mismatch: header={}, footer={}",
                header.block_count, footer.block_count,
            )));
        }
        if header.file_table_offset != footer.file_table_offset {
            return Err(AetherError::Decompression(format!(
                "Header/footer file_table_offset mismatch: header={}, footer={}",
                header.file_table_offset, footer.file_table_offset,
            )));
        }
        if header.block_index_offset != footer.block_index_offset {
            return Err(AetherError::Decompression(format!(
                "Header/footer block_index_offset mismatch: header={}, footer={}",
                header.block_index_offset, footer.block_index_offset,
            )));
        }

        // M6 security fix: validate archive offsets before seeking.
        // Determine archive size by seeking to end.
        let archive_size = archive.seek(SeekFrom::End(0))?;
        if footer.file_table_offset >= archive_size {
            return Err(AetherError::Decompression(format!(
                "File table offset {} exceeds archive size {}",
                footer.file_table_offset, archive_size,
            )));
        }
        if footer.block_index_offset >= archive_size {
            return Err(AetherError::Decompression(format!(
                "Block index offset {} exceeds archive size {}",
                footer.block_index_offset, archive_size,
            )));
        }

        // S10 security fix: validate that table sizes fit within archive bounds.
        let block_index_end = footer
            .block_index_offset
            .checked_add(footer.block_count as u64 * BLOCK_INDEX_ENTRY_SIZE as u64)
            .ok_or_else(|| AetherError::Decompression("Block index size overflow".into()))?;
        if block_index_end > archive_size {
            return Err(AetherError::Decompression(format!(
                "Block index extends past archive end: offset {} + {} blocks",
                footer.block_index_offset, footer.block_count
            )));
        }

        // Read file table
        // Cap with_capacity to prevent speculative over-allocation from
        // crafted headers claiming millions of entries.
        archive.seek(SeekFrom::Start(footer.file_table_offset))?;
        let mut file_entries =
            Vec::with_capacity((footer.file_count as usize).min(MAX_PREALLOC_CAPACITY));
        for _ in 0..footer.file_count {
            file_entries.push(FileEntry::read_from(archive)?);
        }

        // Read solid group table (immediately after file table)
        let mut solid_groups =
            Vec::with_capacity((header.solid_group_count as usize).min(MAX_PREALLOC_CAPACITY));
        for _ in 0..header.solid_group_count {
            solid_groups.push(SolidGroupEntry::read_from(archive)?);
        }

        // Read block index
        archive.seek(SeekFrom::Start(footer.block_index_offset))?;
        let mut block_index =
            Vec::with_capacity((footer.block_count as usize).min(MAX_PREALLOC_CAPACITY));
        for _ in 0..footer.block_count {
            block_index.push(BlockIndexEntry::read_from(archive)?);
        }

        // Security fix: validate block ID uniqueness to prevent nonce reuse
        // in encrypted archives (each block_id maps to a unique nonce).
        {
            let mut seen_ids = std::collections::HashSet::with_capacity(block_index.len());
            for entry in &block_index {
                if !seen_ids.insert(entry.block_id) {
                    return Err(AetherError::Decompression(format!(
                        "Duplicate block_id {} in block index — \
                         this could cause nonce reuse in encrypted archives",
                        entry.block_id,
                    )));
                }
            }
        }

        Ok(ArchiveMetadata {
            header,
            footer,
            file_entries,
            solid_groups,
            block_index,
            #[cfg(feature = "enterprise")]
            encryption_header,
            dictionary_hash,
        })
    }

    /// Extract all files from an archive to `output_dir`.
    pub fn extract_all<R: Read + Seek>(&self, archive: &mut R, output_dir: &Path) -> Result<()> {
        let metadata = self.read_metadata(archive)?;
        self.validate_dictionary(&metadata.dictionary_hash)?;

        // Derive decryption key if the archive is encrypted
        let decrypt_key = derive_decrypt_key(
            metadata.header.flags,
            #[cfg(feature = "enterprise")]
            &metadata.encryption_header,
            #[cfg(feature = "enterprise")]
            &self.password,
        )?;

        // Decompress all blocks — parallel when enterprise + multi-threaded
        #[cfg(feature = "enterprise")]
        let decompressed_blocks = if self.max_threads != 1 && metadata.block_index.len() > 1 {
            self.decompress_blocks_parallel(archive, &metadata, &decrypt_key)?
        } else {
            self.decompress_blocks_sequential(archive, &metadata, &decrypt_key)?
        };
        #[cfg(not(feature = "enterprise"))]
        let decompressed_blocks =
            self.decompress_blocks_sequential(archive, &metadata, &decrypt_key)?;

        // Reassemble files
        for file_entry in &metadata.file_entries {
            let file_data = reassemble_file_from_blocks(file_entry, &decompressed_blocks)?;

            // Verify BLAKE3 hash
            let computed_hash = blake3::hash(&file_data);
            if *computed_hash.as_bytes() != file_entry.blake3_hash {
                return Err(AetherError::ChecksumMismatch {
                    block_id: file_entry.chunk_start_idx,
                    expected: hex_str(&file_entry.blake3_hash),
                    actual: hex_str(computed_hash.as_bytes()),
                });
            }

            // S1 security fix: atomic write with symlink validation and
            // sanitized permissions (strips setuid/setgid/sticky bits).
            write_validated_file(
                output_dir,
                &file_entry.path,
                &file_data,
                file_entry.mtime,
                #[cfg(unix)]
                file_entry.permissions,
                self.no_clobber,
            )?;
        }

        Ok(())
    }

    /// Extract a single file by path.
    pub fn extract_file<R: Read + Seek>(
        &self,
        archive: &mut R,
        file_path: &str,
        output: &mut dyn Write,
    ) -> Result<()> {
        let metadata = self.read_metadata(archive)?;
        self.validate_dictionary(&metadata.dictionary_hash)?;

        // Derive decryption key if the archive is encrypted
        let decrypt_key = derive_decrypt_key(
            metadata.header.flags,
            #[cfg(feature = "enterprise")]
            &metadata.encryption_header,
            #[cfg(feature = "enterprise")]
            &self.password,
        )?;

        let file_entry = metadata
            .file_entries
            .iter()
            .find(|e| e.path == file_path)
            .ok_or_else(|| AetherError::FileNotFound(file_path.to_string()))?;

        // Only decompress the blocks this file needs
        let start = file_entry.chunk_start_idx as usize;
        // S2 security fix: use checked addition to prevent integer overflow on
        // chunk range from untrusted archive data.
        let end = start
            .checked_add(file_entry.chunk_count as usize)
            .ok_or_else(|| {
                AetherError::ResourceLimitExceeded(format!(
                    "Chunk index overflow: start={}, count={}",
                    start, file_entry.chunk_count
                ))
            })?;

        let mut predictor = self.create_predictor()?;

        // Only allocate for the blocks this file actually needs, not all
        // blocks in the archive. This prevents a crafted archive with
        // millions of blocks from forcing a huge allocation when
        // extracting a single small file.
        let range_len = end.saturating_sub(start).min(metadata.block_index.len());
        let mut decompressed_blocks: Vec<Option<Vec<u8>>> = vec![None; range_len];

        // S2 security fix: track cumulative decompressed size to prevent
        // decompression bomb attacks, matching extract_all's protection.
        let mut total_decompressed: u64 = 0;
        for i in start..end.min(metadata.block_index.len()) {
            let data = self.decompress_block(
                archive,
                &metadata.block_index[i],
                predictor.as_mut(),
                &decrypt_key,
            )?;
            total_decompressed += data.len() as u64;
            if total_decompressed > MAX_TOTAL_DECOMPRESSED_SIZE {
                return Err(AetherError::ResourceLimitExceeded(format!(
                    "Total decompressed size {} exceeds safety limit of {} bytes",
                    total_decompressed, MAX_TOTAL_DECOMPRESSED_SIZE,
                )));
            }
            decompressed_blocks[i - start] = Some(data);
        }

        // Reassemble using a temporary FileEntry with chunk_start_idx
        // adjusted to 0 since our decompressed_blocks is offset-relative.
        let adjusted_entry = FileEntry {
            chunk_start_idx: 0,
            ..file_entry.clone()
        };
        let file_data = reassemble_file_from_blocks(&adjusted_entry, &decompressed_blocks)?;

        // Verify
        let computed_hash = blake3::hash(&file_data);
        if *computed_hash.as_bytes() != file_entry.blake3_hash {
            return Err(AetherError::ChecksumMismatch {
                block_id: file_entry.chunk_start_idx,
                expected: hex_str(&file_entry.blake3_hash),
                actual: hex_str(computed_hash.as_bytes()),
            });
        }

        output.write_all(&file_data)?;
        Ok(())
    }

    /// List all files in the archive.
    pub fn list_files<R: Read + Seek>(&self, archive: &mut R) -> Result<Vec<FileEntry>> {
        let metadata = self.read_metadata(archive)?;
        Ok(metadata.file_entries)
    }

    /// Verify integrity of all blocks and files in the archive.
    ///
    /// Verifies both block-level BLAKE3 hashes (via decompression) and
    /// file-level BLAKE3 hashes (via reassembly). This catches both
    /// corrupted blocks and incorrect chunk range metadata.
    pub fn verify<R: Read + Seek>(&self, archive: &mut R) -> Result<VerificationResult> {
        let metadata = self.read_metadata(archive)?;

        // Derive decryption key if the archive is encrypted
        let decrypt_key = derive_decrypt_key(
            metadata.header.flags,
            #[cfg(feature = "enterprise")]
            &metadata.encryption_header,
            #[cfg(feature = "enterprise")]
            &self.password,
        )?;

        let mut result = VerificationResult {
            total_blocks: metadata.block_index.len(),
            verified_blocks: 0,
            corrupted_blocks: Vec::new(),
        };

        // Use per-group predictors for correct cross-block state, matching
        // the streaming verify path. A single predictor shared across groups
        // would cause state divergence and false verification failures.
        let mut predictors: std::collections::HashMap<u32, Box<dyn ProbabilityPredictor>> =
            std::collections::HashMap::new();

        let mut decompressed_blocks: Vec<Option<Vec<u8>>> = vec![None; metadata.block_index.len()];

        for (i, block_entry) in metadata.block_index.iter().enumerate() {
            // Limit predictor creation to prevent unbounded HashMap growth
            if !predictors.contains_key(&block_entry.solid_group_id)
                && predictors.len() >= MAX_SOLID_GROUP_COUNT as usize
            {
                result.corrupted_blocks.push(block_entry.block_id);
                break;
            }
            let predictor = predictors
                .entry(block_entry.solid_group_id)
                .or_insert_with(|| {
                    self.create_predictor()
                        .unwrap_or_else(|_| (self.predictor_factory)())
                });

            match self.decompress_block(archive, block_entry, predictor.as_mut(), &decrypt_key) {
                Ok(data) => {
                    decompressed_blocks[i] = Some(data);
                    result.verified_blocks += 1;
                }
                Err(_) => {
                    result.corrupted_blocks.push(block_entry.block_id);
                }
            }
        }

        // File-level integrity: verify that reassembled files match their
        // BLAKE3 hashes. This catches corrupted chunk_start_idx/chunk_count
        // metadata that could cause incorrect file reassembly even when
        // individual blocks are valid.
        if result.corrupted_blocks.is_empty() {
            for file_entry in &metadata.file_entries {
                if let Ok(file_data) = reassemble_file_from_blocks(file_entry, &decompressed_blocks)
                {
                    let computed_hash = blake3::hash(&file_data);
                    if *computed_hash.as_bytes() != file_entry.blake3_hash {
                        // Report as the first block of the file
                        result.corrupted_blocks.push(file_entry.chunk_start_idx);
                    }
                } else {
                    result.corrupted_blocks.push(file_entry.chunk_start_idx);
                }
            }
        }

        Ok(result)
    }

    /// Decompress a single block via seekable random access.
    ///
    /// If `decrypt_key` contains `Some(DecryptKey)`, the payload is
    /// decrypted before decompression (for encrypted archives).
    ///
    /// Validates sizes from the block header against safety limits before
    /// allocating memory for the compressed payload.
    pub(crate) fn decompress_block<R: Read + Seek>(
        &self,
        archive: &mut R,
        block_entry: &BlockIndexEntry,
        predictor: &mut dyn ProbabilityPredictor,
        decrypt_key: &Option<super::decompress::DecryptKey>,
    ) -> Result<Vec<u8>> {
        let offset = block_entry.archive_offset;
        archive.seek(SeekFrom::Start(offset))?;

        let block_header = BlockHeader::read_from_at(archive, offset)?;

        // S5 security fix: cross-check block_id to prevent block misattribution
        // from crafted archives with inconsistent block index entries.
        if block_header.block_id != block_entry.block_id {
            return Err(AetherError::Decompression(format!(
                "Block ID mismatch: index says {} but header says {} (offset {:#x})",
                block_entry.block_id, block_header.block_id, offset,
            )));
        }

        // Cross-check block header against block index entry to detect
        // inconsistencies from crafted or corrupted archives.
        if block_header.uncompressed_size != block_entry.uncompressed_size {
            return Err(AetherError::Decompression(format!(
                "Block {}: uncompressed_size mismatch between header ({}) and index ({})",
                block_header.block_id,
                block_header.uncompressed_size,
                block_entry.uncompressed_size,
            )));
        }

        // Cross-check solid_group_id between block header and block index
        // to prevent blocks being assigned to wrong groups (which would
        // corrupt predictor state and could cause silent data misinterpretation).
        if block_header.solid_group_id != block_entry.solid_group_id {
            return Err(AetherError::Decompression(format!(
                "Block {}: solid_group_id mismatch between header ({}) and index ({})",
                block_header.block_id, block_header.solid_group_id, block_entry.solid_group_id,
            )));
        }

        // Bounds check: reject implausibly large compressed payloads
        if block_header.compressed_size as usize > MAX_DECOMPRESSED_BLOCK_SIZE {
            return Err(AetherError::ResourceLimitExceeded(format!(
                "Block {} (offset {:#x}, group {}): compressed size {} exceeds safety limit of {} bytes",
                block_header.block_id, offset, block_header.solid_group_id,
                block_header.compressed_size, MAX_DECOMPRESSED_BLOCK_SIZE,
            )));
        }

        // Bounds check: reject implausibly large uncompressed sizes
        if block_header.uncompressed_size as usize > MAX_DECOMPRESSED_BLOCK_SIZE {
            return Err(AetherError::ResourceLimitExceeded(format!(
                "Block {} (offset {:#x}, group {}): uncompressed size {} exceeds safety limit of {} bytes",
                block_header.block_id, offset, block_header.solid_group_id,
                block_header.uncompressed_size, MAX_DECOMPRESSED_BLOCK_SIZE,
            )));
        }

        // Read compressed (possibly encrypted) payload
        let mut payload = vec![0u8; block_header.compressed_size as usize];
        archive.read_exact(&mut payload)?;

        // Read and verify trailer
        let trailer = BlockTrailer::read_from_with_id(archive, block_header.block_id)?;

        // Decrypt if the archive is encrypted
        let payload = maybe_decrypt_payload(payload, decrypt_key, block_header.block_id)?;

        // Decompress
        let decompressed = router::decompress_chunk(
            &payload,
            block_header.compression_method,
            block_header.uncompressed_size as usize,
            predictor,
            !block_header.predictor_state_flag,
        )
        .map_err(|e| {
            AetherError::Decompression(format!(
                "Block {} (offset {:#x}, group {}, method {:?}): {}",
                block_header.block_id,
                offset,
                block_header.solid_group_id,
                block_header.compression_method,
                e,
            ))
        })?;

        // Validate actual decompressed size matches claimed size
        if decompressed.len() != block_header.uncompressed_size as usize {
            return Err(AetherError::Decompression(format!(
                "Block {} (offset {:#x}): actual decompressed size {} differs from claimed {}",
                block_header.block_id,
                offset,
                decompressed.len(),
                block_header.uncompressed_size,
            )));
        }

        // Verify content hash
        let computed_hash = blake3::hash(&decompressed);
        if *computed_hash.as_bytes() != trailer.content_blake3 {
            return Err(AetherError::ChecksumMismatch {
                block_id: block_header.block_id,
                expected: hex_str(&trailer.content_blake3),
                actual: hex_str(computed_hash.as_bytes()),
            });
        }

        Ok(decompressed)
    }

    // ── Block decompression strategies ────────────────────────────────────────

    /// Decompress all blocks sequentially with a single predictor.
    ///
    /// This is the default path (and the only path without the enterprise feature).
    fn decompress_blocks_sequential<R: Read + Seek>(
        &self,
        archive: &mut R,
        metadata: &ArchiveMetadata,
        decrypt_key: &Option<super::decompress::DecryptKey>,
    ) -> Result<Vec<Option<Vec<u8>>>> {
        let mut decompressed_blocks: Vec<Option<Vec<u8>>> = vec![None; metadata.block_index.len()];
        let mut predictor = self.create_predictor()?;
        // S4 security fix: track cumulative decompressed size to prevent
        // decompression bomb attacks (many small blocks → enormous output).
        let mut total_decompressed: u64 = 0;
        // Track cumulative compressed bytes read to prevent OOM from archives
        // with many blocks claiming large compressed_size.
        let mut total_compressed_read: u64 = 0;
        for (i, block_entry) in metadata.block_index.iter().enumerate() {
            total_compressed_read += block_entry.compressed_size as u64;
            if total_compressed_read > MAX_TOTAL_COMPRESSED_READ_SIZE {
                return Err(AetherError::ResourceLimitExceeded(format!(
                    "Total compressed read size {} exceeds safety limit of {} bytes",
                    total_compressed_read, MAX_TOTAL_COMPRESSED_READ_SIZE,
                )));
            }
            let data =
                self.decompress_block(archive, block_entry, predictor.as_mut(), decrypt_key)?;
            total_decompressed += data.len() as u64;
            if total_decompressed > MAX_TOTAL_DECOMPRESSED_SIZE {
                return Err(AetherError::ResourceLimitExceeded(format!(
                    "Total decompressed size {} exceeds safety limit of {} bytes",
                    total_decompressed, MAX_TOTAL_DECOMPRESSED_SIZE,
                )));
            }
            decompressed_blocks[i] = Some(data);
        }
        Ok(decompressed_blocks)
    }

    /// Decompress all blocks in parallel across solid groups (enterprise feature).
    ///
    /// Two-phase approach that separates I/O from CPU:
    ///
    /// 1. **Sequential I/O**: Read all block payloads from the archive into memory
    ///    (and decrypt if needed). This must be sequential because the archive is
    ///    a single `Read + Seek` stream.
    ///
    /// 2. **Parallel CPU**: Group blocks by `solid_group_id`, create one predictor
    ///    per group, and decompress groups concurrently via rayon. Each group is
    ///    independent — its predictor state evolves only within that group's blocks
    ///    (sorted by `block_id`).
    ///
    /// Thread pool size is controlled by `Decompressor::max_threads`:
    /// - `0` = unlimited (global rayon pool, all cores)
    /// - `N > 1` = bounded thread pool with N threads
    #[cfg(feature = "enterprise")]
    fn decompress_blocks_parallel<R: Read + Seek>(
        &self,
        archive: &mut R,
        metadata: &ArchiveMetadata,
        decrypt_key: &Option<super::decompress::DecryptKey>,
    ) -> Result<Vec<Option<Vec<u8>>>> {
        use rayon::prelude::*;

        let block_count = metadata.block_index.len();

        // Phase 1: Sequential I/O — read all block payloads from disk
        let mut raw_blocks: Vec<RawBlock> =
            Vec::with_capacity(block_count.min(MAX_PREALLOC_CAPACITY));
        // Track cumulative compressed bytes read to prevent OOM from archives
        // with many blocks claiming large compressed_size (matching sequential path).
        let mut total_compressed_read: u64 = 0;
        for (i, block_entry) in metadata.block_index.iter().enumerate() {
            let offset = block_entry.archive_offset;
            archive.seek(SeekFrom::Start(offset))?;

            let block_header = BlockHeader::read_from_at(archive, offset)?;

            // S5 security fix: cross-check block_id
            if block_header.block_id != block_entry.block_id {
                return Err(AetherError::Decompression(format!(
                    "Block ID mismatch: index says {} but header says {} (offset {:#x})",
                    block_entry.block_id, block_header.block_id, offset,
                )));
            }

            // Cross-check block header against block index entry
            if block_header.uncompressed_size != block_entry.uncompressed_size {
                return Err(AetherError::Decompression(format!(
                    "Block {}: uncompressed_size mismatch between header ({}) and index ({})",
                    block_header.block_id,
                    block_header.uncompressed_size,
                    block_entry.uncompressed_size,
                )));
            }

            // Cross-check solid_group_id (parallel path)
            if block_header.solid_group_id != block_entry.solid_group_id {
                return Err(AetherError::Decompression(format!(
                    "Block {}: solid_group_id mismatch between header ({}) and index ({})",
                    block_header.block_id, block_header.solid_group_id, block_entry.solid_group_id,
                )));
            }

            // Bounds check: reject implausibly large compressed payloads
            if block_header.compressed_size as usize > MAX_DECOMPRESSED_BLOCK_SIZE {
                return Err(AetherError::ResourceLimitExceeded(format!(
                    "Block {} (offset {:#x}, group {}): compressed size {} exceeds safety limit of {} bytes",
                    block_header.block_id, offset, block_header.solid_group_id,
                    block_header.compressed_size, MAX_DECOMPRESSED_BLOCK_SIZE,
                )));
            }

            // Bounds check: reject implausibly large uncompressed sizes
            if block_header.uncompressed_size as usize > MAX_DECOMPRESSED_BLOCK_SIZE {
                return Err(AetherError::ResourceLimitExceeded(format!(
                    "Block {} (offset {:#x}, group {}): uncompressed size {} exceeds safety limit of {} bytes",
                    block_header.block_id, offset, block_header.solid_group_id,
                    block_header.uncompressed_size, MAX_DECOMPRESSED_BLOCK_SIZE,
                )));
            }

            // Read compressed (possibly encrypted) payload
            let mut payload = vec![0u8; block_header.compressed_size as usize];
            archive.read_exact(&mut payload)?;

            // Cumulative compressed size tracking (matching sequential path)
            total_compressed_read += block_header.compressed_size as u64;
            if total_compressed_read > MAX_TOTAL_COMPRESSED_READ_SIZE {
                return Err(AetherError::ResourceLimitExceeded(format!(
                    "Total compressed read size {} exceeds safety limit of {} bytes",
                    total_compressed_read, MAX_TOTAL_COMPRESSED_READ_SIZE,
                )));
            }

            // Read and verify trailer
            let trailer = BlockTrailer::read_from_with_id(archive, block_header.block_id)?;

            // Decrypt if the archive is encrypted
            let payload = maybe_decrypt_payload(payload, decrypt_key, block_header.block_id)?;

            raw_blocks.push(RawBlock {
                global_idx: i,
                payload,
                compression_method: block_header.compression_method,
                uncompressed_size: block_header.uncompressed_size as usize,
                predictor_state_flag: block_header.predictor_state_flag,
                content_blake3: trailer.content_blake3,
                block_id: block_header.block_id,
                archive_offset: offset,
                solid_group_id: block_header.solid_group_id,
            });
        }

        // S7 security fix: pre-check total expected decompressed size before
        // launching parallel decompression. This prevents OOM from decompression
        // bombs where all blocks decompress successfully but the aggregate exceeds
        // memory limits — the post-hoc check would fire too late.
        let total_expected: u64 = raw_blocks.iter().map(|b| b.uncompressed_size as u64).sum();
        if total_expected > MAX_TOTAL_DECOMPRESSED_SIZE {
            return Err(AetherError::ResourceLimitExceeded(format!(
                "Total expected decompressed size {} exceeds safety limit of {} bytes",
                total_expected, MAX_TOTAL_DECOMPRESSED_SIZE,
            )));
        }

        // Group blocks by solid_group_id
        let mut groups: HashMap<u32, Vec<RawBlock>> = HashMap::new();
        for raw in raw_blocks {
            groups.entry(raw.solid_group_id).or_default().push(raw);
        }

        // Sort blocks within each group by block_id for correct predictor state
        let mut group_work: Vec<Vec<RawBlock>> = groups.into_values().collect();
        for blocks in &mut group_work {
            blocks.sort_by_key(|b| b.block_id);
        }

        // Create predictors on main thread (factory may not be Send+Sync)
        let mut work: Vec<(Vec<RawBlock>, Box<dyn ProbabilityPredictor>)> = Vec::new();
        for blocks in group_work {
            let predictor = self.create_predictor()?;
            work.push((blocks, predictor));
        }

        // Phase 2: Parallel decompression across solid groups
        #[allow(clippy::type_complexity)]
        let group_results: Vec<Result<Vec<(usize, Vec<u8>)>>> = if self.max_threads == 0 {
            // Unlimited: use the global rayon pool (all cores)
            work.par_iter_mut()
                .map(|(blocks, predictor)| decompress_group(blocks, predictor.as_mut()))
                .collect()
        } else {
            // Bounded: create a scoped thread pool with limited threads
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(self.max_threads)
                .build()
                .map_err(|e| {
                    AetherError::Compression(format!(
                        "Failed to create decompression thread pool with {} threads: {e}",
                        self.max_threads,
                    ))
                })?;
            pool.install(|| {
                work.par_iter_mut()
                    .map(|(blocks, predictor)| decompress_group(blocks, predictor.as_mut()))
                    .collect()
            })
        };

        // Assemble results into flat array
        // S4 security fix: track cumulative decompressed size to prevent
        // decompression bomb attacks.
        let mut decompressed_blocks: Vec<Option<Vec<u8>>> = vec![None; block_count];
        let mut total_decompressed: u64 = 0;
        for result in group_results {
            for (global_idx, data) in result? {
                total_decompressed += data.len() as u64;
                if total_decompressed > MAX_TOTAL_DECOMPRESSED_SIZE {
                    return Err(AetherError::ResourceLimitExceeded(format!(
                        "Total decompressed size {} exceeds safety limit of {} bytes",
                        total_decompressed, MAX_TOTAL_DECOMPRESSED_SIZE,
                    )));
                }
                decompressed_blocks[global_idx] = Some(data);
            }
        }

        Ok(decompressed_blocks)
    }
}
