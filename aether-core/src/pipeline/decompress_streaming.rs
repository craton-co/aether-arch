//! Streaming decompression: `Read`-only path for non-seekable sources.
//!
//! Reads Header → FileTable → GroupTable → Blocks sequentially. Enables
//! piped workflows: `cat archive.aet | aet extract -`.

use std::collections::HashMap;
use std::io::Read;
use std::path::Path;

use crate::block::{BlockHeader, BlockTrailer};
use crate::entropy::ProbabilityPredictor;
use crate::error::{AetherError, Result};
use crate::format::*;
use crate::header::{ArchiveHeader, FileEntry, SolidGroupEntry};
use crate::pipeline::router;

use super::decompress::{
    derive_decrypt_key, hex_str, maybe_decrypt_payload, reassemble_file_from_blocks,
    write_validated_file, Decompressor, StreamingMetadata, VerificationResult,
};

#[cfg(feature = "enterprise")]
use crate::crypto;

/// Streaming methods on `Decompressor`.
impl Decompressor {
    /// Read archive metadata from a sequential (non-seekable) stream.
    ///
    /// Reads: Header (48 B) → [EncryptionHeader (57 B)] → FileTable (variable) → SolidGroupTable (24 B × N).
    /// The reader is left positioned at the start of the first block.
    ///
    /// This is a static method — no predictor is needed to read metadata.
    /// Use the returned `StreamingMetadata.header.predictor_id` to create
    /// the correct `Decompressor` before calling `extract_with_streaming_metadata`.
    pub fn read_metadata_streaming<R: Read>(archive: &mut R) -> Result<StreamingMetadata> {
        let header = ArchiveHeader::read_from(archive)?;

        // Bounds checks on counts from untrusted archive data
        if header.file_count > MAX_FILE_COUNT {
            return Err(AetherError::ResourceLimitExceeded(format!(
                "Archive claims {} files, exceeding limit of {}",
                header.file_count, MAX_FILE_COUNT,
            )));
        }
        if header.block_count > MAX_BLOCK_COUNT {
            return Err(AetherError::ResourceLimitExceeded(format!(
                "Archive claims {} blocks, exceeding limit of {}",
                header.block_count, MAX_BLOCK_COUNT,
            )));
        }
        if header.solid_group_count > MAX_SOLID_GROUP_COUNT {
            return Err(AetherError::ResourceLimitExceeded(format!(
                "Archive claims {} solid groups, exceeding limit of {}",
                header.solid_group_count, MAX_SOLID_GROUP_COUNT,
            )));
        }

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

        // Read dictionary hash if present (32 bytes)
        let dictionary_hash = if header.flags & FLAG_HAS_DICTIONARY != 0 {
            let mut hash = [0u8; 32];
            archive.read_exact(&mut hash)?;
            Some(hash)
        } else {
            None
        };

        let mut file_entries =
            Vec::with_capacity((header.file_count as usize).min(MAX_PREALLOC_CAPACITY));
        for _ in 0..header.file_count {
            file_entries.push(FileEntry::read_from(archive)?);
        }

        let mut solid_groups =
            Vec::with_capacity((header.solid_group_count as usize).min(MAX_PREALLOC_CAPACITY));
        for _ in 0..header.solid_group_count {
            solid_groups.push(SolidGroupEntry::read_from(archive)?);
        }

        Ok(StreamingMetadata {
            header,
            file_entries,
            solid_groups,
            #[cfg(feature = "enterprise")]
            encryption_header,
            dictionary_hash,
        })
    }

    /// Extract all files from a streaming (non-seekable) archive to `output_dir`.
    ///
    /// Reads the entire archive sequentially: metadata then blocks.
    /// Does not require `Seek` — works with pipes.
    pub fn extract_all_streaming<R: Read>(&self, archive: &mut R, output_dir: &Path) -> Result<()> {
        let metadata = Self::read_metadata_streaming(archive)?;
        self.extract_with_streaming_metadata(archive, &metadata, output_dir)
    }

    /// Extract all files given pre-read streaming metadata.
    ///
    /// The reader must be positioned at the start of the first block
    /// (immediately after the solid group table). Use this when the
    /// caller has already called `read_metadata_streaming` to inspect
    /// the header (e.g. to auto-detect the predictor).
    pub fn extract_with_streaming_metadata<R: Read>(
        &self,
        archive: &mut R,
        metadata: &StreamingMetadata,
        output_dir: &Path,
    ) -> Result<()> {
        self.validate_dictionary(&metadata.dictionary_hash)?;
        let block_count = metadata.header.block_count as usize;

        // Derive decryption key if the archive is encrypted
        let decrypt_key = derive_decrypt_key(
            metadata.header.flags,
            #[cfg(feature = "enterprise")]
            &metadata.encryption_header,
            #[cfg(feature = "enterprise")]
            &self.password,
        )?;

        // Per-group predictors for correct cross-block state
        let mut predictors: HashMap<u32, Box<dyn ProbabilityPredictor>> = HashMap::new();

        // Decompressed blocks, indexed by sequential block position (0..N)
        let mut decompressed_blocks: Vec<Option<Vec<u8>>> = vec![None; block_count];

        // Read and decompress blocks sequentially
        // S4 security fix: track cumulative decompressed size to prevent
        // decompression bomb attacks (many small blocks → enormous output).
        let mut total_decompressed: u64 = 0;
        // Track cumulative compressed bytes read to prevent OOM from archives
        // with many blocks claiming large compressed_size.
        let mut total_compressed_read: u64 = 0;
        for slot in decompressed_blocks.iter_mut().take(block_count) {
            let (data, _block_id, compressed_size) =
                self.decompress_block_streaming(archive, &mut predictors, &decrypt_key)?;
            total_compressed_read += compressed_size as u64;
            if total_compressed_read > MAX_TOTAL_COMPRESSED_READ_SIZE {
                return Err(AetherError::ResourceLimitExceeded(format!(
                    "Total compressed read size {} exceeds safety limit of {} bytes",
                    total_compressed_read, MAX_TOTAL_COMPRESSED_READ_SIZE,
                )));
            }
            total_decompressed += data.len() as u64;
            if total_decompressed > MAX_TOTAL_DECOMPRESSED_SIZE {
                return Err(AetherError::ResourceLimitExceeded(format!(
                    "Total decompressed size {} exceeds safety limit of {} bytes",
                    total_decompressed, MAX_TOTAL_DECOMPRESSED_SIZE,
                )));
            }
            *slot = Some(data);
        }

        // Reassemble and write files
        for file_entry in &metadata.file_entries {
            let file_data = reassemble_file_from_blocks(file_entry, &decompressed_blocks)?;

            // Verify file-level BLAKE3 hash
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

    /// List files from a streaming (non-seekable) archive.
    ///
    /// Only reads the header and file table — does not read any blocks.
    pub fn list_files_streaming<R: Read>(archive: &mut R) -> Result<Vec<FileEntry>> {
        let metadata = Self::read_metadata_streaming(archive)?;
        Ok(metadata.file_entries)
    }

    /// Verify integrity of all blocks in a streaming (non-seekable) archive.
    ///
    /// Reads all blocks sequentially, verifying BLAKE3 hashes.
    /// If a structural I/O error occurs (corrupt header/trailer, truncation),
    /// verification stops early — the stream cannot be recovered.
    pub fn verify_streaming<R: Read>(&self, archive: &mut R) -> Result<VerificationResult> {
        let metadata = Self::read_metadata_streaming(archive)?;
        self.verify_with_streaming_metadata(archive, &metadata)
    }

    /// Verify integrity given pre-read streaming metadata.
    pub fn verify_with_streaming_metadata<R: Read>(
        &self,
        archive: &mut R,
        metadata: &StreamingMetadata,
    ) -> Result<VerificationResult> {
        let block_count = metadata.header.block_count as usize;

        // Derive decryption key if the archive is encrypted
        let decrypt_key = derive_decrypt_key(
            metadata.header.flags,
            #[cfg(feature = "enterprise")]
            &metadata.encryption_header,
            #[cfg(feature = "enterprise")]
            &self.password,
        )?;

        let mut result = VerificationResult {
            total_blocks: block_count,
            verified_blocks: 0,
            corrupted_blocks: Vec::new(),
        };

        let mut predictors: HashMap<u32, Box<dyn ProbabilityPredictor>> = HashMap::new();

        for block_idx in 0..block_count {
            // Read block header
            let block_header = match BlockHeader::read_from(archive) {
                Ok(h) => h,
                Err(_) => {
                    result.corrupted_blocks.push(block_idx as u32);
                    // Can't continue streaming after a corrupt header
                    break;
                }
            };

            // Bounds check before allocation
            if block_header.compressed_size as usize > MAX_DECOMPRESSED_BLOCK_SIZE {
                result.corrupted_blocks.push(block_header.block_id);
                break;
            }
            if block_header.uncompressed_size as usize > MAX_DECOMPRESSED_BLOCK_SIZE {
                result.corrupted_blocks.push(block_header.block_id);
                break;
            }

            // Read payload
            let mut payload = vec![0u8; block_header.compressed_size as usize];
            if archive.read_exact(&mut payload).is_err() {
                result.corrupted_blocks.push(block_header.block_id);
                break;
            }

            // Read trailer
            let trailer = match BlockTrailer::read_from_with_id(archive, block_header.block_id) {
                Ok(t) => t,
                Err(_) => {
                    result.corrupted_blocks.push(block_header.block_id);
                    break;
                }
            };

            // Decrypt if encrypted
            let payload = match maybe_decrypt_payload(payload, &decrypt_key, block_header.block_id)
            {
                Ok(p) => p,
                Err(_) => {
                    result.corrupted_blocks.push(block_header.block_id);
                    continue;
                }
            };

            // Q7 security fix: limit predictor creation in verify path too
            if !predictors.contains_key(&block_header.solid_group_id)
                && predictors.len() >= MAX_SOLID_GROUP_COUNT as usize
            {
                result.corrupted_blocks.push(block_header.block_id);
                break;
            }
            predictors
                .entry(block_header.solid_group_id)
                .or_insert_with(|| (self.predictor_factory)());
            let predictor = predictors.get_mut(&block_header.solid_group_id).unwrap();

            match router::decompress_chunk(
                &payload,
                block_header.compression_method,
                block_header.uncompressed_size as usize,
                predictor.as_mut(),
                !block_header.predictor_state_flag,
            ) {
                Ok(data) => {
                    let computed_hash = blake3::hash(&data);
                    if *computed_hash.as_bytes() != trailer.content_blake3 {
                        result.corrupted_blocks.push(block_header.block_id);
                    } else {
                        result.verified_blocks += 1;
                    }
                }
                Err(_) => {
                    result.corrupted_blocks.push(block_header.block_id);
                    // Decompression failed but stream position is correct
                    // (payload was already read), so we can continue.
                }
            }
        }

        Ok(result)
    }

    /// Decompress one block from a sequential stream, using per-group predictors.
    ///
    /// Returns `(decompressed_data, block_id, compressed_size)`.
    fn decompress_block_streaming<R: Read>(
        &self,
        archive: &mut R,
        predictors: &mut HashMap<u32, Box<dyn ProbabilityPredictor>>,
        decrypt_key: &Option<super::decompress::DecryptKey>,
    ) -> Result<(Vec<u8>, u32, u32)> {
        let block_header = BlockHeader::read_from(archive)?;

        // Bounds check: reject implausibly large payloads before allocating
        if block_header.compressed_size as usize > MAX_DECOMPRESSED_BLOCK_SIZE {
            return Err(AetherError::ResourceLimitExceeded(format!(
                "Block {} (group {}, streaming): compressed size {} exceeds safety limit of {} bytes",
                block_header.block_id, block_header.solid_group_id,
                block_header.compressed_size, MAX_DECOMPRESSED_BLOCK_SIZE,
            )));
        }

        // Bounds check: reject implausibly large uncompressed sizes
        if block_header.uncompressed_size as usize > MAX_DECOMPRESSED_BLOCK_SIZE {
            return Err(AetherError::ResourceLimitExceeded(format!(
                "Block {} (group {}, streaming): uncompressed size {} exceeds safety limit of {} bytes",
                block_header.block_id, block_header.solid_group_id,
                block_header.uncompressed_size, MAX_DECOMPRESSED_BLOCK_SIZE,
            )));
        }

        let mut payload = vec![0u8; block_header.compressed_size as usize];
        archive.read_exact(&mut payload)?;

        let trailer = BlockTrailer::read_from_with_id(archive, block_header.block_id)?;

        // Decrypt if encrypted
        let payload = maybe_decrypt_payload(payload, decrypt_key, block_header.block_id)?;

        // Q7 security fix: limit predictor creation to prevent unbounded HashMap
        // growth from crafted archives with unique solid_group_id per block.
        if !predictors.contains_key(&block_header.solid_group_id)
            && predictors.len() >= MAX_SOLID_GROUP_COUNT as usize
        {
            return Err(AetherError::ResourceLimitExceeded(format!(
                "Too many distinct solid groups encountered in blocks (>{MAX_SOLID_GROUP_COUNT})",
            )));
        }
        predictors
            .entry(block_header.solid_group_id)
            .or_insert_with(|| {
                self.create_predictor()
                    .unwrap_or_else(|_| (self.predictor_factory)())
            });
        let predictor = predictors.get_mut(&block_header.solid_group_id).unwrap();

        let decompressed = router::decompress_chunk(
            &payload,
            block_header.compression_method,
            block_header.uncompressed_size as usize,
            predictor.as_mut(),
            !block_header.predictor_state_flag,
        )
        .map_err(|e| {
            AetherError::Decompression(format!(
                "Block {} (group {}, method {:?}, streaming): {}",
                block_header.block_id,
                block_header.solid_group_id,
                block_header.compression_method,
                e,
            ))
        })?;

        // Validate actual decompressed size matches claimed size
        if decompressed.len() != block_header.uncompressed_size as usize {
            return Err(AetherError::Decompression(format!(
                "Block {} (group {}, streaming): actual decompressed size {} differs from claimed {}",
                block_header.block_id, block_header.solid_group_id,
                decompressed.len(), block_header.uncompressed_size,
            )));
        }

        // Verify BLAKE3
        let computed_hash = blake3::hash(&decompressed);
        if *computed_hash.as_bytes() != trailer.content_blake3 {
            return Err(AetherError::ChecksumMismatch {
                block_id: block_header.block_id,
                expected: hex_str(&trailer.content_blake3),
                actual: hex_str(computed_hash.as_bytes()),
            });
        }

        Ok((
            decompressed,
            block_header.block_id,
            block_header.compressed_size,
        ))
    }
}
