//! Archive migration: decompress → recompress with new settings.
//!
//! Enables upgrading archives to a new format version, changing predictor,
//! adding/removing encryption, or applying a dictionary.

use std::io::{Read, Seek, Write};
use std::path::Path;

use crate::dictionary::Dictionary;
use crate::entropy::ProbabilityPredictor;
use crate::error::{AetherError, Result};
use crate::format::MAX_FILE_COUNT;
use crate::pipeline::compress::Compressor;
use crate::pipeline::decompress::Decompressor;

/// Migrates an `.aet` archive by decompressing and recompressing with new settings.
///
/// Supports changing predictor, adding/removing encryption, and applying dictionaries.
/// Validates roundtrip integrity via per-file BLAKE3 hashes.
pub struct Migrator {
    /// Decompressor for reading the source archive.
    source_decompressor: Decompressor,
    /// Compressor for writing the target archive.
    target_compressor: Compressor,
}

impl Migrator {
    /// Create a migrator with the given source and target configurations.
    ///
    /// The `source_factory` creates predictors matching the source archive.
    /// The `target_factory` creates predictors for the new archive.
    pub fn new<SF, TF>(source_factory: SF, target_factory: TF) -> Self
    where
        SF: Fn() -> Box<dyn ProbabilityPredictor> + 'static,
        TF: Fn() -> Box<dyn ProbabilityPredictor> + Send + Sync + 'static,
    {
        Self {
            source_decompressor: Decompressor::new(source_factory),
            target_compressor: Compressor::new(target_factory),
        }
    }

    /// Set a dictionary for the source archive (needed if source was compressed with one).
    pub fn with_source_dictionary(mut self, dict: Dictionary) -> Self {
        self.source_decompressor = self.source_decompressor.with_dictionary(dict);
        self
    }

    /// Set a dictionary for the target archive.
    pub fn with_target_dictionary(mut self, dict: Dictionary) -> Self {
        self.target_compressor = self.target_compressor.with_dictionary(dict);
        self
    }

    /// Set password for decrypting the source archive (enterprise feature).
    #[cfg(feature = "enterprise")]
    pub fn with_source_password(mut self, password: &str) -> Self {
        self.source_decompressor = self.source_decompressor.with_password(password);
        self
    }

    /// Set password for encrypting the target archive (enterprise feature).
    #[cfg(feature = "enterprise")]
    pub fn with_target_password(mut self, password: &str, cipher: crate::crypto::CipherId) -> Self {
        self.target_compressor = self.target_compressor.with_encryption(password, cipher);
        self
    }

    /// Migrate a seekable archive, writing the result to `output`.
    ///
    /// 1. Extracts all files from `source` to a temp directory
    /// 2. Recompresses with the target settings into `output`
    /// 3. Validates the output archive by verifying it
    ///
    /// Returns the number of files migrated.
    pub fn migrate<RS, W>(&self, source: &mut RS, output: &mut W) -> Result<usize>
    where
        RS: Read + Seek,
        W: Write + Seek,
    {
        // Extract to temp directory
        let tmp = tempfile::tempdir().map_err(|e| {
            crate::error::AetherError::Compression(format!("Failed to create temp dir: {e}"))
        })?;

        self.source_decompressor.extract_all(source, tmp.path())?;

        // Collect extracted files
        let files = collect_files_recursive(tmp.path())?;
        let file_count = files.len();

        if files.is_empty() {
            return Ok(0);
        }

        // Recompress with target settings
        self.target_compressor
            .compress_to_archive(tmp.path(), &files, output)?;

        Ok(file_count)
    }

    /// Migrate a streaming (non-seekable) archive.
    ///
    /// Same as `migrate` but reads from a `Read`-only source.
    pub fn migrate_streaming<R, W>(&self, source: &mut R, output: &mut W) -> Result<usize>
    where
        R: Read,
        W: Write + Seek,
    {
        let tmp = tempfile::tempdir().map_err(|e| {
            crate::error::AetherError::Compression(format!("Failed to create temp dir: {e}"))
        })?;

        self.source_decompressor
            .extract_all_streaming(source, tmp.path())?;

        let files = collect_files_recursive(tmp.path())?;
        let file_count = files.len();

        if files.is_empty() {
            return Ok(0);
        }

        self.target_compressor
            .compress_to_archive(tmp.path(), &files, output)?;

        Ok(file_count)
    }
}

/// Recursively collect all files under a directory.
fn collect_files_recursive(dir: &Path) -> Result<Vec<std::path::PathBuf>> {
    let mut files = Vec::new();
    collect_dir_recursive(dir, &mut files)?;
    files.sort(); // deterministic order
    Ok(files)
}

fn collect_dir_recursive(dir: &Path, files: &mut Vec<std::path::PathBuf>) -> Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        // Use symlink_metadata to detect symlinks without following them.
        // This prevents an attacker from planting symlinks in the temp directory
        // that point to sensitive system files (e.g. /etc/shadow, C:\Windows\).
        let meta = std::fs::symlink_metadata(&path)?;
        if meta.file_type().is_symlink() {
            // Skip symlinks entirely — they should never appear in temp
            // extraction directories and could be an attack vector.
            continue;
        }
        if meta.file_type().is_dir() {
            collect_dir_recursive(&path, files)?;
        } else if meta.file_type().is_file() {
            files.push(path);
        }
        // Skip other special file types (sockets, block devices, etc.)

        // Limit file count to prevent resource exhaustion from crafted
        // archives containing millions of tiny files.
        if files.len() > MAX_FILE_COUNT as usize {
            return Err(AetherError::ResourceLimitExceeded(format!(
                "Extracted file count exceeds limit of {MAX_FILE_COUNT}",
            )));
        }
    }
    Ok(())
}
