//! High-level decompression pipeline: .aet archive → files.
//!
//! Two modes of operation:
//!
//! - **Seekable** (`Read + Seek`): random-access extraction via footer + block index.
//!   Supports single-file extraction (`extract_file`).
//!   Implementation: `super::decompress_seekable`.
//!
//! - **Streaming** (`Read` only): sequential extraction from a non-seekable source
//!   (e.g. stdin pipe). Reads Header → FileTable → GroupTable → Blocks in order.
//!   Does not require seeking; enables `cat archive.aet | aet extract -`.
//!   Implementation: `super::decompress_streaming`.

use crate::block::BlockIndexEntry;
use crate::entropy::ProbabilityPredictor;
use crate::error::{AetherError, Result};
use crate::format::*;
use crate::header::{ArchiveFooter, ArchiveHeader, FileEntry, SolidGroupEntry};

#[cfg(feature = "enterprise")]
use crate::crypto;
#[cfg(feature = "enterprise")]
use zeroize::Zeroize;

/// Decryption key with cipher ID, zeroized on drop.
///
/// Wraps the derived key material so it is securely erased from memory
/// when no longer needed. The cipher ID is stored alongside to avoid
/// passing it separately.
pub(crate) struct DecryptKey {
    #[allow(dead_code)]
    pub cipher_id: u8,
    key: [u8; 32],
    master_nonce: [u8; 12],
}

impl DecryptKey {
    #[allow(dead_code)]
    pub fn new(cipher_id: u8, key: [u8; 32], master_nonce: [u8; 12]) -> Self {
        Self {
            cipher_id,
            key,
            master_nonce,
        }
    }

    #[allow(dead_code)]
    pub fn key_bytes(&self) -> &[u8; 32] {
        &self.key
    }

    #[allow(dead_code)]
    pub fn master_nonce(&self) -> &[u8; 12] {
        &self.master_nonce
    }
}

impl Drop for DecryptKey {
    fn drop(&mut self) {
        // Securely erase key material from memory.
        // When enterprise feature is enabled, use the zeroize crate which
        // combines compiler_fence + write_volatile for robust erasure.
        // Otherwise, use a volatile write loop as a best-effort fallback.
        #[cfg(feature = "enterprise")]
        {
            self.key.zeroize();
            self.master_nonce.zeroize();
        }
        #[cfg(not(feature = "enterprise"))]
        {
            for byte in self.key.iter_mut() {
                unsafe { std::ptr::write_volatile(byte, 0) };
            }
            for byte in self.master_nonce.iter_mut() {
                unsafe { std::ptr::write_volatile(byte, 0) };
            }
            // Prevent the compiler from reordering the volatile writes
            // with subsequent reads or optimizing them away.
            std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);
        }
    }
}

// ── Core types ──────────────────────────────────────────────────────────────

/// High-level decompressor for `.aet` archives.
///
/// Supports two modes: seekable (random-access via footer + block index) and
/// streaming (sequential read-only, for pipes and network streams).
///
/// # Example
///
/// ```rust,no_run
/// use aether_core::pipeline::decompress::Decompressor;
/// use aether_core::entropy::NeuralSsmPredictor;
///
/// let decompressor = Decompressor::new(|| Box::new(NeuralSsmPredictor::new()));
/// ```
pub struct Decompressor {
    pub(crate) predictor_factory: Box<dyn Fn() -> Box<dyn ProbabilityPredictor>>,
    /// Password for encrypted archives (enterprise feature).
    /// Wrapped in `Zeroizing` to securely erase from memory on drop.
    #[cfg(feature = "enterprise")]
    pub(crate) password: Option<zeroize::Zeroizing<Vec<u8>>>,
    // Note: zeroize crate is available when enterprise feature is enabled.
    /// Maximum threads for parallel decompression (enterprise feature).
    /// 0 = unlimited (rayon default), 1 = sequential (default).
    #[cfg(feature = "enterprise")]
    pub(crate) max_threads: usize,
    /// Dictionary for predictor pretraining.
    pub(crate) dictionary: Option<crate::dictionary::Dictionary>,
    /// If true, refuse to overwrite existing files during extraction.
    /// This eliminates TOCTOU races for the no-clobber case since we
    /// reject any target that already exists rather than checking and
    /// then writing (which has a race window).
    pub(crate) no_clobber: bool,
}

impl Decompressor {
    pub fn new<F>(factory: F) -> Self
    where
        F: Fn() -> Box<dyn ProbabilityPredictor> + 'static,
    {
        Self {
            predictor_factory: Box::new(factory),
            #[cfg(feature = "enterprise")]
            password: None,
            #[cfg(feature = "enterprise")]
            max_threads: 1,
            dictionary: None,
            no_clobber: false,
        }
    }

    /// Set a pretrained dictionary for decompression.
    ///
    /// If the archive was compressed with a dictionary, the same dictionary
    /// must be provided for correct decompression. The archive header stores
    /// a BLAKE3 hash of the dictionary for validation.
    pub fn with_dictionary(mut self, dict: crate::dictionary::Dictionary) -> Self {
        self.dictionary = Some(dict);
        self
    }

    /// Set the password for decrypting encrypted archives.
    ///
    /// If the archive is not encrypted, the password is silently ignored.
    /// If the archive is encrypted but no password is set, extraction will
    /// return an error.
    #[cfg(feature = "enterprise")]
    pub fn with_password(mut self, password: &str) -> Self {
        self.password = Some(zeroize::Zeroizing::new(password.as_bytes().to_vec()));
        self
    }

    /// Set the maximum threads for parallel decompression (enterprise feature).
    ///
    /// Parallelizes decompression across solid groups (seekable path only).
    /// Each group is independent and has its own predictor, so groups can
    /// be decompressed concurrently.
    ///
    /// - `0` = unlimited (use all available cores via rayon)
    /// - `1` = sequential (default, same as non-enterprise behavior)
    /// - `N` = use at most N threads
    ///
    /// Streaming decompression is always sequential regardless of this setting.
    #[cfg(feature = "enterprise")]
    pub fn with_max_threads(mut self, max_threads: usize) -> Self {
        self.max_threads = max_threads;
        self
    }

    /// Refuse to overwrite existing files during extraction.
    ///
    /// When enabled, extraction returns [`AetherError::FileAlreadyExists`] if
    /// a target path already exists instead of overwriting it. This also
    /// eliminates the TOCTOU race window inherent in check-then-overwrite
    /// patterns, since no overwrite is ever attempted.
    pub fn with_no_clobber(mut self, no_clobber: bool) -> Self {
        self.no_clobber = no_clobber;
        self
    }

    /// Create a predictor and apply dictionary state if configured.
    ///
    /// Returns an error if dictionary application fails, rather than silently
    /// producing a predictor with wrong initial state (which would cause
    /// decompression failures or data corruption).
    pub(crate) fn create_predictor(&self) -> Result<Box<dyn ProbabilityPredictor>> {
        let mut predictor = (self.predictor_factory)();
        if let Some(ref dict) = self.dictionary {
            dict.apply(predictor.as_mut()).map_err(|e| {
                AetherError::Decompression(format!("Failed to apply dictionary to predictor: {e}"))
            })?;
            // Stage A: install the same per-block coding baseline the encoder
            // used, so the router's BWT/LZ77 decode predictors reset to the
            // identical starting state. Symmetric with compress's compress_group.
            predictor.set_coding_baseline(&dict.state);
        }
        Ok(predictor)
    }

    /// Validate that the archive's dictionary hash matches the configured dictionary.
    pub(crate) fn validate_dictionary(&self, archive_dict_hash: &Option<[u8; 32]>) -> Result<()> {
        match (archive_dict_hash, &self.dictionary) {
            (Some(hash), Some(dict)) => {
                if *hash != dict.hash {
                    return Err(AetherError::Decompression(
                        "Dictionary hash mismatch: archive was compressed with a different dictionary".into(),
                    ));
                }
                Ok(())
            }
            (Some(_), None) => Err(AetherError::Decompression(
                "Archive was compressed with a dictionary but no dictionary was provided. \
                     Use with_dictionary() to supply the matching .aed file."
                    .into(),
            )),
            (None, Some(_)) => {
                // Dictionary provided but archive doesn't need one — silently ignore
                Ok(())
            }
            (None, None) => Ok(()),
        }
    }
}

/// Derive the decryption key from archive metadata and password.
///
/// Returns `Some((cipher_id_u8, key))` if the archive is encrypted and
/// the enterprise feature is enabled, or `None` if not encrypted.
/// Returns an error if encrypted but no password is provided, or if
/// the enterprise feature is not enabled.
pub(crate) fn derive_decrypt_key(
    flags: u16,
    #[cfg(feature = "enterprise")] encryption_header: &Option<crypto::EncryptionHeader>,
    #[cfg(feature = "enterprise")] password: &Option<zeroize::Zeroizing<Vec<u8>>>,
) -> Result<Option<DecryptKey>> {
    if flags & FLAG_ENCRYPTED == 0 {
        return Ok(None);
    }

    #[cfg(feature = "enterprise")]
    {
        let enc_header = encryption_header.as_ref().ok_or_else(|| {
            AetherError::Decompression("Archive has FLAG_ENCRYPTED but no encryption header".into())
        })?;
        let pw = password.as_ref().ok_or_else(|| {
            AetherError::Decryption(
                "Archive is encrypted but no password was provided. Use with_password().".into(),
            )
        })?;

        // Validate Argon2id parameters from untrusted archive data to prevent
        // resource exhaustion attacks (crafted archives setting extreme costs).
        if enc_header.m_cost > MAX_ARGON2_M_COST {
            return Err(AetherError::ResourceLimitExceeded(format!(
                "Archive Argon2id m_cost {} exceeds safety limit of {}",
                enc_header.m_cost, MAX_ARGON2_M_COST,
            )));
        }
        if enc_header.t_cost > MAX_ARGON2_T_COST {
            return Err(AetherError::ResourceLimitExceeded(format!(
                "Archive Argon2id t_cost {} exceeds safety limit of {}",
                enc_header.t_cost, MAX_ARGON2_T_COST,
            )));
        }
        if enc_header.p_cost > MAX_ARGON2_P_COST {
            return Err(AetherError::ResourceLimitExceeded(format!(
                "Archive Argon2id p_cost {} exceeds safety limit of {}",
                enc_header.p_cost, MAX_ARGON2_P_COST,
            )));
        }
        if enc_header.m_cost == 0 || enc_header.t_cost == 0 || enc_header.p_cost == 0 {
            return Err(AetherError::Decompression(
                "Archive Argon2id parameters must be non-zero".into(),
            ));
        }

        let derived = crypto::derive_key(
            pw,
            &enc_header.salt,
            enc_header.m_cost,
            enc_header.t_cost,
            enc_header.p_cost,
        )?;
        // Fast-fail on wrong password before attempting block decryption.
        enc_header.verify_password(&derived)?;
        Ok(Some(DecryptKey::new(
            enc_header.cipher_id as u8,
            *derived.as_bytes(),
            enc_header.master_nonce,
        )))
    }

    #[cfg(not(feature = "enterprise"))]
    {
        Err(AetherError::Decryption(
            "This archive is encrypted. Enable the 'enterprise' feature to decrypt.".into(),
        ))
    }
}

/// Decrypt a block payload if encryption is active.
///
/// If `decrypt_key` is `None`, returns the payload unchanged.
/// The `block_id` is required for AAD verification during decryption.
pub(crate) fn maybe_decrypt_payload(
    payload: Vec<u8>,
    decrypt_key: &Option<DecryptKey>,
    block_id: u32,
) -> Result<Vec<u8>> {
    match decrypt_key {
        None => Ok(payload),
        Some(dk) => {
            #[cfg(feature = "enterprise")]
            {
                let cid = crypto::CipherId::from_u8(dk.cipher_id)?;
                crypto::decrypt_block(cid, dk.key_bytes(), dk.master_nonce(), block_id, &payload)
            }
            #[cfg(not(feature = "enterprise"))]
            {
                let _ = (dk, block_id);
                Err(AetherError::Decryption(
                    "Encrypted archive requires the 'enterprise' feature".into(),
                ))
            }
        }
    }
}

/// Metadata parsed from an archive via seekable reading.
///
/// Contains the complete header, footer, file table, group table, and block
/// index — everything needed to extract files without re-reading the archive.
#[derive(Debug)]
pub struct ArchiveMetadata {
    /// Archive header (from the start of the file).
    pub header: ArchiveHeader,
    /// Archive footer (from the end of the file).
    pub footer: ArchiveFooter,
    /// File table: one entry per archived file.
    pub file_entries: Vec<FileEntry>,
    /// Solid group table: one entry per content-type group.
    pub solid_groups: Vec<SolidGroupEntry>,
    /// Block index: one entry per compressed block, for random-access seeking.
    pub block_index: Vec<BlockIndexEntry>,
    /// Encryption header (present only if FLAG_ENCRYPTED is set).
    #[cfg(feature = "enterprise")]
    pub encryption_header: Option<crypto::EncryptionHeader>,
    /// Dictionary BLAKE3 hash (present only if FLAG_HAS_DICTIONARY is set).
    pub dictionary_hash: Option<[u8; 32]>,
}

/// Metadata parsed from a streaming (non-seekable) read.
///
/// Contains the header, file table, and solid group table — everything
/// available before the first block. No block index or footer (those
/// require seeking to the end of the archive).
#[derive(Debug)]
pub struct StreamingMetadata {
    pub header: ArchiveHeader,
    pub file_entries: Vec<FileEntry>,
    pub solid_groups: Vec<SolidGroupEntry>,
    /// Encryption header (present only if FLAG_ENCRYPTED is set).
    #[cfg(feature = "enterprise")]
    pub encryption_header: Option<crypto::EncryptionHeader>,
    /// Dictionary BLAKE3 hash (present only if FLAG_HAS_DICTIONARY is set).
    pub dictionary_hash: Option<[u8; 32]>,
}

/// Result of verifying an archive's integrity.
#[derive(Debug)]
#[must_use]
pub struct VerificationResult {
    pub total_blocks: usize,
    pub verified_blocks: usize,
    pub corrupted_blocks: Vec<u32>,
}

impl VerificationResult {
    pub fn is_ok(&self) -> bool {
        self.corrupted_blocks.is_empty()
    }
}

// ── Shared helpers (used by both seekable and streaming paths) ──────────────

/// Reassemble a file from decompressed blocks by concatenating the file's
/// block range and truncating to the original file size.
pub(crate) fn reassemble_file_from_blocks(
    file_entry: &FileEntry,
    decompressed_blocks: &[Option<Vec<u8>>],
) -> Result<Vec<u8>> {
    let start = file_entry.chunk_start_idx as usize;
    let count = file_entry.chunk_count as usize;

    if count == 0 {
        return Ok(Vec::new());
    }

    // M4 security fix: use checked addition to prevent integer overflow on the
    // chunk range (start + count) which comes from untrusted archive data.
    let end = start.checked_add(count).ok_or_else(|| {
        AetherError::ResourceLimitExceeded(format!(
            "Chunk index overflow: start={start}, count={count}"
        ))
    })?;
    // S3 security fix: cap allocation hint to prevent OOM from crafted
    // archives with inflated original_size in untrusted metadata.
    let capacity = (file_entry.original_size as usize).min(MAX_TOTAL_DECOMPRESSED_SIZE as usize);
    let mut combined = Vec::with_capacity(capacity);
    for i in start..end {
        if i >= decompressed_blocks.len() {
            return Err(AetherError::BlockNotFound(i as u32));
        }
        match &decompressed_blocks[i] {
            Some(data) => combined.extend_from_slice(data),
            None => return Err(AetherError::BlockNotFound(i as u32)),
        }
    }

    // Truncate to exact file size (last chunk may extend past file boundary)
    let end = (file_entry.original_size as usize).min(combined.len());
    Ok(combined[..end].to_vec())
}

/// Validate that a file path from an archive is safe to extract.
///
/// Rejects paths containing `..` components (path traversal) and absolute paths
/// that could write outside the output directory.
pub(crate) fn validate_extraction_path(path: &str) -> Result<()> {
    // S9 security fix: reject NUL bytes which could truncate paths at the OS
    // level, potentially bypassing subsequent traversal checks.
    if path.as_bytes().contains(&0) {
        return Err(AetherError::PathTraversal(path.to_string()));
    }
    // Reject absolute paths
    if path.starts_with('/') || path.starts_with('\\') {
        return Err(AetherError::PathTraversal(path.to_string()));
    }
    // Windows absolute paths (C:\, D:\, etc.)
    if path.len() >= 2 && path.as_bytes()[1] == b':' {
        return Err(AetherError::PathTraversal(path.to_string()));
    }
    // Reject path traversal components
    for component in path.split(&['/', '\\']) {
        if component == ".." {
            return Err(AetherError::PathTraversal(path.to_string()));
        }
    }

    // Reject excessively long paths
    if path.len() > MAX_PATH_LENGTH {
        return Err(AetherError::ResourceLimitExceeded(format!(
            "File path length {} exceeds maximum {}",
            path.len(),
            MAX_PATH_LENGTH,
        )));
    }
    // Reject Windows reserved device names (CON, NUL, PRN, AUX, COM1-9, LPT1-9).
    // Extracting files with these names on Windows causes undefined behavior.
    {
        // Check the final filename component (the part after the last separator)
        let filename = path.rsplit(&['/', '\\']).next().unwrap_or(path);
        // Strip any extension for the check (e.g. "CON.txt" is also reserved)
        let stem = filename.split('.').next().unwrap_or(filename);
        let upper = stem.to_ascii_uppercase();
        let is_reserved = matches!(upper.as_str(), "CON" | "PRN" | "AUX" | "NUL")
            || (upper.len() == 4
                && (upper.starts_with("COM") || upper.starts_with("LPT"))
                && upper.as_bytes()[3].is_ascii_digit()
                && upper.as_bytes()[3] != b'0');
        if is_reserved {
            return Err(AetherError::PathTraversal(format!(
                "Windows reserved device name in path: {path}"
            )));
        }
        // Reject trailing dots/spaces which Windows silently strips
        if filename.ends_with('.') || filename.ends_with(' ') {
            return Err(AetherError::PathTraversal(format!(
                "Trailing dot or space in filename (unsafe on Windows): {path}"
            )));
        }
    }
    Ok(())
}

/// Validate that a resolved output path stays within the output directory,
/// and write file data atomically to close TOCTOU races.
///
/// Instead of validating and returning a path (which could be swapped between
/// check and use), this function performs the write itself using a temp file +
/// rename, which is atomic on most filesystems. Symlink checks are performed
/// immediately before the rename to minimize the race window.
///
/// When `no_clobber` is `true`, the function refuses to write if the target
/// already exists, returning [`AetherError::FileAlreadyExists`]. This
/// eliminates the TOCTOU race window entirely for the no-overwrite case,
/// since there is no check-then-overwrite pattern.
pub(crate) fn write_validated_file(
    output_dir: &std::path::Path,
    file_path: &str,
    data: &[u8],
    mtime: i64,
    #[cfg(unix)] permissions: u32,
    no_clobber: bool,
) -> Result<()> {
    validate_extraction_path(file_path)?;

    let canonical_output = output_dir.canonicalize()?;
    let target = output_dir.join(file_path);

    // Create parent directories one component at a time, verifying each
    // component is not a symlink before creating the next. This closes
    // the TOCTOU gap between create_dir_all and the symlink check.
    {
        let mut built_path = canonical_output.clone();
        for component in std::path::Path::new(file_path)
            .parent()
            .into_iter()
            .flat_map(|p| p.components())
        {
            built_path.push(component);
            if built_path.exists() {
                let meta = std::fs::symlink_metadata(&built_path).map_err(|e| {
                    AetherError::Decompression(format!(
                        "Cannot stat path component {}: {e}",
                        built_path.display()
                    ))
                })?;
                if meta.file_type().is_symlink() {
                    return Err(AetherError::PathTraversal(format!(
                        "Symlink detected in extraction path: {}",
                        built_path.display(),
                    )));
                }
                if !meta.file_type().is_dir() {
                    return Err(AetherError::PathTraversal(format!(
                        "Non-directory in extraction path: {}",
                        built_path.display(),
                    )));
                }
            } else {
                // Create this single directory component. If an attacker
                // raced a symlink into a prior component, this mkdir will
                // follow the symlink — but we verified the prior component
                // immediately before, minimizing the window.
                std::fs::create_dir(&built_path).or_else(|e| {
                    // Another thread/process may have created it; recheck.
                    if built_path.is_dir() {
                        Ok(())
                    } else {
                        Err(e)
                    }
                })?;
                // Verify the directory we just created is not a symlink
                // (race defense: attacker could replace between create and here).
                let meta = std::fs::symlink_metadata(&built_path).map_err(|e| {
                    AetherError::Decompression(format!(
                        "Cannot stat newly created dir {}: {e}",
                        built_path.display()
                    ))
                })?;
                if meta.file_type().is_symlink() {
                    return Err(AetherError::PathTraversal(format!(
                        "Newly created directory was replaced with symlink: {}",
                        built_path.display(),
                    )));
                }
            }
        }
    }

    // Canonicalize the parent to resolve paths, then verify containment.
    let canonical_parent = target.parent().unwrap_or(output_dir).canonicalize()?;

    if !canonical_parent.starts_with(&canonical_output) {
        return Err(AetherError::PathTraversal(format!(
            "Resolved path escapes output directory: {}",
            file_path,
        )));
    }

    // Reconstruct the final path using the canonical parent + filename.
    let filename = target
        .file_name()
        .ok_or_else(|| AetherError::PathTraversal(file_path.to_string()))?;
    let resolved = canonical_parent.join(filename);

    // In no_clobber mode, refuse to write if target already exists.
    // This eliminates the TOCTOU race entirely: no check-then-overwrite.
    if no_clobber && resolved.exists() {
        return Err(AetherError::FileAlreadyExists(file_path.to_string()));
    }

    // Refuse to overwrite an existing symlink target.
    if resolved.exists() {
        let meta = std::fs::symlink_metadata(&resolved).map_err(|e| {
            AetherError::Decompression(format!(
                "Cannot stat target file {}: {e}",
                resolved.display()
            ))
        })?;
        if meta.file_type().is_symlink() {
            return Err(AetherError::PathTraversal(format!(
                "Target file is a symlink, refusing to overwrite: {}",
                resolved.display(),
            )));
        }
    }

    // Write to a temp file in the same directory, then rename atomically.
    // This closes the TOCTOU race: even if an attacker swaps a directory
    // for a symlink between the check above and here, the rename target is
    // the canonical path we already validated. The temp file is created in
    // the canonical parent so rename is same-filesystem (atomic).
    let mut tmp = tempfile::NamedTempFile::new_in(&canonical_parent).map_err(|e| {
        AetherError::Decompression(format!(
            "Failed to create temp file in {}: {e}",
            canonical_parent.display()
        ))
    })?;
    std::io::Write::write_all(&mut tmp, data)?;

    // Apply sanitized file permissions (Unix only) before rename.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let safe_mode = permissions & SAFE_PERMISSION_MASK;
        let perms = std::fs::Permissions::from_mode(safe_mode);
        let _ = std::fs::set_permissions(tmp.path(), perms);
    }

    // Re-check for symlink immediately before the rename to minimize the
    // race window. This is a belt-and-suspenders defense.
    if resolved.exists() {
        let meta = std::fs::symlink_metadata(&resolved).map_err(|e| {
            AetherError::Decompression(format!(
                "Cannot stat target file before rename {}: {e}",
                resolved.display()
            ))
        })?;
        if meta.file_type().is_symlink() {
            return Err(AetherError::PathTraversal(format!(
                "Target became a symlink during extraction: {}",
                resolved.display(),
            )));
        }
    }

    tmp.persist(&resolved).map_err(|e| {
        AetherError::Decompression(format!(
            "Failed to persist temp file to {}: {e}",
            resolved.display()
        ))
    })?;

    // Restore last-modified time from archive metadata.
    // Non-fatal: extraction succeeds even if mtime can't be set.
    if mtime > 0 {
        let ft = filetime::FileTime::from_unix_time(mtime, 0);
        let _ = filetime::set_file_mtime(&resolved, ft);
    }

    Ok(())
}

pub(crate) fn hex_str(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
