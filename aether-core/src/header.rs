use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::{Read, Write};

use crate::error::{AetherError, Result};
use crate::format::*;

// ── Archive Header (48 bytes) ────────────────────────────────────────────────

/// Fixed-size preamble at the start of every `.aet` archive (48 bytes).
///
/// Contains the magic bytes, format flags, predictor identifier, and offsets
/// to the file table and block index for random-access reading.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArchiveHeader {
    /// Bitfield of `FLAG_*` constants (neural model, solid archive, encrypted).
    pub flags: u16,
    /// Which predictor was used as the default for this archive.
    pub predictor_id: PredictorId,
    /// Total number of files stored in the archive.
    pub file_count: u32,
    /// Number of solid groups.
    pub solid_group_count: u32,
    /// Total number of compressed blocks.
    pub block_count: u32,
    /// Byte offset of the file table within the archive.
    pub file_table_offset: u64,
    /// Byte offset of the block index within the archive.
    pub block_index_offset: u64,
}

impl ArchiveHeader {
    /// Serialize to writer. Writes exactly [`ARCHIVE_HEADER_SIZE`] bytes.
    pub fn write_to<W: Write>(&self, w: &mut W) -> Result<()> {
        // Bytes 0..8: magic
        w.write_all(&MAGIC)?;
        // Bytes 8..10: flags
        w.write_u16::<LittleEndian>(self.flags)?;
        // Bytes 10..12: predictor_id
        w.write_u16::<LittleEndian>(self.predictor_id as u16)?;
        // Bytes 12..16: file_count
        w.write_u32::<LittleEndian>(self.file_count)?;
        // Bytes 16..20: solid_group_count
        w.write_u32::<LittleEndian>(self.solid_group_count)?;
        // Bytes 20..24: block_count
        w.write_u32::<LittleEndian>(self.block_count)?;
        // Bytes 24..32: file_table_offset
        w.write_u64::<LittleEndian>(self.file_table_offset)?;
        // Bytes 32..40: block_index_offset
        w.write_u64::<LittleEndian>(self.block_index_offset)?;

        // Bytes 40..48: BLAKE3 integrity tag (truncated to 8 bytes).
        // Replaces the former CRC32 + 4-byte reserved field with a
        // cryptographic hash for stronger integrity checking.
        let tag = self.compute_blake3_tag();
        w.write_all(&tag)?;

        Ok(())
    }

    /// Deserialize from reader. Reads exactly [`ARCHIVE_HEADER_SIZE`] bytes.
    pub fn read_from<R: Read>(r: &mut R) -> Result<Self> {
        // Read magic
        let mut magic = [0u8; 8];
        r.read_exact(&mut magic)?;
        if magic != MAGIC && magic != LEGACY_MAGIC {
            return Err(AetherError::InvalidMagic);
        }

        let flags = r.read_u16::<LittleEndian>()?;
        let predictor_id = PredictorId::from_u16(r.read_u16::<LittleEndian>()?)?;
        let file_count = r.read_u32::<LittleEndian>()?;
        let solid_group_count = r.read_u32::<LittleEndian>()?;
        let block_count = r.read_u32::<LittleEndian>()?;
        let file_table_offset = r.read_u64::<LittleEndian>()?;
        let block_index_offset = r.read_u64::<LittleEndian>()?;
        let mut stored_tag = [0u8; 8];
        r.read_exact(&mut stored_tag)?;

        let header = ArchiveHeader {
            flags,
            predictor_id,
            file_count,
            solid_group_count,
            block_count,
            file_table_offset,
            block_index_offset,
        };

        // Verify BLAKE3 integrity tag
        let computed_tag = header.compute_blake3_tag();
        if stored_tag != computed_tag {
            return Err(AetherError::HeaderIntegrityMismatch);
        }

        // Resource limit checks on untrusted header values
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

        Ok(header)
    }

    /// Compute truncated BLAKE3 hash over the first 40 bytes of the header.
    ///
    /// Returns an 8-byte (64-bit) tag that fits in the same space as the
    /// former CRC32 + reserved fields.
    ///
    /// # Security note on 64-bit truncation
    ///
    /// The 64-bit tag provides ~2^64 collision resistance, which is sufficient
    /// for corruption detection but not for strong anti-tampering against a
    /// determined offline attacker. This is an acceptable tradeoff because:
    ///
    /// 1. **Encrypted archives**: The AEAD layer (AES-GCM / ChaCha20-Poly1305)
    ///    provides full 128-bit integrity for all block data. The encryption
    ///    header's Argon2 parameters are implicitly authenticated by the key
    ///    derivation chain — modifying them produces a different key, which
    ///    causes the BLAKE3 verification tag check to fail (constant-time).
    ///
    /// 2. **Unencrypted archives**: The header and footer cross-validate
    ///    redundant fields (file_count, block_count, offsets), and each block
    ///    has its own BLAKE3 content hash. Forging a header that passes the
    ///    64-bit tag AND produces valid downstream checksums is infeasible.
    ///
    /// 3. **Format constraint**: The 48-byte header size is fixed and cannot
    ///    be expanded without a major version bump.
    fn compute_blake3_tag(&self) -> [u8; 8] {
        let mut buf = [0u8; 40];
        let mut pos = 0;
        macro_rules! write_bytes {
            ($src:expr) => {
                let s = $src;
                buf[pos..pos + s.len()].copy_from_slice(&s);
                pos += s.len();
            };
        }
        write_bytes!(MAGIC);
        write_bytes!(self.flags.to_le_bytes());
        write_bytes!((self.predictor_id as u16).to_le_bytes());
        write_bytes!(self.file_count.to_le_bytes());
        write_bytes!(self.solid_group_count.to_le_bytes());
        write_bytes!(self.block_count.to_le_bytes());
        write_bytes!(self.file_table_offset.to_le_bytes());
        write_bytes!(self.block_index_offset.to_le_bytes());
        let hash = blake3::hash(&buf[..pos]);
        let mut tag = [0u8; 8];
        tag.copy_from_slice(&hash.as_bytes()[..8]);
        tag
    }
}

// ── File Entry (variable length) ─────────────────────────────────────────────

/// Metadata for a single file within the archive (variable length).
///
/// The file's content is spread across one or more contiguous blocks,
/// identified by `chunk_start_idx` and `chunk_count`. The BLAKE3 hash
/// covers the entire original file content for integrity verification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileEntry {
    /// Relative file path within the archive (UTF-8, forward-slash separated).
    pub path: String,
    /// Original uncompressed file size in bytes.
    pub original_size: u64,
    /// BLAKE3 hash of the complete original file content.
    pub blake3_hash: [u8; 32],
    /// Solid group this file belongs to.
    pub solid_group_id: u32,
    /// Index of the first block containing this file's data.
    pub chunk_start_idx: u32,
    /// Number of contiguous blocks containing this file's data.
    pub chunk_count: u32,
    /// Unix file permissions (e.g. 0o644).
    pub permissions: u32,
    /// Last modification time as Unix timestamp (seconds since epoch).
    pub mtime: i64,
}

impl FileEntry {
    fn shared_prefix_len(&self, previous_path: &str) -> usize {
        let path_bytes = self.path.as_bytes();
        let mut prefix_len = path_bytes
            .iter()
            .zip(previous_path.as_bytes())
            .take_while(|(left, right)| left == right)
            .count();
        while prefix_len > 0 && !self.path.is_char_boundary(prefix_len) {
            prefix_len -= 1;
        }
        prefix_len
    }

    /// Bytes used by the prefix-compressed path portion of this entry.
    pub fn prefixed_path_size(&self, previous_path: &str) -> usize {
        4 + self.path.len() - self.shared_prefix_len(previous_path)
    }

    pub fn write_to<W: Write>(&self, w: &mut W) -> Result<()> {
        let path_bytes = self.path.as_bytes();
        w.write_u16::<LittleEndian>(path_bytes.len() as u16)?;
        w.write_all(path_bytes)?;
        w.write_u64::<LittleEndian>(self.original_size)?;
        w.write_all(&self.blake3_hash)?;
        w.write_u32::<LittleEndian>(self.solid_group_id)?;
        w.write_u32::<LittleEndian>(self.chunk_start_idx)?;
        w.write_u32::<LittleEndian>(self.chunk_count)?;
        w.write_u32::<LittleEndian>(self.permissions)?;
        w.write_i64::<LittleEndian>(self.mtime)?;
        Ok(())
    }

    pub fn read_from<R: Read>(r: &mut R) -> Result<Self> {
        let path_len = r.read_u16::<LittleEndian>()? as usize;

        // Enforce MAX_PATH_LENGTH to reject crafted archives with huge paths
        if path_len > MAX_PATH_LENGTH {
            return Err(AetherError::ResourceLimitExceeded(format!(
                "File path length {path_len} exceeds maximum {MAX_PATH_LENGTH}",
            )));
        }

        let mut path_bytes = vec![0u8; path_len];
        r.read_exact(&mut path_bytes)?;
        let path = String::from_utf8(path_bytes)
            .map_err(|e| AetherError::Decompression(format!("Invalid UTF-8 path: {e}")))?;

        // Reject NUL bytes which could truncate paths at the OS level,
        // potentially bypassing subsequent traversal checks.
        if path.as_bytes().contains(&0) {
            return Err(AetherError::PathTraversal(path));
        }

        // Validate path safety: reject traversal attacks and absolute paths
        // early, so callers don't need to remember to check later.
        if path.starts_with('/') || path.starts_with('\\') {
            return Err(AetherError::PathTraversal(path));
        }
        if path.len() >= 2 && path.as_bytes()[1] == b':' {
            return Err(AetherError::PathTraversal(path));
        }
        for component in path.split(&['/', '\\']) {
            if component == ".." {
                return Err(AetherError::PathTraversal(path));
            }
        }

        let original_size = r.read_u64::<LittleEndian>()?;
        let mut blake3_hash = [0u8; 32];
        r.read_exact(&mut blake3_hash)?;
        let solid_group_id = r.read_u32::<LittleEndian>()?;
        let chunk_start_idx = r.read_u32::<LittleEndian>()?;
        let chunk_count = r.read_u32::<LittleEndian>()?;
        let permissions = r.read_u32::<LittleEndian>()?;

        // Sanitize permissions from untrusted archive data: mask to standard
        // Unix permission bits (rwx for owner/group/other = 0o777). Reject
        // setuid (0o4000), setgid (0o2000), and sticky (0o1000) bits which
        // could enable privilege escalation if applied during extraction.
        let permissions = permissions & SAFE_PERMISSION_MASK;

        let mtime = r.read_i64::<LittleEndian>()?;

        Ok(FileEntry {
            path,
            original_size,
            blake3_hash,
            solid_group_id,
            chunk_start_idx,
            chunk_count,
            permissions,
            mtime,
        })
    }

    /// Serialize using the longest UTF-8-safe byte prefix shared with the
    /// previous file-table path.
    pub fn write_prefixed<W: Write>(&self, w: &mut W, previous_path: &str) -> Result<()> {
        let path_bytes = self.path.as_bytes();
        let prefix_len = self.shared_prefix_len(previous_path);
        let suffix = &path_bytes[prefix_len..];
        w.write_u16::<LittleEndian>(
            u16::try_from(prefix_len).map_err(|_| {
                AetherError::ResourceLimitExceeded("Path prefix exceeds u16".into())
            })?,
        )?;
        w.write_u16::<LittleEndian>(
            u16::try_from(suffix.len()).map_err(|_| {
                AetherError::ResourceLimitExceeded("Path suffix exceeds u16".into())
            })?,
        )?;
        w.write_all(suffix)?;
        w.write_u64::<LittleEndian>(self.original_size)?;
        w.write_all(&self.blake3_hash)?;
        w.write_u32::<LittleEndian>(self.solid_group_id)?;
        w.write_u32::<LittleEndian>(self.chunk_start_idx)?;
        w.write_u32::<LittleEndian>(self.chunk_count)?;
        w.write_u32::<LittleEndian>(self.permissions)?;
        w.write_i64::<LittleEndian>(self.mtime)?;
        Ok(())
    }

    /// Deserialize a prefix-compressed file-table entry.
    pub fn read_prefixed<R: Read>(r: &mut R, previous_path: &str) -> Result<Self> {
        let prefix_len = r.read_u16::<LittleEndian>()? as usize;
        let suffix_len = r.read_u16::<LittleEndian>()? as usize;
        if prefix_len > previous_path.len()
            || !previous_path.is_char_boundary(prefix_len)
            || prefix_len.saturating_add(suffix_len) > MAX_PATH_LENGTH
        {
            return Err(AetherError::ResourceLimitExceeded(format!(
                "Invalid compressed path lengths: prefix={prefix_len}, suffix={suffix_len}"
            )));
        }

        let mut path = previous_path.as_bytes()[..prefix_len].to_vec();
        path.resize(prefix_len + suffix_len, 0);
        r.read_exact(&mut path[prefix_len..])?;

        // Reuse the canonical parser and all of its path/security checks.
        const FIXED_FIELDS_SIZE: usize = 64;
        let mut fixed_fields = [0u8; FIXED_FIELDS_SIZE];
        r.read_exact(&mut fixed_fields)?;
        let mut canonical = Vec::with_capacity(2 + path.len() + FIXED_FIELDS_SIZE);
        canonical.write_u16::<LittleEndian>(path.len() as u16)?;
        canonical.extend_from_slice(&path);
        canonical.extend_from_slice(&fixed_fields);
        Self::read_from(&mut std::io::Cursor::new(canonical))
    }
}

// ── Solid Group Entry (24 bytes, fixed) ──────────────────────────────────────

/// Describes one solid group (24 bytes): a set of semantically similar files
/// compressed together to maximize cross-file redundancy.
///
/// Files within a group share a predictor instance, allowing the predictor
/// to learn patterns across file boundaries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SolidGroupEntry {
    /// Unique group identifier.
    pub group_id: u32,
    /// Semantic content type of files in this group.
    pub content_type: ContentType,
    /// Dominant compression method for blocks in this group.
    pub compression_method: CompressionMethod,
    /// Index of the first block in this group.
    pub first_block_idx: u32,
    /// Number of blocks in this group.
    pub block_count: u32,
    /// Number of files in this group.
    pub file_count: u32,
}

impl SolidGroupEntry {
    pub fn write_to<W: Write>(&self, w: &mut W) -> Result<()> {
        w.write_u32::<LittleEndian>(self.group_id)?;
        w.write_u16::<LittleEndian>(self.content_type as u16)?;
        w.write_u16::<LittleEndian>(self.compression_method as u8 as u16)?;
        w.write_u32::<LittleEndian>(self.first_block_idx)?;
        w.write_u32::<LittleEndian>(self.block_count)?;
        w.write_u32::<LittleEndian>(self.file_count)?;
        w.write_u32::<LittleEndian>(0)?; // reserved
        Ok(())
    }

    pub fn read_from<R: Read>(r: &mut R) -> Result<Self> {
        let group_id = r.read_u32::<LittleEndian>()?;
        let content_type = ContentType::from_u16(r.read_u16::<LittleEndian>()?)?;
        let method_raw = r.read_u16::<LittleEndian>()?;
        let compression_method = CompressionMethod::from_u8(method_raw as u8)?;
        let first_block_idx = r.read_u32::<LittleEndian>()?;
        let block_count = r.read_u32::<LittleEndian>()?;
        let file_count = r.read_u32::<LittleEndian>()?;
        let _reserved = r.read_u32::<LittleEndian>()?;

        Ok(SolidGroupEntry {
            group_id,
            content_type,
            compression_method,
            first_block_idx,
            block_count,
            file_count,
        })
    }
}

// ── Archive Footer (32 bytes) ────────────────────────────────────────────────

/// Fixed-size trailer at the end of every `.aet` archive (32 bytes).
///
/// Contains redundant pointers (duplicated from the header) that allow
/// reading the archive from the end — seek to EOF-32, read the footer,
/// then jump to the block index and file table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArchiveFooter {
    /// Byte offset of the block index within the archive.
    pub block_index_offset: u64,
    /// Byte offset of the file table within the archive.
    pub file_table_offset: u64,
    /// Total number of compressed blocks (redundant with header).
    pub block_count: u32,
    /// Total number of files (redundant with header).
    pub file_count: u32,
}

impl ArchiveFooter {
    pub fn write_to<W: Write>(&self, w: &mut W) -> Result<()> {
        w.write_u64::<LittleEndian>(self.block_index_offset)?;
        w.write_u64::<LittleEndian>(self.file_table_offset)?;
        w.write_u32::<LittleEndian>(self.block_count)?;
        w.write_u32::<LittleEndian>(self.file_count)?;

        // CRC32 of bytes 0..23
        let crc = self.compute_crc();
        w.write_u32::<LittleEndian>(crc)?;

        // Footer magic
        w.write_u32::<LittleEndian>(FOOTER_MAGIC)?;

        Ok(())
    }

    pub fn read_from<R: Read>(r: &mut R) -> Result<Self> {
        let block_index_offset = r.read_u64::<LittleEndian>()?;
        let file_table_offset = r.read_u64::<LittleEndian>()?;
        let block_count = r.read_u32::<LittleEndian>()?;
        let file_count = r.read_u32::<LittleEndian>()?;
        let stored_crc = r.read_u32::<LittleEndian>()?;
        let magic = r.read_u32::<LittleEndian>()?;

        if magic != FOOTER_MAGIC {
            return Err(AetherError::InvalidFooterMagic);
        }

        let footer = ArchiveFooter {
            block_index_offset,
            file_table_offset,
            block_count,
            file_count,
        };

        let computed_crc = footer.compute_crc();
        if stored_crc != computed_crc {
            return Err(AetherError::HeaderCrcMismatch {
                expected: stored_crc,
                actual: computed_crc,
            });
        }

        Ok(footer)
    }

    fn compute_crc(&self) -> u32 {
        let mut buf = [0u8; 24];
        let mut pos = 0;
        buf[pos..pos + 8].copy_from_slice(&self.block_index_offset.to_le_bytes());
        pos += 8;
        buf[pos..pos + 8].copy_from_slice(&self.file_table_offset.to_le_bytes());
        pos += 8;
        buf[pos..pos + 4].copy_from_slice(&self.block_count.to_le_bytes());
        pos += 4;
        buf[pos..pos + 4].copy_from_slice(&self.file_count.to_le_bytes());
        pos += 4;
        crc32fast::hash(&buf[..pos])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn archive_header_roundtrip() {
        let header = ArchiveHeader {
            flags: FLAG_SOLID_ARCHIVE,
            predictor_id: PredictorId::ContextMixer,
            file_count: 42,
            solid_group_count: 3,
            block_count: 100,
            file_table_offset: 48,
            block_index_offset: 999_999,
        };

        let mut buf = Vec::new();
        header.write_to(&mut buf).unwrap();
        assert_eq!(buf.len(), ARCHIVE_HEADER_SIZE);

        let mut cursor = Cursor::new(&buf);
        let decoded = ArchiveHeader::read_from(&mut cursor).unwrap();
        assert_eq!(header, decoded);
    }

    #[test]
    fn archive_header_bad_magic() {
        let mut buf = vec![0u8; ARCHIVE_HEADER_SIZE];
        buf[0] = 0xFF; // corrupt magic
        let mut cursor = Cursor::new(&buf);
        assert!(matches!(
            ArchiveHeader::read_from(&mut cursor),
            Err(AetherError::InvalidMagic)
        ));
    }

    #[test]
    fn archive_header_bad_integrity() {
        let header = ArchiveHeader {
            flags: 0,
            predictor_id: PredictorId::Order0,
            file_count: 1,
            solid_group_count: 1,
            block_count: 1,
            file_table_offset: 48,
            block_index_offset: 100,
        };

        let mut buf = Vec::new();
        header.write_to(&mut buf).unwrap();

        // Corrupt one data byte (not the integrity tag itself)
        buf[12] ^= 0xFF;

        let mut cursor = Cursor::new(&buf);
        assert!(matches!(
            ArchiveHeader::read_from(&mut cursor),
            Err(AetherError::HeaderIntegrityMismatch)
        ));
    }

    #[test]
    fn file_entry_roundtrip() {
        let entry = FileEntry {
            path: "src/main.rs".into(),
            original_size: 12345,
            blake3_hash: [0xAB; 32],
            solid_group_id: 0,
            chunk_start_idx: 0,
            chunk_count: 3,
            permissions: 0o644,
            mtime: 1700000000,
        };

        let mut buf = Vec::new();
        entry.write_to(&mut buf).unwrap();

        let mut cursor = Cursor::new(&buf);
        let decoded = FileEntry::read_from(&mut cursor).unwrap();
        assert_eq!(entry, decoded);
    }

    #[test]
    fn file_entry_prefix_roundtrip() {
        let entry = FileEntry {
            path: "src/components/renderer.rs".into(),
            original_size: 42,
            blake3_hash: [7; 32],
            solid_group_id: 2,
            chunk_start_idx: 3,
            chunk_count: 1,
            permissions: 0o644,
            mtime: 1_700_000_000,
        };
        let previous = "src/components/parser.rs";
        let mut buffer = Vec::new();
        entry.write_prefixed(&mut buffer, previous).unwrap();
        let decoded = FileEntry::read_prefixed(&mut Cursor::new(buffer), previous).unwrap();
        assert_eq!(entry, decoded);
    }

    #[test]
    fn solid_group_entry_roundtrip() {
        let entry = SolidGroupEntry {
            group_id: 7,
            content_type: ContentType::Text,
            compression_method: CompressionMethod::PredictorRans,
            first_block_idx: 10,
            block_count: 5,
            file_count: 3,
        };

        let mut buf = Vec::new();
        entry.write_to(&mut buf).unwrap();
        assert_eq!(buf.len(), SOLID_GROUP_ENTRY_SIZE);

        let mut cursor = Cursor::new(&buf);
        let decoded = SolidGroupEntry::read_from(&mut cursor).unwrap();
        assert_eq!(entry, decoded);
    }

    #[test]
    fn archive_footer_roundtrip() {
        let footer = ArchiveFooter {
            block_index_offset: 123456,
            file_table_offset: 48,
            block_count: 50,
            file_count: 10,
        };

        let mut buf = Vec::new();
        footer.write_to(&mut buf).unwrap();
        assert_eq!(buf.len(), ARCHIVE_FOOTER_SIZE);

        let mut cursor = Cursor::new(&buf);
        let decoded = ArchiveFooter::read_from(&mut cursor).unwrap();
        assert_eq!(footer, decoded);
    }

    #[test]
    fn footer_bad_magic() {
        let footer = ArchiveFooter {
            block_index_offset: 100,
            file_table_offset: 48,
            block_count: 1,
            file_count: 1,
        };

        let mut buf = Vec::new();
        footer.write_to(&mut buf).unwrap();

        // Corrupt footer magic (last 4 bytes)
        let len = buf.len();
        buf[len - 1] ^= 0xFF;

        let mut cursor = Cursor::new(&buf);
        assert!(matches!(
            ArchiveFooter::read_from(&mut cursor),
            Err(AetherError::InvalidFooterMagic)
        ));
    }
}
