use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::{Read, Write};

use crate::error::{AetherError, Result};
use crate::format::*;

// ── Block Header (28 bytes) ──────────────────────────────────────────────────

/// Header preceding each compressed data block (28 bytes).
///
/// Contains the block's compression method, sizes, and a predictor state flag
/// that controls whether the decompressor should sync its predictor state
/// after decompressing this block.
///
/// # Integrity
///
/// The header uses CRC32 for corruption detection (bytes 24..28). CRC32 is
/// not cryptographically secure — an attacker can forge data matching any
/// target CRC32 in O(1). However, this is acceptable because:
///
/// 1. **Encrypted archives**: AEAD authentication covers the entire block
///    payload; header metadata fields (`compressed_size`, `uncompressed_size`)
///    are cross-validated against the block index, which is itself within the
///    AEAD-protected region of the archive.
///
/// 2. **Unencrypted archives**: Tampered size fields cause decompression
///    failures or content hash mismatches (the block trailer stores a full
///    32-byte BLAKE3 hash of the decompressed content). A forged header
///    that passes CRC32 but has wrong sizes cannot produce valid content.
///
/// 3. **Format constraint**: The 28-byte block header leaves only 4 bytes
///    for the integrity field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockHeader {
    /// Unique block identifier within the archive.
    pub block_id: u32,
    /// Solid group this block belongs to.
    pub solid_group_id: u32,
    /// Compression method used for this block's payload.
    pub compression_method: CompressionMethod,
    /// If `true`, predictor sync was skipped during compression (BWT won decisively).
    /// The decompressor must also skip sync to match.
    pub predictor_state_flag: bool,
    /// Size of the compressed payload in bytes.
    pub compressed_size: u32,
    /// Size of the original uncompressed data in bytes.
    pub uncompressed_size: u32,
}

impl BlockHeader {
    pub fn write_to<W: Write>(&self, w: &mut W) -> Result<()> {
        // Bytes 0..4: block magic
        w.write_u32::<LittleEndian>(BLOCK_MAGIC)?;
        // Bytes 4..8: block_id
        w.write_u32::<LittleEndian>(self.block_id)?;
        // Bytes 8..12: solid_group_id
        w.write_u32::<LittleEndian>(self.solid_group_id)?;
        // Byte 12: compression_method
        w.write_u8(self.compression_method as u8)?;
        // Byte 13: predictor_state_flag
        w.write_u8(if self.predictor_state_flag { 1 } else { 0 })?;
        // Bytes 14..16: reserved
        w.write_u16::<LittleEndian>(0)?;
        // Bytes 16..20: compressed_size
        w.write_u32::<LittleEndian>(self.compressed_size)?;
        // Bytes 20..24: uncompressed_size
        w.write_u32::<LittleEndian>(self.uncompressed_size)?;

        // Bytes 24..28: header CRC (over bytes 0..23)
        let crc = self.compute_crc();
        w.write_u32::<LittleEndian>(crc)?;

        Ok(())
    }

    /// Read a block header, using the given archive offset for error reporting.
    pub fn read_from_at<R: Read>(r: &mut R, archive_offset: u64) -> Result<Self> {
        let magic = r.read_u32::<LittleEndian>()?;
        if magic != BLOCK_MAGIC {
            return Err(AetherError::InvalidBlockMagic {
                offset: archive_offset,
            });
        }

        let block_id = r.read_u32::<LittleEndian>()?;
        let solid_group_id = r.read_u32::<LittleEndian>()?;
        let compression_method = CompressionMethod::from_u8(r.read_u8()?)?;
        let predictor_state_flag = r.read_u8()? != 0;
        let _reserved = r.read_u16::<LittleEndian>()?;
        let compressed_size = r.read_u32::<LittleEndian>()?;
        let uncompressed_size = r.read_u32::<LittleEndian>()?;
        let stored_crc = r.read_u32::<LittleEndian>()?;

        let header = BlockHeader {
            block_id,
            solid_group_id,
            compression_method,
            predictor_state_flag,
            compressed_size,
            uncompressed_size,
        };

        let computed_crc = header.compute_crc();
        if stored_crc != computed_crc {
            return Err(AetherError::BlockCrcMismatch { block_id });
        }

        Ok(header)
    }

    /// Read a block header from a stream (offset unknown, reported as 0 in errors).
    pub fn read_from<R: Read>(r: &mut R) -> Result<Self> {
        Self::read_from_at(r, 0)
    }

    fn compute_crc(&self) -> u32 {
        let mut buf = [0u8; 24];
        let mut pos = 0;
        buf[pos..pos + 4].copy_from_slice(&BLOCK_MAGIC.to_le_bytes());
        pos += 4;
        buf[pos..pos + 4].copy_from_slice(&self.block_id.to_le_bytes());
        pos += 4;
        buf[pos..pos + 4].copy_from_slice(&self.solid_group_id.to_le_bytes());
        pos += 4;
        buf[pos] = self.compression_method as u8;
        pos += 1;
        buf[pos] = if self.predictor_state_flag { 1 } else { 0 };
        pos += 1;
        buf[pos..pos + 2].copy_from_slice(&0u16.to_le_bytes());
        pos += 2; // reserved
        buf[pos..pos + 4].copy_from_slice(&self.compressed_size.to_le_bytes());
        pos += 4;
        buf[pos..pos + 4].copy_from_slice(&self.uncompressed_size.to_le_bytes());
        pos += 4;
        crc32fast::hash(&buf[..pos])
    }
}

// ── Block Trailer (36 bytes) ─────────────────────────────────────────────────

/// Trailer following each compressed data block (36 bytes).
///
/// Contains a BLAKE3 hash of the original uncompressed data for integrity
/// verification, followed by a CRC32 of the hash itself.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockTrailer {
    /// BLAKE3 hash of the original uncompressed block data.
    pub content_blake3: [u8; 32],
}

impl BlockTrailer {
    pub fn write_to<W: Write>(&self, w: &mut W) -> Result<()> {
        w.write_all(&self.content_blake3)?;
        let crc = crc32fast::hash(&self.content_blake3);
        w.write_u32::<LittleEndian>(crc)?;
        Ok(())
    }

    /// Read and validate a block trailer.
    ///
    /// `block_id` is used only for error reporting — pass the block ID from
    /// the preceding `BlockHeader` so CRC errors identify the right block.
    pub fn read_from_with_id<R: Read>(r: &mut R, block_id: u32) -> Result<Self> {
        let mut content_blake3 = [0u8; 32];
        r.read_exact(&mut content_blake3)?;
        let stored_crc = r.read_u32::<LittleEndian>()?;
        let computed_crc = crc32fast::hash(&content_blake3);
        if stored_crc != computed_crc {
            return Err(AetherError::BlockCrcMismatch { block_id });
        }
        Ok(BlockTrailer { content_blake3 })
    }

    /// Read and validate a block trailer (block_id defaults to 0 for error reporting).
    pub fn read_from<R: Read>(r: &mut R) -> Result<Self> {
        Self::read_from_with_id(r, 0)
    }
}

// ── Block Index Entry (24 bytes, fixed) ──────────────────────────────────────

/// One entry in the block index (24 bytes), enabling random-access seeking.
///
/// The block index is stored near the end of the archive, before the footer.
/// Each entry maps a block ID to its byte offset within the archive file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockIndexEntry {
    /// Unique block identifier.
    pub block_id: u32,
    /// Byte offset of this block's header within the archive file.
    pub archive_offset: u64,
    /// Compressed payload size in bytes.
    pub compressed_size: u32,
    /// Original uncompressed data size in bytes.
    pub uncompressed_size: u32,
    /// Solid group this block belongs to.
    pub solid_group_id: u32,
}

impl BlockIndexEntry {
    pub fn write_to<W: Write>(&self, w: &mut W) -> Result<()> {
        w.write_u32::<LittleEndian>(self.block_id)?;
        w.write_u64::<LittleEndian>(self.archive_offset)?;
        w.write_u32::<LittleEndian>(self.compressed_size)?;
        w.write_u32::<LittleEndian>(self.uncompressed_size)?;
        w.write_u32::<LittleEndian>(self.solid_group_id)?;
        Ok(())
    }

    pub fn read_from<R: Read>(r: &mut R) -> Result<Self> {
        Ok(BlockIndexEntry {
            block_id: r.read_u32::<LittleEndian>()?,
            archive_offset: r.read_u64::<LittleEndian>()?,
            compressed_size: r.read_u32::<LittleEndian>()?,
            uncompressed_size: r.read_u32::<LittleEndian>()?,
            solid_group_id: r.read_u32::<LittleEndian>()?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn block_header_roundtrip() {
        let header = BlockHeader {
            block_id: 42,
            solid_group_id: 3,
            compression_method: CompressionMethod::PredictorRans,
            predictor_state_flag: false,
            compressed_size: 1024,
            uncompressed_size: 4096,
        };

        let mut buf = Vec::new();
        header.write_to(&mut buf).unwrap();
        assert_eq!(buf.len(), BLOCK_HEADER_SIZE);

        let mut cursor = Cursor::new(&buf);
        let decoded = BlockHeader::read_from(&mut cursor).unwrap();
        assert_eq!(header, decoded);
    }

    #[test]
    fn block_header_bad_magic() {
        let mut buf = vec![0u8; BLOCK_HEADER_SIZE];
        // Write wrong magic
        buf[0..4].copy_from_slice(&0xDEADBEEFu32.to_le_bytes());
        let mut cursor = Cursor::new(&buf);
        assert!(matches!(
            BlockHeader::read_from(&mut cursor),
            Err(AetherError::InvalidBlockMagic { .. })
        ));
    }

    #[test]
    fn block_trailer_roundtrip() {
        let trailer = BlockTrailer {
            content_blake3: [0xCD; 32],
        };

        let mut buf = Vec::new();
        trailer.write_to(&mut buf).unwrap();
        assert_eq!(buf.len(), BLOCK_TRAILER_SIZE);

        let mut cursor = Cursor::new(&buf);
        let decoded = BlockTrailer::read_from(&mut cursor).unwrap();
        assert_eq!(trailer, decoded);
    }

    #[test]
    fn block_index_entry_roundtrip() {
        let entry = BlockIndexEntry {
            block_id: 0,
            archive_offset: 48 + 256,
            compressed_size: 512,
            uncompressed_size: 2048,
            solid_group_id: 1,
        };

        let mut buf = Vec::new();
        entry.write_to(&mut buf).unwrap();
        assert_eq!(buf.len(), BLOCK_INDEX_ENTRY_SIZE);

        let mut cursor = Cursor::new(&buf);
        let decoded = BlockIndexEntry::read_from(&mut cursor).unwrap();
        assert_eq!(entry, decoded);
    }

    #[test]
    fn block_header_all_methods() {
        for method in [
            CompressionMethod::PredictorRans,
            CompressionMethod::Zstd,
            CompressionMethod::Store,
            CompressionMethod::LzPredictorRans,
            CompressionMethod::Lz77PredictorRans,
            CompressionMethod::BwtPredictorRans,
        ] {
            let header = BlockHeader {
                block_id: 0,
                solid_group_id: 0,
                compression_method: method,
                predictor_state_flag: true,
                compressed_size: 100,
                uncompressed_size: 200,
            };

            let mut buf = Vec::new();
            header.write_to(&mut buf).unwrap();
            let mut cursor = Cursor::new(&buf);
            let decoded = BlockHeader::read_from(&mut cursor).unwrap();
            assert_eq!(header, decoded);
        }
    }
}
