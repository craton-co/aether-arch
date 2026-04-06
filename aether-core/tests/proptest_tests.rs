use proptest::prelude::*;
use std::io::Cursor;

use aether_core::block::{BlockHeader, BlockIndexEntry, BlockTrailer};
use aether_core::format::{shannon_entropy, CompressionMethod, ContentType, PredictorId};
use aether_core::header::FileEntry;

// ── Strategies ───────────────────────────────────────────────────────────────

fn arb_compression_method() -> impl Strategy<Value = CompressionMethod> {
    prop_oneof![
        Just(CompressionMethod::PredictorRans),
        Just(CompressionMethod::Zstd),
        Just(CompressionMethod::Store),
        Just(CompressionMethod::LzPredictorRans),
        Just(CompressionMethod::Lz77PredictorRans),
        Just(CompressionMethod::BwtPredictorRans),
        Just(CompressionMethod::BytePlanePredictorRans),
    ]
}

fn arb_block_header() -> impl Strategy<Value = BlockHeader> {
    (
        any::<u32>(),
        any::<u32>(),
        arb_compression_method(),
        any::<bool>(),
        any::<u32>(),
        any::<u32>(),
    )
        .prop_map(
            |(
                block_id,
                solid_group_id,
                compression_method,
                predictor_state_flag,
                compressed_size,
                uncompressed_size,
            )| {
                BlockHeader {
                    block_id,
                    solid_group_id,
                    compression_method,
                    predictor_state_flag,
                    compressed_size,
                    uncompressed_size,
                }
            },
        )
}

fn arb_block_trailer() -> impl Strategy<Value = BlockTrailer> {
    any::<[u8; 32]>().prop_map(|content_blake3| BlockTrailer { content_blake3 })
}

fn arb_block_index_entry() -> impl Strategy<Value = BlockIndexEntry> {
    (
        any::<u32>(),
        any::<u64>(),
        any::<u32>(),
        any::<u32>(),
        any::<u32>(),
    )
        .prop_map(
            |(block_id, archive_offset, compressed_size, uncompressed_size, solid_group_id)| {
                BlockIndexEntry {
                    block_id,
                    archive_offset,
                    compressed_size,
                    uncompressed_size,
                    solid_group_id,
                }
            },
        )
}

// ── Property Tests ───────────────────────────────────────────────────────────

proptest! {
    // 1. BlockHeader roundtrip
    #[test]
    fn block_header_roundtrip(header in arb_block_header()) {
        let mut buf = Vec::new();
        header.write_to(&mut buf).unwrap();

        let mut cursor = Cursor::new(&buf);
        let decoded = BlockHeader::read_from(&mut cursor).unwrap();

        prop_assert_eq!(header, decoded);
    }

    // 2. BlockTrailer roundtrip
    #[test]
    fn block_trailer_roundtrip(trailer in arb_block_trailer()) {
        let mut buf = Vec::new();
        trailer.write_to(&mut buf).unwrap();

        let mut cursor = Cursor::new(&buf);
        let decoded = BlockTrailer::read_from(&mut cursor).unwrap();

        prop_assert_eq!(trailer, decoded);
    }

    // 3. BlockIndexEntry roundtrip
    #[test]
    fn block_index_entry_roundtrip(entry in arb_block_index_entry()) {
        let mut buf = Vec::new();
        entry.write_to(&mut buf).unwrap();

        let mut cursor = Cursor::new(&buf);
        let decoded = BlockIndexEntry::read_from(&mut cursor).unwrap();

        prop_assert_eq!(entry, decoded);
    }

    // 4. CompressionMethod::from_u8 never panics
    #[test]
    fn compression_method_from_u8_never_panics(v in any::<u8>()) {
        let result = CompressionMethod::from_u8(v);
        // Valid values 0..=6 should return Ok, everything else Err
        if v <= 6 {
            prop_assert!(result.is_ok());
        } else {
            prop_assert!(result.is_err());
        }
    }

    // 5. PredictorId::from_u16 never panics
    #[test]
    fn predictor_id_from_u16_never_panics(v in any::<u16>()) {
        // Must not panic for any input; Ok or Err is fine
        let _result = PredictorId::from_u16(v);
    }

    // 6. ContentType::from_u16 never panics
    #[test]
    fn content_type_from_u16_never_panics(v in any::<u16>()) {
        // Must not panic for any input; Ok or Err is fine
        let _result = ContentType::from_u16(v);
    }

    // 7. Shannon entropy is always in [0.0, 8.0]
    #[test]
    fn entropy_in_valid_range(data in proptest::collection::vec(any::<u8>(), 0..4096)) {
        let e = shannon_entropy(&data);
        prop_assert!(e >= 0.0, "entropy {} < 0.0", e);
        prop_assert!(e <= 8.0, "entropy {} > 8.0", e);
        prop_assert!(!e.is_nan(), "entropy is NaN");
    }

    // 8. FileEntry roundtrip for safe paths (no traversal, no absolute, no NUL)
    #[test]
    fn file_entry_roundtrip(
        depth in 0usize..3,
        segments in proptest::collection::vec("[a-z0-9_]{1,16}", 1..4),
        original_size in any::<u64>(),
        blake3_hash in any::<[u8; 32]>(),
        solid_group_id in any::<u32>(),
        chunk_start_idx in any::<u32>(),
        chunk_count in any::<u32>(),
        permissions in any::<u32>(),
        mtime in any::<i64>(),
    ) {
        // Build a safe relative path from random segments
        let path = if depth == 0 {
            segments[0].clone()
        } else {
            segments.iter().take(depth + 1).cloned().collect::<Vec<_>>().join("/")
        };

        let entry = FileEntry {
            path: path.clone(),
            original_size,
            blake3_hash,
            solid_group_id,
            chunk_start_idx,
            chunk_count,
            permissions,
            mtime,
        };

        let mut buf = Vec::new();
        entry.write_to(&mut buf).unwrap();

        let mut cursor = Cursor::new(&buf);
        let decoded = FileEntry::read_from(&mut cursor).unwrap();

        prop_assert_eq!(&entry.path, &decoded.path);
        prop_assert_eq!(entry.original_size, decoded.original_size);
        prop_assert_eq!(entry.blake3_hash, decoded.blake3_hash);
        prop_assert_eq!(entry.solid_group_id, decoded.solid_group_id);
        prop_assert_eq!(entry.chunk_start_idx, decoded.chunk_start_idx);
        prop_assert_eq!(entry.chunk_count, decoded.chunk_count);
    }

    // 9. FileEntry rejects adversarial paths without panicking
    #[test]
    fn file_entry_adversarial_path_never_panics(
        path in ".*",
        original_size in any::<u64>(),
    ) {
        let entry = FileEntry {
            path,
            original_size,
            blake3_hash: [0u8; 32],
            solid_group_id: 0,
            chunk_start_idx: 0,
            chunk_count: 1,
            permissions: 0o644,
            mtime: 0,
        };

        let mut buf = Vec::new();
        // write_to may succeed or fail depending on path validity
        if entry.write_to(&mut buf).is_ok() && !buf.is_empty() {
            // read_from must not panic — Ok or Err is fine
            let _ = FileEntry::read_from(&mut Cursor::new(&buf));
        }
    }
}
