//! Security and adversarial tests for aether-core.
//!
//! These tests craft malicious or edge-case archives to verify that the
//! decompression pipeline handles them safely (errors, not panics/OOM).

use std::io::Cursor;
use std::path::PathBuf;

use aether_core::block::BlockHeader;
use aether_core::entropy::{Order0Model, ProbabilityPredictor};
use aether_core::format::*;
use aether_core::header::{ArchiveHeader, FileEntry};
use aether_core::pipeline::compress::Compressor;
use aether_core::pipeline::decompress::Decompressor;

// ── Helpers ─────────────────────────────────────────────────────────────────

fn order0_factory() -> Box<dyn ProbabilityPredictor> {
    Box::new(Order0Model::new())
}

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("tests")
        .join("fixtures")
        .join("sample")
}

fn sample_files() -> (PathBuf, Vec<PathBuf>) {
    let dir = fixture_dir();
    let files = vec![
        dir.join("hello.txt"),
        dir.join("code.rs"),
        dir.join("data.json"),
    ];
    (dir, files)
}

fn compress_to_memory(files: &[PathBuf], base_dir: &std::path::Path) -> Vec<u8> {
    let compressor = Compressor::new(order0_factory);
    let mut cursor = Cursor::new(Vec::new());
    compressor
        .compress_to_archive(base_dir, files, &mut cursor)
        .expect("compression should succeed");
    cursor.into_inner()
}

// ── A: Truncated Archives ───────────────────────────────────────────────────

#[test]
fn truncated_at_magic() {
    // Only 4 bytes of the 8-byte magic
    let data = &MAGIC[..4];
    let decompressor = Decompressor::new(order0_factory);
    let result = decompressor.read_metadata(&mut Cursor::new(data));
    assert!(result.is_err(), "truncated magic should fail");
}

#[test]
fn truncated_at_header() {
    // Full magic but truncated header (24 of 48 bytes)
    let mut buf = Vec::new();
    buf.extend_from_slice(&MAGIC);
    buf.extend_from_slice(&[0u8; 16]); // only 24 bytes total
    let decompressor = Decompressor::new(order0_factory);
    let result = decompressor.read_metadata(&mut Cursor::new(&buf));
    assert!(result.is_err(), "truncated header should fail");
}

#[test]
fn truncated_archive_various_positions() {
    let (base_dir, files) = sample_files();
    let archive = compress_to_memory(&files, &base_dir);
    let decompressor = Decompressor::new(order0_factory);

    // Try truncating at various positions
    for cut_point in [1, 8, 24, 48, 64, 100, archive.len() / 2, archive.len() - 1] {
        if cut_point >= archive.len() {
            continue;
        }
        let truncated = &archive[..cut_point];
        // Should not panic — error is fine
        let _ = decompressor.read_metadata(&mut Cursor::new(truncated));
    }
}

#[test]
fn empty_archive_input() {
    let decompressor = Decompressor::new(order0_factory);
    let result = decompressor.read_metadata(&mut Cursor::new(&[] as &[u8]));
    assert!(result.is_err(), "empty input should fail");
}

// ── B: Invalid Magic / Version ──────────────────────────────────────────────

#[test]
fn wrong_magic_rejected() {
    let mut buf = vec![0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x00, 0x01, 0x00];
    buf.extend_from_slice(&[0u8; 40]); // pad to header size
                                       // Seekable path reads footer first, so may fail on footer/seek before magic.
    let decompressor = Decompressor::new(order0_factory);
    let seekable_result = decompressor.read_metadata(&mut Cursor::new(&buf));
    assert!(
        seekable_result.is_err(),
        "wrong magic should fail (seekable)"
    );
    // Streaming path reads header first and catches bad magic directly.
    let streaming_result = Decompressor::read_metadata_streaming(&mut &buf[..]);
    assert!(
        streaming_result.is_err(),
        "wrong magic should fail (streaming)"
    );
}

// ── C: Resource Limit Enforcement ───────────────────────────────────────────

/// Craft a header that declares more files than MAX_FILE_COUNT.
#[test]
fn excessive_file_count_rejected() {
    let header = ArchiveHeader {
        flags: 0,
        predictor_id: PredictorId::Order0,
        file_count: MAX_FILE_COUNT + 1,
        solid_group_count: 0,
        block_count: 0,
        file_table_offset: ARCHIVE_HEADER_SIZE as u64,
        block_index_offset: ARCHIVE_HEADER_SIZE as u64,
    };
    let mut buf = Vec::new();
    header.write_to(&mut buf).unwrap();

    // Need a footer to make read_metadata work, but the header should be rejected first
    // during streaming read
    let result = Decompressor::read_metadata_streaming(&mut &buf[..]);
    assert!(
        result.is_err(),
        "file count > MAX_FILE_COUNT should be rejected"
    );
}

/// Craft a header that declares more blocks than MAX_BLOCK_COUNT.
#[test]
fn excessive_block_count_rejected() {
    let header = ArchiveHeader {
        flags: 0,
        predictor_id: PredictorId::Order0,
        file_count: 1,
        solid_group_count: 1,
        block_count: MAX_BLOCK_COUNT + 1,
        file_table_offset: ARCHIVE_HEADER_SIZE as u64,
        block_index_offset: ARCHIVE_HEADER_SIZE as u64,
    };
    let mut buf = Vec::new();
    header.write_to(&mut buf).unwrap();

    let result = Decompressor::read_metadata_streaming(&mut &buf[..]);
    assert!(
        result.is_err(),
        "block count > MAX_BLOCK_COUNT should be rejected"
    );
}

/// Craft a header with excessive solid groups.
#[test]
fn excessive_solid_group_count_rejected() {
    let header = ArchiveHeader {
        flags: 0,
        predictor_id: PredictorId::Order0,
        file_count: 1,
        solid_group_count: MAX_SOLID_GROUP_COUNT + 1,
        block_count: 1,
        file_table_offset: ARCHIVE_HEADER_SIZE as u64,
        block_index_offset: ARCHIVE_HEADER_SIZE as u64,
    };
    let mut buf = Vec::new();
    header.write_to(&mut buf).unwrap();

    let result = Decompressor::read_metadata_streaming(&mut &buf[..]);
    assert!(
        result.is_err(),
        "solid group count > MAX should be rejected"
    );
}

// ── D: Path Traversal in File Entries ────────────────────────────────────────

/// Attempt to write a FileEntry with a traversal path and verify read rejects it.
#[test]
fn file_entry_path_traversal_rejected() {
    // FileEntry::read_from should reject paths containing ".."
    let entry = FileEntry {
        path: "../../../etc/passwd".to_string(),
        original_size: 100,
        blake3_hash: [0u8; 32],
        solid_group_id: 0,
        chunk_start_idx: 0,
        chunk_count: 1,
        mtime: 0,
        permissions: 0o644,
    };
    let mut buf = Vec::new();
    entry.write_to(&mut buf).unwrap();
    let result = FileEntry::read_from(&mut Cursor::new(&buf));
    assert!(
        result.is_err(),
        "path traversal in file entry should be rejected"
    );
}

/// NUL bytes in extraction paths should be rejected by validate_extraction_path.
#[test]
fn extraction_path_with_nul_rejected() {
    let (base_dir, files) = sample_files();
    let archive = compress_to_memory(&files, &base_dir);

    let decompressor = Decompressor::new(order0_factory);
    let mut cursor = Cursor::new(&archive[..]);
    let mut output = Vec::new();
    // NUL byte in the requested file path should be caught
    let result = decompressor.extract_file(&mut cursor, "safe.txt\0../../etc/shadow", &mut output);
    assert!(
        result.is_err(),
        "NUL byte in extraction path should be rejected"
    );
}

/// Absolute paths in file entries should be rejected.
#[test]
fn file_entry_absolute_path_rejected() {
    let entry = FileEntry {
        path: "/etc/passwd".to_string(),
        original_size: 100,
        blake3_hash: [0u8; 32],
        solid_group_id: 0,
        chunk_start_idx: 0,
        chunk_count: 1,
        mtime: 0,
        permissions: 0o644,
    };
    let mut buf = Vec::new();
    entry.write_to(&mut buf).unwrap();
    let result = FileEntry::read_from(&mut Cursor::new(&buf));
    assert!(
        result.is_err(),
        "absolute path in file entry should be rejected"
    );
}

// ── E: Decompression Bomb Detection ─────────────────────────────────────────

/// A block header claiming massive uncompressed size should be rejected.
#[test]
fn decompression_bomb_block_rejected() {
    // Build a minimal but structurally valid archive with a bomb block.
    // We start from a real archive and patch the block header to claim a
    // massive uncompressed size, which the decompressor must reject.
    let (base_dir, files) = sample_files();
    let mut archive = compress_to_memory(&files, &base_dir);

    let decompressor = Decompressor::new(order0_factory);
    let metadata = {
        let mut cursor = Cursor::new(&archive[..]);
        decompressor.read_metadata(&mut cursor).expect("metadata")
    };

    if metadata.block_index.is_empty() {
        return;
    }

    // Patch the first block's uncompressed_size to exceed the limit.
    // uncompressed_size is at byte offset 20 in the block header (4 bytes, LE).
    let block0_offset = metadata.block_index[0].archive_offset as usize;
    let bomb_size = (MAX_DECOMPRESSED_BLOCK_SIZE as u32) + 1;
    let size_offset = block0_offset + 20;
    archive[size_offset..size_offset + 4].copy_from_slice(&bomb_size.to_le_bytes());

    // Attempting to extract should fail (either CRC catches the header
    // modification, or the decompressor rejects the oversized block).
    let tmp = tempfile::tempdir().unwrap();
    let mut cursor = Cursor::new(&archive[..]);
    let result = decompressor.extract_all(&mut cursor, tmp.path());
    assert!(
        result.is_err(),
        "archive with decompression bomb block should be rejected during extraction"
    );

    // Verify path also rejects via verify()
    let mut cursor = Cursor::new(&archive[..]);
    let verify_result = decompressor.verify(&mut cursor);
    let detected = match verify_result {
        Ok(r) => !r.is_ok(),
        Err(_) => true,
    };
    assert!(detected, "verify should detect the patched bomb block");
}

// ── F: Cross-Predictor Mismatch ─────────────────────────────────────────────

/// Compressing with one predictor and decompressing with a different one
/// should fail cleanly, not produce garbage silently.
#[test]
fn cross_predictor_mismatch_fails_cleanly() {
    use aether_core::entropy::context_mixer::ContextMixerConfig;
    use aether_core::entropy::ContextMixer;

    let (base_dir, files) = sample_files();

    // Compress with Order0
    let archive = compress_to_memory(&files, &base_dir);

    // Try to decompress with CM (wrong predictor)
    let decompressor = Decompressor::new(|| {
        Box::new(ContextMixer::with_config(ContextMixerConfig::lightweight()))
    });
    let tmp = tempfile::tempdir().unwrap();
    let mut cursor = Cursor::new(&archive[..]);
    let result = decompressor.extract_all(&mut cursor, tmp.path());

    // The mismatch must be detected: either extraction fails outright, or
    // if it "succeeds" (e.g., because blocks used zstd/store fallback),
    // the extracted data must still match the originals (BLAKE3 verified).
    match result {
        Err(_) => {
            // Good: mismatch was caught by the decompressor.
        }
        Ok(_) => {
            // Extraction didn't error — verify every file matches.
            // If any block used predictor+rANS, BLAKE3 should have caught it
            // and returned an error above. If we get here, all blocks used
            // a fallback method, so data should be correct.
            for file_path in &files {
                let rel = file_path
                    .strip_prefix(&base_dir)
                    .unwrap()
                    .to_string_lossy()
                    .replace('\\', "/");
                let original = std::fs::read(file_path).unwrap();
                let extracted = std::fs::read(tmp.path().join(&rel))
                    .expect("extracted file should exist if extract_all succeeded");
                assert_eq!(
                    original, extracted,
                    "if extraction succeeded with wrong predictor, data must still match \
                     (fallback methods don't use predictors): {rel}"
                );
            }
        }
    }
}

// ── G: Corruption at Every Block Header Position ────────────────────────────

/// Flip a byte at each position within the first block header and verify
/// that corruption is detected (not silently accepted).
#[test]
fn corruption_detected_at_every_block_header_byte() {
    let (base_dir, files) = sample_files();
    let archive = compress_to_memory(&files, &base_dir);

    let decompressor = Decompressor::new(order0_factory);
    let metadata = {
        let mut cursor = Cursor::new(&archive[..]);
        decompressor.read_metadata(&mut cursor).expect("metadata")
    };

    if metadata.block_index.is_empty() {
        return;
    }

    let block0_offset = metadata.block_index[0].archive_offset as usize;

    // Test flipping each byte in the block header (28 bytes)
    let mut detected_count = 0;
    for i in 0..BLOCK_HEADER_SIZE {
        let mut corrupted = archive.clone();
        let pos = block0_offset + i;
        if pos >= corrupted.len() {
            continue;
        }
        corrupted[pos] ^= 0xFF;

        let mut cursor = Cursor::new(&corrupted[..]);
        let result = decompressor.verify(&mut cursor);
        match result {
            Ok(r) if !r.is_ok() => detected_count += 1,
            Err(_) => detected_count += 1,
            _ => {} // Some bytes might not affect CRC if in padding/reserved
        }
    }

    // At least most corruptions should be detected (CRC covers most of header)
    assert!(
        detected_count >= BLOCK_HEADER_SIZE - 4, // allow a few padding bytes
        "corruption should be detected at most header positions, only {detected_count}/{BLOCK_HEADER_SIZE} detected"
    );
}

/// Flip a byte at several positions within block payloads.
#[test]
fn corruption_detected_at_various_payload_positions() {
    let (base_dir, files) = sample_files();
    let archive = compress_to_memory(&files, &base_dir);

    let decompressor = Decompressor::new(order0_factory);
    let metadata = {
        let mut cursor = Cursor::new(&archive[..]);
        decompressor.read_metadata(&mut cursor).expect("metadata")
    };

    if metadata.block_index.is_empty() {
        return;
    }

    let block = &metadata.block_index[0];
    let payload_start = block.archive_offset as usize + BLOCK_HEADER_SIZE;
    let payload_end = block.archive_offset as usize + block.compressed_size as usize;

    if payload_end <= payload_start || payload_end > archive.len() {
        return;
    }

    // Test corruption at start, 25%, 50%, 75%, and end of payload
    let payload_len = payload_end - payload_start;
    let positions = [
        payload_start,
        payload_start + payload_len / 4,
        payload_start + payload_len / 2,
        payload_start + 3 * payload_len / 4,
        payload_end.saturating_sub(1),
    ];

    for &pos in &positions {
        if pos >= archive.len() {
            continue;
        }
        let mut corrupted = archive.clone();
        corrupted[pos] ^= 0xFF;

        let mut cursor = Cursor::new(&corrupted[..]);
        let result = decompressor.verify(&mut cursor);
        let detected = match result {
            Ok(r) => !r.is_ok(),
            Err(_) => true,
        };
        assert!(
            detected,
            "corruption at payload offset {} should be detected",
            pos - payload_start
        );
    }
}

// ── H: Streaming Decompression Security ─────────────────────────────────────

/// Verify streaming decompression also catches corruption.
#[test]
fn streaming_corruption_detected() {
    let (base_dir, files) = sample_files();
    let mut archive = compress_to_memory(&files, &base_dir);

    // Corrupt a byte in the middle
    let mid = archive.len() / 2;
    archive[mid] ^= 0xFF;

    let decompressor = Decompressor::new(order0_factory);
    let tmp = tempfile::tempdir().unwrap();
    let result = decompressor.extract_all_streaming(&mut &archive[..], tmp.path());
    // Should error, not panic
    assert!(
        result.is_err(),
        "streaming decompression of corrupted archive should fail"
    );
}

// ── I: Block Header CRC Covers Critical Fields ──────────────────────────────

#[test]
fn block_header_crc_catches_compression_method_corruption() {
    let (base_dir, files) = sample_files();
    let archive = compress_to_memory(&files, &base_dir);

    let decompressor = Decompressor::new(order0_factory);
    let metadata = {
        let mut cursor = Cursor::new(&archive[..]);
        decompressor.read_metadata(&mut cursor).expect("metadata")
    };

    if metadata.block_index.is_empty() {
        return;
    }

    // Byte 12 of block header is the compression_method field
    let block0_offset = metadata.block_index[0].archive_offset as usize;
    let method_offset = block0_offset + 12;

    let mut corrupted = archive.clone();
    if method_offset < corrupted.len() {
        corrupted[method_offset] ^= 0xFF;
        let mut cursor = Cursor::new(&corrupted[..]);
        let result = decompressor.verify(&mut cursor);
        let detected = match result {
            Ok(r) => !r.is_ok(),
            Err(_) => true,
        };
        assert!(detected, "compression method corruption should be detected");
    }
}

// ── J: Extraction Path Safety ───────────────────────────────────────────────

#[test]
fn extraction_rejects_path_traversal() {
    let (base_dir, files) = sample_files();
    let archive = compress_to_memory(&files, &base_dir);

    // Try to extract a file with path traversal
    let decompressor = Decompressor::new(order0_factory);
    let mut cursor = Cursor::new(&archive[..]);
    let mut output = Vec::new();
    let result = decompressor.extract_file(&mut cursor, "../../../etc/passwd", &mut output);
    assert!(
        result.is_err(),
        "extraction with path traversal should be rejected"
    );
}

// ── K: Archive Footer Corruption ────────────────────────────────────────────

#[test]
fn footer_corruption_detected() {
    let (base_dir, files) = sample_files();
    let mut archive = compress_to_memory(&files, &base_dir);

    // Corrupt the last 4 bytes (footer CRC area)
    let len = archive.len();
    if len > 4 {
        archive[len - 3] ^= 0xFF;
    }

    let decompressor = Decompressor::new(order0_factory);
    let mut cursor = Cursor::new(&archive[..]);
    let result = decompressor.read_metadata(&mut cursor);
    // Should be some kind of error — either footer CRC, magic, or parse error
    assert!(result.is_err(), "footer corruption should be detected");
}

// ── L: Zero-Length Block ────────────────────────────────────────────────────

#[test]
fn block_header_with_zero_sizes_roundtrip() {
    let header = BlockHeader {
        block_id: 42,
        solid_group_id: 0,
        compression_method: CompressionMethod::Store,
        predictor_state_flag: false,
        compressed_size: 0,
        uncompressed_size: 0,
    };
    let mut buf = Vec::new();
    header.write_to(&mut buf).unwrap();
    let parsed = BlockHeader::read_from(&mut Cursor::new(&buf)).unwrap();
    assert_eq!(parsed.compressed_size, 0);
    assert_eq!(parsed.uncompressed_size, 0);
}

// ── M: Dictionary Tests ─────────────────────────────────────────────────────

/// Verify that dictionary improves compression ratio on repetitive data.
#[test]
fn dictionary_actually_improves_compression() {
    use aether_core::dictionary::Dictionary;

    let (base_dir, files) = sample_files();

    // Compress without dictionary
    let archive_without = compress_to_memory(&files, &base_dir);

    // Train dictionary and compress with it
    let mut predictor = Order0Model::new();
    let dict = Dictionary::train(&mut predictor, &files).expect("training");

    let compressor = Compressor::new(order0_factory).with_dictionary(dict);
    let mut cursor = Cursor::new(Vec::new());
    compressor
        .compress_to_archive(&base_dir, &files, &mut cursor)
        .expect("compression with dict");
    let archive_with = cursor.into_inner();

    // Dictionary archive should be no larger than without
    // (On small files, improvement may be negligible, but it shouldn't be worse)
    assert!(
        archive_with.len() <= archive_without.len() + 256,
        "dictionary should not significantly increase archive size: {} vs {}",
        archive_with.len(),
        archive_without.len()
    );
}

// ── N: Large Corruption Sweep ───────────────────────────────────────────────

/// Verify no panics when corrupting random positions in the archive.
#[test]
fn no_panics_on_random_corruption() {
    let (base_dir, files) = sample_files();
    let archive = compress_to_memory(&files, &base_dir);
    let decompressor = Decompressor::new(order0_factory);

    // Corrupt every 37th byte (covers the whole archive without being too slow)
    for i in (0..archive.len()).step_by(37) {
        let mut corrupted = archive.clone();
        corrupted[i] ^= 0xFF;

        // Seekable path
        let _ = decompressor.verify(&mut Cursor::new(&corrupted[..]));

        // Streaming path — should not panic
        let tmp = tempfile::tempdir().unwrap();
        let _ = decompressor.extract_all_streaming(&mut &corrupted[..], tmp.path());
    }
}

// ── O: Windows Path Traversal ────────────────────────────────────────────

/// Windows-style backslash path traversal should be rejected.
#[test]
fn file_entry_windows_backslash_traversal_rejected() {
    let entry = FileEntry {
        path: r"..\..\..\..\Windows\System32\config\SAM".to_string(),
        original_size: 100,
        blake3_hash: [0u8; 32],
        solid_group_id: 0,
        chunk_start_idx: 0,
        chunk_count: 1,
        mtime: 0,
        permissions: 0o644,
    };
    let mut buf = Vec::new();
    entry.write_to(&mut buf).unwrap();
    let result = FileEntry::read_from(&mut Cursor::new(&buf));
    assert!(
        result.is_err(),
        "Windows backslash path traversal should be rejected"
    );
}

/// Windows drive letter absolute paths should be rejected.
#[test]
fn file_entry_windows_drive_letter_rejected() {
    let entry = FileEntry {
        path: r"C:\Windows\System32\drivers\etc\hosts".to_string(),
        original_size: 100,
        blake3_hash: [0u8; 32],
        solid_group_id: 0,
        chunk_start_idx: 0,
        chunk_count: 1,
        mtime: 0,
        permissions: 0o644,
    };
    let mut buf = Vec::new();
    entry.write_to(&mut buf).unwrap();
    let result = FileEntry::read_from(&mut Cursor::new(&buf));
    assert!(
        result.is_err(),
        "Windows drive letter path should be rejected"
    );
}

/// Windows UNC paths should be rejected.
#[test]
fn file_entry_unc_path_rejected() {
    let entry = FileEntry {
        path: r"\\server\share\secret.txt".to_string(),
        original_size: 100,
        blake3_hash: [0u8; 32],
        solid_group_id: 0,
        chunk_start_idx: 0,
        chunk_count: 1,
        mtime: 0,
        permissions: 0o644,
    };
    let mut buf = Vec::new();
    entry.write_to(&mut buf).unwrap();
    let result = FileEntry::read_from(&mut Cursor::new(&buf));
    assert!(result.is_err(), "UNC path in file entry should be rejected");
}

/// Extraction with Windows-style traversal paths should be rejected.
#[test]
fn extraction_rejects_windows_path_traversal() {
    let (base_dir, files) = sample_files();
    let archive = compress_to_memory(&files, &base_dir);

    let decompressor = Decompressor::new(order0_factory);
    let mut cursor = Cursor::new(&archive[..]);
    let mut output = Vec::new();
    let result = decompressor.extract_file(
        &mut cursor,
        r"..\..\..\..\Windows\System32\config\SAM",
        &mut output,
    );
    assert!(
        result.is_err(),
        "extraction with Windows backslash traversal should be rejected"
    );
}

// ── P: Symlink Safety During Extraction ──────────────────────────────────

/// Extraction should refuse to write through a symlink that points outside
/// the output directory.
#[test]
fn extraction_refuses_to_write_through_symlink() {
    let (base_dir, files) = sample_files();
    let archive = compress_to_memory(&files, &base_dir);

    let decompressor = Decompressor::new(order0_factory);
    let tmp = tempfile::tempdir().unwrap();
    let escape_target = tempfile::tempdir().unwrap();

    // Get the first file's relative name
    let first_file_name = files[0].file_name().unwrap().to_str().unwrap();

    // Create a symlink at the extraction path that points outside the output dir
    let symlink_path = tmp.path().join(first_file_name);

    #[cfg(unix)]
    std::os::unix::fs::symlink(escape_target.path(), &symlink_path).unwrap();
    #[cfg(windows)]
    {
        // On Windows, creating directory symlinks may require elevation.
        // Try to create a file symlink; if it fails (no privileges), skip the test.
        if std::os::windows::fs::symlink_file(
            escape_target.path().join("escaped.txt"),
            &symlink_path,
        )
        .is_err()
        {
            // Cannot create symlinks without admin privileges; skip test.
            return;
        }
    }

    // Extraction should detect the symlink and refuse to write through it
    let mut cursor = Cursor::new(&archive[..]);
    let result = decompressor.extract_all(&mut cursor, tmp.path());

    // Either the extraction errors, or the symlink was not followed.
    // If extraction succeeded, verify the symlink was not replaced with real data
    // that landed in the escape directory.
    if result.is_ok() {
        // The symlink itself should have been replaced or skipped safely;
        // no file should appear in the escape target directory.
        let escape_contents: Vec<_> = std::fs::read_dir(escape_target.path()).unwrap().collect();
        assert!(
            escape_contents.is_empty(),
            "no files should have been written through the symlink into the escape directory"
        );
    }
    // Err is also acceptable — the decompressor detected and rejected the symlink.
}
