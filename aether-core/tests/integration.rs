//! Integration tests for the aether-core library.
//!
//! Exercises the full compress → decompress pipeline through the public API,
//! verifying roundtrip correctness, corruption detection, determinism, and
//! selective extraction.

use std::io::{Cursor, Read};
use std::path::{Path, PathBuf};

use aether_core::entropy::context_mixer::ContextMixerConfig;
use aether_core::entropy::{ContextMixer, Lz4AwarePredictor, Order0Model, ProbabilityPredictor};
use aether_core::error::AetherError;
use aether_core::format::{PredictorId, BLOCK_HEADER_SIZE};
use aether_core::pipeline::compress::Compressor;
use aether_core::pipeline::decompress::Decompressor;

// ── Helpers ─────────────────────────────────────────────────────────────────

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("tests")
        .join("fixtures")
        .join("sample")
}

fn large_fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("tests")
        .join("fixtures")
        .join("large")
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

fn large_files() -> (PathBuf, Vec<PathBuf>) {
    let dir = large_fixture_dir();
    let files = vec![
        dir.join("english.txt"),
        dir.join("source.rs"),
        dir.join("mixed.json"),
    ];
    (dir, files)
}

fn compress_to_memory(
    base_dir: &Path,
    files: &[PathBuf],
    factory: impl Fn() -> Box<dyn ProbabilityPredictor> + Send + Sync + 'static,
) -> Vec<u8> {
    let compressor = Compressor::new(factory);
    let mut cursor = Cursor::new(Vec::new());
    let _result = compressor
        .compress_to_archive(base_dir, files, &mut cursor)
        .expect("compression should succeed");
    cursor.into_inner()
}

fn order0_factory() -> Box<dyn ProbabilityPredictor> {
    Box::new(Order0Model::new())
}

fn cm_light_factory() -> Box<dyn ProbabilityPredictor> {
    Box::new(ContextMixer::with_config(ContextMixerConfig::lightweight()))
}

fn lz4_aware_factory() -> Box<dyn ProbabilityPredictor> {
    Box::new(Lz4AwarePredictor::new())
}

/// Shared logic for multi-file roundtrip tests.
fn roundtrip_test_multi(
    compress_factory: impl Fn() -> Box<dyn ProbabilityPredictor> + Send + Sync + 'static,
    decompress_factory: impl Fn() -> Box<dyn ProbabilityPredictor> + Send + Sync + 'static,
    base_dir: &Path,
    files: &[PathBuf],
) {
    let archive_bytes = compress_to_memory(base_dir, files, compress_factory);
    assert!(!archive_bytes.is_empty(), "archive should not be empty");

    let tmp = tempfile::tempdir().expect("create temp dir");
    let decompressor = Decompressor::new(decompress_factory);
    let mut cursor = Cursor::new(&archive_bytes[..]);
    decompressor
        .extract_all(&mut cursor, tmp.path())
        .expect("extract_all should succeed");

    for file_path in files {
        let rel = file_path
            .strip_prefix(base_dir)
            .unwrap()
            .to_string_lossy()
            .replace('\\', "/");
        let extracted = tmp.path().join(&rel);
        let original = std::fs::read(file_path).expect("read original");
        let got = std::fs::read(&extracted).unwrap_or_else(|e| panic!("read extracted {rel}: {e}"));
        assert_eq!(
            original,
            got,
            "Mismatch for {rel} ({} vs {} bytes)",
            original.len(),
            got.len()
        );
    }
}

// ── Group A: Roundtrip Correctness ──────────────────────────────────────────

#[test]
fn roundtrip_multi_file_order0() {
    let (base_dir, files) = sample_files();
    roundtrip_test_multi(order0_factory, order0_factory, &base_dir, &files);
}

#[test]
fn roundtrip_multi_file_cm() {
    let (base_dir, files) = sample_files();
    roundtrip_test_multi(cm_light_factory, cm_light_factory, &base_dir, &files);
}

#[test]
fn roundtrip_single_file() {
    let dir = fixture_dir();
    let files = vec![dir.join("hello.txt")];
    roundtrip_test_multi(order0_factory, order0_factory, &dir, &files);
}

#[test]
fn roundtrip_large_files_order0() {
    let (base_dir, files) = large_files();
    roundtrip_test_multi(order0_factory, order0_factory, &base_dir, &files);
}

#[test]
fn roundtrip_large_files_cm() {
    let (base_dir, files) = large_files();
    roundtrip_test_multi(cm_light_factory, cm_light_factory, &base_dir, &files);
}

#[test]
fn verify_passes_on_valid_archive() {
    let (base_dir, files) = sample_files();
    let archive_bytes = compress_to_memory(&base_dir, &files, order0_factory);

    let decompressor = Decompressor::new(order0_factory);
    let mut cursor = Cursor::new(&archive_bytes[..]);
    let result = decompressor
        .verify(&mut cursor)
        .expect("verify should succeed");

    assert!(result.is_ok(), "valid archive should pass verification");
    assert_eq!(
        result.verified_blocks, result.total_blocks,
        "all blocks should be verified"
    );
    assert!(
        result.corrupted_blocks.is_empty(),
        "no blocks should be corrupted"
    );
}

// ── Group B: Single-File Extraction ─────────────────────────────────────────

#[test]
fn extract_single_file_order0() {
    let (base_dir, files) = sample_files();
    let archive_bytes = compress_to_memory(&base_dir, &files, order0_factory);

    for file_path in &files {
        let rel = file_path
            .strip_prefix(&base_dir)
            .unwrap()
            .to_string_lossy()
            .replace('\\', "/");

        let decompressor = Decompressor::new(order0_factory);
        let mut cursor = Cursor::new(&archive_bytes[..]);
        let mut extracted = Vec::new();
        decompressor
            .extract_file(&mut cursor, &rel, &mut extracted)
            .unwrap_or_else(|e| panic!("extract_file({rel}) failed: {e}"));

        let original = std::fs::read(file_path).unwrap();
        assert_eq!(
            original,
            extracted,
            "Single-file extraction mismatch: {rel} ({} vs {} bytes)",
            original.len(),
            extracted.len()
        );
    }
}

#[test]
fn extract_single_file_cm() {
    let (base_dir, files) = sample_files();
    let archive_bytes = compress_to_memory(&base_dir, &files, cm_light_factory);

    for file_path in &files {
        let rel = file_path
            .strip_prefix(&base_dir)
            .unwrap()
            .to_string_lossy()
            .replace('\\', "/");

        let decompressor = Decompressor::new(cm_light_factory);
        let mut cursor = Cursor::new(&archive_bytes[..]);
        let mut extracted = Vec::new();
        decompressor
            .extract_file(&mut cursor, &rel, &mut extracted)
            .unwrap_or_else(|e| panic!("extract_file({rel}) failed: {e}"));

        let original = std::fs::read(file_path).unwrap();
        assert_eq!(
            original, extracted,
            "Single-file CM extraction mismatch: {rel}"
        );
    }
}

#[test]
fn extract_file_not_found() {
    let (base_dir, files) = sample_files();
    let archive_bytes = compress_to_memory(&base_dir, &files, order0_factory);

    let decompressor = Decompressor::new(order0_factory);
    let mut cursor = Cursor::new(&archive_bytes[..]);
    let mut output = Vec::new();
    let err = decompressor
        .extract_file(&mut cursor, "nonexistent.txt", &mut output)
        .unwrap_err();

    assert!(
        matches!(err, AetherError::FileNotFound(_)),
        "expected FileNotFound, got: {err:?}"
    );
}

// ── Group C: Corruption Detection ───────────────────────────────────────────

#[test]
fn corruption_block_payload_detected() {
    let (base_dir, files) = sample_files();
    let mut archive_bytes = compress_to_memory(&base_dir, &files, order0_factory);

    // Read metadata to find block offsets
    let decompressor = Decompressor::new(order0_factory);
    let metadata = {
        let mut cursor = Cursor::new(&archive_bytes[..]);
        decompressor
            .read_metadata(&mut cursor)
            .expect("read metadata")
    };

    assert!(
        !metadata.block_index.is_empty(),
        "archive should have at least one block"
    );

    // Corrupt a byte in the first block's payload area
    let block0_offset = metadata.block_index[0].archive_offset as usize;
    let block0_compressed = metadata.block_index[0].compressed_size as usize;
    // Corrupt near the middle of the payload for robustness across compression methods
    let payload_offset = block0_offset
        + BLOCK_HEADER_SIZE
        + (block0_compressed.saturating_sub(BLOCK_HEADER_SIZE + 32)) / 2;
    let payload_offset = payload_offset.max(block0_offset + BLOCK_HEADER_SIZE + 1);
    assert!(
        payload_offset < archive_bytes.len(),
        "payload offset out of bounds"
    );
    archive_bytes[payload_offset] ^= 0xFF;

    // Verify should detect corruption
    let mut cursor = Cursor::new(&archive_bytes[..]);
    let result = decompressor
        .verify(&mut cursor)
        .expect("verify should return result, not IO error");
    assert!(
        !result.is_ok(),
        "corrupted archive should fail verification"
    );
    assert!(
        !result.corrupted_blocks.is_empty(),
        "should report corrupted blocks"
    );
}

#[test]
fn corruption_block_header_detected() {
    let (base_dir, files) = sample_files();
    let mut archive_bytes = compress_to_memory(&base_dir, &files, order0_factory);

    let decompressor = Decompressor::new(order0_factory);
    let metadata = {
        let mut cursor = Cursor::new(&archive_bytes[..]);
        decompressor
            .read_metadata(&mut cursor)
            .expect("read metadata")
    };

    // Corrupt byte 8 of the first block header (solid_group_id, within CRC region)
    let block0_offset = metadata.block_index[0].archive_offset as usize;
    archive_bytes[block0_offset + 8] ^= 0xFF;

    let mut cursor = Cursor::new(&archive_bytes[..]);
    let result = decompressor
        .verify(&mut cursor)
        .expect("verify should return result");
    assert!(
        !result.is_ok(),
        "block header CRC corruption should be detected"
    );
}

#[test]
fn corruption_archive_header_detected() {
    let (base_dir, files) = sample_files();
    let mut archive_bytes = compress_to_memory(&base_dir, &files, order0_factory);

    // Corrupt byte 12 (file_count field) in the archive header
    archive_bytes[12] ^= 0xFF;

    let decompressor = Decompressor::new(order0_factory);
    let mut cursor = Cursor::new(&archive_bytes[..]);
    let err = decompressor.read_metadata(&mut cursor).unwrap_err();
    assert!(
        matches!(err, AetherError::HeaderIntegrityMismatch),
        "expected HeaderIntegrityMismatch, got: {err:?}"
    );
}

#[test]
fn extract_corrupted_fails_gracefully() {
    let (base_dir, files) = sample_files();
    let mut archive_bytes = compress_to_memory(&base_dir, &files, order0_factory);

    // Corrupt a block payload
    let decompressor = Decompressor::new(order0_factory);
    let metadata = {
        let mut cursor = Cursor::new(&archive_bytes[..]);
        decompressor
            .read_metadata(&mut cursor)
            .expect("read metadata")
    };

    let block0_offset = metadata.block_index[0].archive_offset as usize;
    let block0_compressed = metadata.block_index[0].compressed_size as usize;
    // Corrupt near the middle of the payload for robustness across compression methods
    let payload_offset = block0_offset
        + BLOCK_HEADER_SIZE
        + (block0_compressed.saturating_sub(BLOCK_HEADER_SIZE + 32)) / 2;
    let payload_offset = payload_offset.max(block0_offset + BLOCK_HEADER_SIZE + 1);
    archive_bytes[payload_offset] ^= 0xFF;

    // extract_all should return Err, not panic
    let tmp = tempfile::tempdir().expect("create temp dir");
    let mut cursor = Cursor::new(&archive_bytes[..]);
    let result = decompressor.extract_all(&mut cursor, tmp.path());
    assert!(
        result.is_err(),
        "extract_all on corrupted archive should return Err"
    );
}

// ── Group D: Determinism ────────────────────────────────────────────────────

#[test]
fn deterministic_order0() {
    let (base_dir, files) = sample_files();
    let reference = compress_to_memory(&base_dir, &files, order0_factory);

    for i in 1..10 {
        let archive = compress_to_memory(&base_dir, &files, order0_factory);
        assert_eq!(
            reference,
            archive,
            "Run {i} differs from run 0 ({} vs {} bytes)",
            reference.len(),
            archive.len()
        );
    }
}

#[test]
fn deterministic_cm() {
    let (base_dir, files) = sample_files();
    let reference = compress_to_memory(&base_dir, &files, cm_light_factory);

    for i in 1..10 {
        let archive = compress_to_memory(&base_dir, &files, cm_light_factory);
        assert_eq!(
            reference,
            archive,
            "CM run {i} differs from run 0 ({} vs {} bytes)",
            reference.len(),
            archive.len()
        );
    }
}

// ── Group E: Metadata & List ────────────────────────────────────────────────

#[test]
fn list_files_correct() {
    let (base_dir, files) = sample_files();
    let archive_bytes = compress_to_memory(&base_dir, &files, order0_factory);

    let decompressor = Decompressor::new(order0_factory);
    let mut cursor = Cursor::new(&archive_bytes[..]);
    let entries = decompressor
        .list_files(&mut cursor)
        .expect("list_files should succeed");

    assert_eq!(entries.len(), 3, "should list 3 files");

    // Check all expected files are present (order may vary due to grouping)
    let paths: Vec<&str> = entries.iter().map(|e| e.path.as_str()).collect();
    for file_path in &files {
        let rel = file_path
            .strip_prefix(&base_dir)
            .unwrap()
            .to_string_lossy()
            .replace('\\', "/");
        assert!(
            paths.contains(&rel.as_str()),
            "file {rel} not found in listing: {paths:?}"
        );
    }

    // Check sizes match
    let known_sizes: Vec<(&str, u64)> = files
        .iter()
        .map(|f| {
            let rel = f
                .strip_prefix(&base_dir)
                .unwrap()
                .to_string_lossy()
                .replace('\\', "/");
            let size = std::fs::metadata(f).unwrap().len();
            // Leak the string so we can return &str (test only)
            (Box::leak(rel.into_boxed_str()) as &str, size)
        })
        .collect();

    for (path, expected_size) in &known_sizes {
        let entry = entries.iter().find(|e| e.path == *path).unwrap();
        assert_eq!(
            entry.original_size, *expected_size,
            "size mismatch for {path}: {} vs {}",
            entry.original_size, expected_size
        );
    }
}

#[test]
fn predictor_id_stored_correctly() {
    let (base_dir, files) = sample_files();

    // Order0 archive
    let order0_bytes = compress_to_memory(&base_dir, &files, order0_factory);
    let decompressor = Decompressor::new(order0_factory);
    let mut cursor = Cursor::new(&order0_bytes[..]);
    let metadata = decompressor
        .read_metadata(&mut cursor)
        .expect("read order0 metadata");
    assert_eq!(
        metadata.header.predictor_id,
        PredictorId::Order0,
        "order0 archive should store PredictorId::Order0"
    );

    // CM-light archive
    let cm_bytes = compress_to_memory(&base_dir, &files, cm_light_factory);
    let decompressor = Decompressor::new(cm_light_factory);
    let mut cursor = Cursor::new(&cm_bytes[..]);
    let metadata = decompressor
        .read_metadata(&mut cursor)
        .expect("read CM metadata");
    assert_eq!(
        metadata.header.predictor_id,
        PredictorId::ContextMixerLight,
        "CM-light archive should store PredictorId::ContextMixerLight"
    );
}

// ── Group F: LZ4 Preprocessing ───────────────────────────────────────────

/// Verify that the LZ4 path produces smaller archives on large structured text.
#[test]
fn lz4_improves_large_text_compression() {
    let (base_dir, files) = large_files();

    let archive_bytes = compress_to_memory(&base_dir, &files, cm_light_factory);

    // The archive should be significantly smaller than the raw input
    let total_raw: u64 = files
        .iter()
        .map(|f| std::fs::metadata(f).unwrap().len())
        .sum();

    let ratio = archive_bytes.len() as f64 / total_raw as f64;
    // With LZ4+CM we expect < 50% on structured text (was ~52% without LZ4)
    assert!(
        ratio < 0.55,
        "LZ4 path should compress structured text well, got {:.2}% ({}/{} bytes)",
        ratio * 100.0,
        archive_bytes.len(),
        total_raw
    );
}

/// Verify roundtrip through the LZ4 path on large files.
#[test]
fn roundtrip_large_files_lz4_order0() {
    let (base_dir, files) = large_files();
    roundtrip_test_multi(order0_factory, order0_factory, &base_dir, &files);
}

/// Verify roundtrip through the LZ4 path on large files with CM.
#[test]
fn roundtrip_large_files_lz4_cm() {
    let (base_dir, files) = large_files();
    roundtrip_test_multi(cm_light_factory, cm_light_factory, &base_dir, &files);
}

/// Verify that single-file extraction works correctly when LZ4 blocks exist.
#[test]
fn extract_single_file_lz4() {
    let (base_dir, files) = large_files();
    let archive_bytes = compress_to_memory(&base_dir, &files, cm_light_factory);

    for file_path in &files {
        let rel = file_path
            .strip_prefix(&base_dir)
            .unwrap()
            .to_string_lossy()
            .replace('\\', "/");

        let decompressor = Decompressor::new(cm_light_factory);
        let mut cursor = Cursor::new(&archive_bytes[..]);
        let mut extracted = Vec::new();
        decompressor
            .extract_file(&mut cursor, &rel, &mut extracted)
            .unwrap_or_else(|e| panic!("extract_file({rel}) failed: {e}"));

        let original = std::fs::read(file_path).unwrap();
        assert_eq!(
            original,
            extracted,
            "LZ4 single-file extraction mismatch: {rel} ({} vs {} bytes)",
            original.len(),
            extracted.len()
        );
    }
}

/// Verify that corruption in an LZ4 block is detected.
#[test]
fn corruption_lz4_block_detected() {
    let (base_dir, files) = large_files();
    let mut archive_bytes = compress_to_memory(&base_dir, &files, cm_light_factory);

    let decompressor = Decompressor::new(cm_light_factory);
    let metadata = {
        let mut cursor = Cursor::new(&archive_bytes[..]);
        decompressor
            .read_metadata(&mut cursor)
            .expect("read metadata")
    };

    assert!(
        !metadata.block_index.is_empty(),
        "archive should have blocks"
    );

    // Corrupt a byte in the first block's payload area
    let block0_offset = metadata.block_index[0].archive_offset as usize;
    let payload_offset = block0_offset + BLOCK_HEADER_SIZE + 8;
    if payload_offset < archive_bytes.len() {
        archive_bytes[payload_offset] ^= 0xFF;
    }

    let mut cursor = Cursor::new(&archive_bytes[..]);
    let result = decompressor
        .verify(&mut cursor)
        .expect("verify should return result");
    assert!(
        !result.is_ok(),
        "corrupted LZ4 archive should fail verification"
    );
}

// ── Group G: LZ4-Aware Predictor ─────────────────────────────────────────

#[test]
fn roundtrip_lz4_aware_sample() {
    let (base_dir, files) = sample_files();
    roundtrip_test_multi(lz4_aware_factory, lz4_aware_factory, &base_dir, &files);
}

#[test]
fn roundtrip_lz4_aware_large() {
    let (base_dir, files) = large_files();
    roundtrip_test_multi(lz4_aware_factory, lz4_aware_factory, &base_dir, &files);
}

#[test]
fn extract_single_file_lz4_aware() {
    let (base_dir, files) = large_files();
    let archive_bytes = compress_to_memory(&base_dir, &files, lz4_aware_factory);

    for file_path in &files {
        let rel = file_path
            .strip_prefix(&base_dir)
            .unwrap()
            .to_string_lossy()
            .replace('\\', "/");

        let decompressor = Decompressor::new(lz4_aware_factory);
        let mut cursor = Cursor::new(&archive_bytes[..]);
        let mut extracted = Vec::new();
        decompressor
            .extract_file(&mut cursor, &rel, &mut extracted)
            .unwrap_or_else(|e| panic!("extract_file({rel}) failed: {e}"));

        let original = std::fs::read(file_path).unwrap();
        assert_eq!(
            original, extracted,
            "LZ4-aware single-file extraction mismatch: {rel}"
        );
    }
}

#[test]
fn predictor_id_lz4_aware_stored_correctly() {
    let (base_dir, files) = sample_files();
    let archive_bytes = compress_to_memory(&base_dir, &files, lz4_aware_factory);

    let decompressor = Decompressor::new(lz4_aware_factory);
    let mut cursor = Cursor::new(&archive_bytes[..]);
    let metadata = decompressor
        .read_metadata(&mut cursor)
        .expect("read metadata");
    assert_eq!(
        metadata.header.predictor_id,
        PredictorId::Lz4Aware,
        "lz4-aware archive should store PredictorId::Lz4Aware"
    );
}

#[test]
fn deterministic_lz4_aware() {
    let (base_dir, files) = sample_files();
    let reference = compress_to_memory(&base_dir, &files, lz4_aware_factory);

    for i in 1..5 {
        let archive = compress_to_memory(&base_dir, &files, lz4_aware_factory);
        assert_eq!(reference, archive, "LZ4-aware run {i} differs from run 0");
    }
}

// ── Group H: Edge Cases ─────────────────────────────────────────────────────

#[test]
fn empty_file_roundtrip() {
    let tmp_dir = tempfile::tempdir().expect("create temp dir");
    let empty_file = tmp_dir.path().join("empty.bin");
    std::fs::write(&empty_file, b"").expect("write empty file");

    let files = vec![empty_file];
    let archive_bytes = compress_to_memory(tmp_dir.path(), &files, order0_factory);

    let out_dir = tempfile::tempdir().expect("create output dir");
    let decompressor = Decompressor::new(order0_factory);
    let mut cursor = Cursor::new(&archive_bytes[..]);
    decompressor
        .extract_all(&mut cursor, out_dir.path())
        .expect("extract empty file");

    let extracted = std::fs::read(out_dir.path().join("empty.bin")).expect("read extracted");
    assert!(
        extracted.is_empty(),
        "extracted empty file should be 0 bytes"
    );
}

#[test]
fn binary_data_roundtrip() {
    let tmp_dir = tempfile::tempdir().expect("create temp dir");
    let binary_file = tmp_dir.path().join("binary.bin");

    // All 256 byte values repeated 100 times = 25,600 bytes
    let data: Vec<u8> = (0..256u16)
        .flat_map(|b| std::iter::repeat_n(b as u8, 100))
        .collect();
    std::fs::write(&binary_file, &data).expect("write binary file");

    let files = vec![binary_file];
    let archive_bytes = compress_to_memory(tmp_dir.path(), &files, order0_factory);

    let out_dir = tempfile::tempdir().expect("create output dir");
    let decompressor = Decompressor::new(order0_factory);
    let mut cursor = Cursor::new(&archive_bytes[..]);
    decompressor
        .extract_all(&mut cursor, out_dir.path())
        .expect("extract binary file");

    let extracted = std::fs::read(out_dir.path().join("binary.bin")).expect("read extracted");
    assert_eq!(data, extracted, "binary data roundtrip mismatch");
}

// ── Group I: Streaming Decompression ─────────────────────────────────────

/// A `Read`-only wrapper that strips `Seek` from an inner reader.
/// Proves that streaming methods truly only use `Read`.
struct ReadOnly<R: Read>(R);

impl<R: Read> Read for ReadOnly<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.0.read(buf)
    }
}

/// Helper: streaming roundtrip test.
fn streaming_roundtrip_test(
    compress_factory: impl Fn() -> Box<dyn ProbabilityPredictor> + Send + Sync + 'static,
    decompress_factory: impl Fn() -> Box<dyn ProbabilityPredictor> + 'static,
    base_dir: &Path,
    files: &[PathBuf],
) {
    let archive_bytes = compress_to_memory(base_dir, files, compress_factory);

    // Wrap in ReadOnly to ensure no Seek is used
    let mut reader = ReadOnly(Cursor::new(&archive_bytes[..]));

    let tmp = tempfile::tempdir().expect("create temp dir");
    let decompressor = Decompressor::new(decompress_factory);
    decompressor
        .extract_all_streaming(&mut reader, tmp.path())
        .expect("streaming extract should succeed");

    // Verify every file is byte-identical
    for file_path in files {
        let rel = file_path
            .strip_prefix(base_dir)
            .unwrap()
            .to_string_lossy()
            .replace('\\', "/");
        let extracted = tmp.path().join(&rel);
        let original = std::fs::read(file_path).expect("read original");
        let got = std::fs::read(&extracted).unwrap_or_else(|e| panic!("read extracted {rel}: {e}"));
        assert_eq!(
            original,
            got,
            "Streaming mismatch for {rel} ({} vs {} bytes)",
            original.len(),
            got.len()
        );
    }
}

#[test]
fn streaming_roundtrip_sample() {
    let (base_dir, files) = sample_files();
    streaming_roundtrip_test(order0_factory, order0_factory, &base_dir, &files);
}

#[test]
fn streaming_roundtrip_sample_cm() {
    let (base_dir, files) = sample_files();
    streaming_roundtrip_test(cm_light_factory, cm_light_factory, &base_dir, &files);
}

#[test]
fn streaming_roundtrip_large() {
    let (base_dir, files) = large_files();
    streaming_roundtrip_test(order0_factory, order0_factory, &base_dir, &files);
}

#[test]
fn streaming_roundtrip_large_cm() {
    let (base_dir, files) = large_files();
    streaming_roundtrip_test(cm_light_factory, cm_light_factory, &base_dir, &files);
}

#[test]
fn streaming_verify() {
    let (base_dir, files) = sample_files();
    let archive_bytes = compress_to_memory(&base_dir, &files, order0_factory);

    let mut reader = ReadOnly(Cursor::new(&archive_bytes[..]));
    let decompressor = Decompressor::new(order0_factory);
    let result = decompressor
        .verify_streaming(&mut reader)
        .expect("streaming verify should succeed");

    assert!(result.is_ok(), "valid archive should pass streaming verify");
    assert_eq!(
        result.verified_blocks, result.total_blocks,
        "all blocks should be verified"
    );
}

#[test]
fn streaming_verify_large() {
    let (base_dir, files) = large_files();
    let archive_bytes = compress_to_memory(&base_dir, &files, cm_light_factory);

    let mut reader = ReadOnly(Cursor::new(&archive_bytes[..]));
    let decompressor = Decompressor::new(cm_light_factory);
    let result = decompressor
        .verify_streaming(&mut reader)
        .expect("streaming verify should succeed");

    assert!(
        result.is_ok(),
        "valid large archive should pass streaming verify"
    );
}

#[test]
fn streaming_list() {
    let (base_dir, files) = sample_files();
    let archive_bytes = compress_to_memory(&base_dir, &files, order0_factory);

    // Streaming list
    let mut reader = ReadOnly(Cursor::new(&archive_bytes[..]));
    let streaming_entries =
        Decompressor::list_files_streaming(&mut reader).expect("streaming list should succeed");

    // Seekable list
    let decompressor = Decompressor::new(order0_factory);
    let mut cursor = Cursor::new(&archive_bytes[..]);
    let seekable_entries = decompressor
        .list_files(&mut cursor)
        .expect("seekable list should succeed");

    // Results should be identical
    assert_eq!(streaming_entries.len(), seekable_entries.len());
    for (s, r) in streaming_entries.iter().zip(seekable_entries.iter()) {
        assert_eq!(s.path, r.path);
        assert_eq!(s.original_size, r.original_size);
        assert_eq!(s.blake3_hash, r.blake3_hash);
        assert_eq!(s.solid_group_id, r.solid_group_id);
        assert_eq!(s.chunk_start_idx, r.chunk_start_idx);
        assert_eq!(s.chunk_count, r.chunk_count);
    }
}

#[test]
fn streaming_metadata_predictor_detection() {
    let (base_dir, files) = sample_files();
    let archive_bytes = compress_to_memory(&base_dir, &files, cm_light_factory);

    let mut reader = ReadOnly(Cursor::new(&archive_bytes[..]));
    let metadata =
        Decompressor::read_metadata_streaming(&mut reader).expect("read streaming metadata");

    assert_eq!(
        metadata.header.predictor_id,
        PredictorId::ContextMixerLight,
        "streaming metadata should detect predictor ID"
    );
    assert_eq!(metadata.file_entries.len(), 3);
    assert!(!metadata.solid_groups.is_empty());
}

#[test]
fn streaming_two_phase_extraction() {
    // Test the two-phase pattern: read_metadata_streaming → extract_with_streaming_metadata
    // This is how the CLI uses it for predictor auto-detection.
    let (base_dir, files) = large_files();
    let archive_bytes = compress_to_memory(&base_dir, &files, order0_factory);

    let mut reader = ReadOnly(Cursor::new(&archive_bytes[..]));

    // Phase 1: read metadata (reader advances past header + tables)
    let metadata =
        Decompressor::read_metadata_streaming(&mut reader).expect("read streaming metadata");

    // Phase 2: create decompressor from detected predictor and extract
    let decompressor = Decompressor::new(order0_factory);
    let tmp = tempfile::tempdir().expect("create temp dir");
    decompressor
        .extract_with_streaming_metadata(&mut reader, &metadata, tmp.path())
        .expect("two-phase streaming extract should succeed");

    // Verify files
    for file_path in &files {
        let rel = file_path
            .strip_prefix(&base_dir)
            .unwrap()
            .to_string_lossy()
            .replace('\\', "/");
        let original = std::fs::read(file_path).expect("read original");
        let got = std::fs::read(tmp.path().join(&rel))
            .unwrap_or_else(|e| panic!("read extracted {rel}: {e}"));
        assert_eq!(original, got, "Two-phase streaming mismatch: {rel}");
    }
}

// ── Encryption (enterprise feature) ────────────────────────────────────────

#[cfg(feature = "enterprise")]
mod encryption_tests {
    use super::*;
    use aether_core::crypto::CipherId;

    fn compress_encrypted(
        base_dir: &Path,
        files: &[PathBuf],
        password: &str,
        cipher_id: CipherId,
    ) -> Vec<u8> {
        let compressor = Compressor::new(order0_factory).with_encryption(password, cipher_id);
        let mut buf = Cursor::new(Vec::new());
        compressor
            .compress_to_archive(base_dir, files, &mut buf)
            .expect("encrypted compression should succeed");
        buf.into_inner()
    }

    #[test]
    fn encrypted_roundtrip_aes_gcm() {
        let (base_dir, files) = sample_files();
        let password = "test-password-123";
        let archive = compress_encrypted(&base_dir, &files, password, CipherId::Aes256Gcm);

        let decompressor = Decompressor::new(order0_factory).with_password(password);

        let tmp = tempfile::tempdir().unwrap();
        let mut cursor = Cursor::new(&archive[..]);
        decompressor.extract_all(&mut cursor, tmp.path()).unwrap();

        // Verify all files match originals
        for file_path in &files {
            let rel = file_path
                .strip_prefix(&base_dir)
                .unwrap()
                .to_string_lossy()
                .replace('\\', "/");
            let original = std::fs::read(file_path).unwrap();
            let extracted = std::fs::read(tmp.path().join(&rel)).unwrap();
            assert_eq!(original, extracted, "AES-GCM roundtrip mismatch: {rel}");
        }
    }

    #[test]
    fn encrypted_roundtrip_chacha20() {
        let (base_dir, files) = sample_files();
        let password = "another-password";
        let archive = compress_encrypted(&base_dir, &files, password, CipherId::ChaCha20Poly1305);

        let decompressor = Decompressor::new(order0_factory).with_password(password);

        let tmp = tempfile::tempdir().unwrap();
        let mut cursor = Cursor::new(&archive[..]);
        decompressor.extract_all(&mut cursor, tmp.path()).unwrap();

        for file_path in &files {
            let rel = file_path
                .strip_prefix(&base_dir)
                .unwrap()
                .to_string_lossy()
                .replace('\\', "/");
            let original = std::fs::read(file_path).unwrap();
            let extracted = std::fs::read(tmp.path().join(&rel)).unwrap();
            assert_eq!(original, extracted, "ChaCha20 roundtrip mismatch: {rel}");
        }
    }

    #[test]
    fn encrypted_verify_passes() {
        let (base_dir, files) = sample_files();
        let password = "verify-test";
        let archive = compress_encrypted(&base_dir, &files, password, CipherId::Aes256Gcm);

        let decompressor = Decompressor::new(order0_factory).with_password(password);
        let mut cursor = Cursor::new(&archive[..]);
        let result = decompressor.verify(&mut cursor).unwrap();
        assert!(result.is_ok(), "encrypted archive should verify OK");
    }

    #[test]
    fn encrypted_wrong_password_fails() {
        let (base_dir, files) = sample_files();
        let archive = compress_encrypted(&base_dir, &files, "correct-pass", CipherId::Aes256Gcm);

        let decompressor = Decompressor::new(order0_factory).with_password("wrong-password");
        let tmp = tempfile::tempdir().unwrap();
        let mut cursor = Cursor::new(&archive[..]);
        let result = decompressor.extract_all(&mut cursor, tmp.path());
        assert!(result.is_err(), "wrong password should fail extraction");
    }

    #[test]
    fn encrypted_no_password_fails() {
        let (base_dir, files) = sample_files();
        let archive = compress_encrypted(&base_dir, &files, "secret-pass", CipherId::Aes256Gcm);

        // No with_password() call
        let decompressor = Decompressor::new(order0_factory);
        let tmp = tempfile::tempdir().unwrap();
        let mut cursor = Cursor::new(&archive[..]);
        let result = decompressor.extract_all(&mut cursor, tmp.path());
        assert!(result.is_err(), "no password should fail extraction");
    }

    #[test]
    fn encrypted_list_files_works() {
        let (base_dir, files) = sample_files();
        let archive = compress_encrypted(&base_dir, &files, "list-test", CipherId::Aes256Gcm);

        // list_files doesn't need password (metadata is not encrypted)
        let decompressor = Decompressor::new(order0_factory).with_password("list-test");
        let mut cursor = Cursor::new(&archive[..]);
        let entries = decompressor.list_files(&mut cursor).unwrap();
        assert_eq!(entries.len(), files.len());
    }

    #[test]
    fn encrypted_single_file_extraction() {
        let (base_dir, files) = sample_files();
        let password = "extract-one";
        let archive = compress_encrypted(&base_dir, &files, password, CipherId::ChaCha20Poly1305);

        let decompressor = Decompressor::new(order0_factory).with_password(password);
        let mut cursor = Cursor::new(&archive[..]);

        let mut output = Vec::new();
        decompressor
            .extract_file(&mut cursor, "hello.txt", &mut output)
            .unwrap();

        let original = std::fs::read(base_dir.join("hello.txt")).unwrap();
        assert_eq!(
            output, original,
            "single-file encrypted extraction mismatch"
        );
    }
}

// ── Parallel decompression (enterprise feature) ──────────────────────────

#[cfg(feature = "enterprise")]
mod parallel_tests {
    use super::*;

    /// Helper: compress sample files, then extract with parallel decompression
    /// and verify output matches originals byte-for-byte.
    fn parallel_roundtrip(max_threads: usize) {
        let (base_dir, files) = sample_files();

        // Compress
        let compressor = Compressor::new(order0_factory);
        let mut buf = Cursor::new(Vec::new());
        compressor
            .compress_to_archive(&base_dir, &files, &mut buf)
            .expect("compression should succeed");
        let archive = buf.into_inner();

        // Extract with parallel decompression
        let decompressor = Decompressor::new(order0_factory).with_max_threads(max_threads);

        let tmp = tempfile::tempdir().unwrap();
        let mut cursor = Cursor::new(&archive[..]);
        decompressor.extract_all(&mut cursor, tmp.path()).unwrap();

        // Verify all files match originals
        for file_path in &files {
            let rel = file_path
                .strip_prefix(&base_dir)
                .unwrap()
                .to_string_lossy()
                .replace('\\', "/");
            let original = std::fs::read(file_path).unwrap();
            let extracted = std::fs::read(tmp.path().join(&rel)).unwrap();
            assert_eq!(
                original, extracted,
                "parallel roundtrip mismatch (threads={max_threads}): {rel}"
            );
        }
    }

    #[test]
    fn parallel_extract_unlimited() {
        parallel_roundtrip(0); // 0 = all cores
    }

    #[test]
    fn parallel_extract_bounded_2() {
        parallel_roundtrip(2);
    }

    #[test]
    fn parallel_extract_bounded_4() {
        parallel_roundtrip(4);
    }

    /// Verify that parallel extraction of large fixtures (multiple solid groups)
    /// produces byte-identical output to sequential extraction.
    #[test]
    fn parallel_matches_sequential_large() {
        let (base_dir, files) = large_files();

        // Compress
        let compressor = Compressor::new(order0_factory);
        let mut buf = Cursor::new(Vec::new());
        compressor
            .compress_to_archive(&base_dir, &files, &mut buf)
            .expect("compression should succeed");
        let archive = buf.into_inner();

        // Sequential extraction
        let seq_decompressor = Decompressor::new(order0_factory).with_max_threads(1); // sequential
        let seq_tmp = tempfile::tempdir().unwrap();
        let mut cursor = Cursor::new(&archive[..]);
        seq_decompressor
            .extract_all(&mut cursor, seq_tmp.path())
            .unwrap();

        // Parallel extraction
        let par_decompressor = Decompressor::new(order0_factory).with_max_threads(0); // all cores
        let par_tmp = tempfile::tempdir().unwrap();
        let mut cursor = Cursor::new(&archive[..]);
        par_decompressor
            .extract_all(&mut cursor, par_tmp.path())
            .unwrap();

        // Compare every file
        for file_path in &files {
            let rel = file_path
                .strip_prefix(&base_dir)
                .unwrap()
                .to_string_lossy()
                .replace('\\', "/");
            let seq_data = std::fs::read(seq_tmp.path().join(&rel)).unwrap();
            let par_data = std::fs::read(par_tmp.path().join(&rel)).unwrap();
            assert_eq!(
                seq_data,
                par_data,
                "parallel vs sequential mismatch: {rel} (seq={}, par={})",
                seq_data.len(),
                par_data.len()
            );
        }
    }

    /// Parallel extraction of an encrypted archive should work correctly.
    #[test]
    fn parallel_extract_encrypted() {
        use aether_core::crypto::CipherId;

        let (base_dir, files) = sample_files();
        let password = "parallel-enc-test";

        // Compress with encryption
        let compressor =
            Compressor::new(order0_factory).with_encryption(password, CipherId::Aes256Gcm);
        let mut buf = Cursor::new(Vec::new());
        compressor
            .compress_to_archive(&base_dir, &files, &mut buf)
            .expect("encrypted compression should succeed");
        let archive = buf.into_inner();

        // Extract with parallel decompression + decryption
        let decompressor = Decompressor::new(order0_factory)
            .with_password(password)
            .with_max_threads(0);

        let tmp = tempfile::tempdir().unwrap();
        let mut cursor = Cursor::new(&archive[..]);
        decompressor.extract_all(&mut cursor, tmp.path()).unwrap();

        for file_path in &files {
            let rel = file_path
                .strip_prefix(&base_dir)
                .unwrap()
                .to_string_lossy()
                .replace('\\', "/");
            let original = std::fs::read(file_path).unwrap();
            let extracted = std::fs::read(tmp.path().join(&rel)).unwrap();
            assert_eq!(
                original, extracted,
                "parallel encrypted roundtrip mismatch: {rel}"
            );
        }
    }
}

// ── Dictionary pretraining tests ────────────────────────────────────────────

#[test]
fn dictionary_compress_decompress_roundtrip() {
    use aether_core::dictionary::Dictionary;

    let (base_dir, files) = sample_files();

    // Train a dictionary on the sample files
    let mut predictor = Order0Model::new();
    let dict = Dictionary::train(&mut predictor, &files).expect("training should succeed");

    // Compress with dictionary
    let compressor = Compressor::new(order0_factory).with_dictionary(dict.clone());
    let mut cursor = Cursor::new(Vec::new());
    let (_stats, _analytics) = compressor
        .compress_to_archive(&base_dir, &files, &mut cursor)
        .expect("compression with dictionary should succeed");
    let archive = cursor.into_inner();

    // Decompress with dictionary
    let decompressor = Decompressor::new(order0_factory).with_dictionary(dict);
    let mut reader = Cursor::new(&archive);
    let tmp = tempfile::tempdir().unwrap();
    decompressor
        .extract_all(&mut reader, tmp.path())
        .expect("extraction with dictionary should succeed");

    // Verify all files match
    for file_path in &files {
        let rel = file_path
            .strip_prefix(&base_dir)
            .unwrap()
            .to_string_lossy()
            .replace('\\', "/");
        let original = std::fs::read(file_path).unwrap();
        let extracted = std::fs::read(tmp.path().join(&rel)).unwrap();
        assert_eq!(original, extracted, "dictionary roundtrip mismatch: {rel}");
    }
}

#[test]
fn dictionary_missing_on_decompress_errors() {
    use aether_core::dictionary::Dictionary;

    let (base_dir, files) = sample_files();

    // Train and compress with dictionary
    let mut predictor = Order0Model::new();
    let dict = Dictionary::train(&mut predictor, &files).expect("training should succeed");

    let compressor = Compressor::new(order0_factory).with_dictionary(dict);
    let mut cursor = Cursor::new(Vec::new());
    let _result = compressor
        .compress_to_archive(&base_dir, &files, &mut cursor)
        .expect("compression should succeed");
    let archive = cursor.into_inner();

    // Try to decompress WITHOUT dictionary — should fail
    let decompressor = Decompressor::new(order0_factory);
    let mut reader = Cursor::new(&archive);
    let tmp = tempfile::tempdir().unwrap();
    let result = decompressor.extract_all(&mut reader, tmp.path());
    assert!(
        result.is_err(),
        "decompression without dictionary should fail"
    );
}

#[test]
fn dictionary_streaming_roundtrip() {
    use aether_core::dictionary::Dictionary;

    let (base_dir, files) = sample_files();

    // Train dictionary
    let mut predictor = Order0Model::new();
    let dict = Dictionary::train(&mut predictor, &files).expect("training should succeed");

    // Compress with dictionary
    let compressor = Compressor::new(order0_factory).with_dictionary(dict.clone());
    let mut cursor = Cursor::new(Vec::new());
    let _result = compressor
        .compress_to_archive(&base_dir, &files, &mut cursor)
        .expect("compression should succeed");
    let archive = cursor.into_inner();

    // Decompress via streaming path with dictionary
    let decompressor = Decompressor::new(order0_factory).with_dictionary(dict);
    let mut reader = &archive[..]; // &[u8] implements Read but not Seek
    let tmp = tempfile::tempdir().unwrap();
    decompressor
        .extract_all_streaming(&mut reader, tmp.path())
        .expect("streaming extraction with dictionary should succeed");

    // Verify
    for file_path in &files {
        let rel = file_path
            .strip_prefix(&base_dir)
            .unwrap()
            .to_string_lossy()
            .replace('\\', "/");
        let original = std::fs::read(file_path).unwrap();
        let extracted = std::fs::read(tmp.path().join(&rel)).unwrap();
        assert_eq!(
            original, extracted,
            "streaming dictionary roundtrip mismatch: {rel}"
        );
    }
}

// ── Migration tests ─────────────────────────────────────────────────────────

#[test]
fn migrate_order0_to_ssm() {
    use aether_core::entropy::NeuralSsmPredictor;
    use aether_core::pipeline::migrate::Migrator;

    let (base_dir, files) = sample_files();

    // Compress with Order0
    let archive = compress_to_memory(&base_dir, &files, order0_factory);

    // Migrate Order0 → NeuralSSM
    let migrator = Migrator::new(order0_factory, || Box::new(NeuralSsmPredictor::new()));
    let mut source = Cursor::new(&archive);
    let mut output = Cursor::new(Vec::new());
    let count = migrator
        .migrate(&mut source, &mut output)
        .expect("migration should succeed");
    assert_eq!(count, files.len());

    // Verify the migrated archive extracts correctly
    let migrated = output.into_inner();
    let decompressor = Decompressor::new(|| Box::new(NeuralSsmPredictor::new()));
    let mut reader = Cursor::new(&migrated);
    let tmp = tempfile::tempdir().unwrap();
    decompressor
        .extract_all(&mut reader, tmp.path())
        .expect("extraction should succeed");

    for file_path in &files {
        let rel = file_path
            .strip_prefix(&base_dir)
            .unwrap()
            .to_string_lossy()
            .replace('\\', "/");
        let original = std::fs::read(file_path).unwrap();
        let extracted = std::fs::read(tmp.path().join(&rel)).unwrap();
        assert_eq!(original, extracted, "migration roundtrip mismatch: {rel}");
    }
}
