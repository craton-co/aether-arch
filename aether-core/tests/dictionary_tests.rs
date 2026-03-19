//! Dictionary tests: train/save/load roundtrip, hash validation, size limits,
//! and compress/decompress integration with dictionary pretraining.

use std::io::Cursor;
use std::path::PathBuf;

use aether_core::dictionary::Dictionary;
use aether_core::entropy::{NeuralSsmPredictor, Order0Model, ProbabilityPredictor};
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

fn sample_files() -> Vec<PathBuf> {
    let dir = fixture_dir();
    vec![
        dir.join("hello.txt"),
        dir.join("code.rs"),
        dir.join("data.json"),
    ]
}

// ── Train, save, load — verify hash matches ─────────────────────────────────

#[test]
fn train_save_load_roundtrip_order0() {
    let files = sample_files();
    let mut predictor = Order0Model::new();
    let dict = Dictionary::train(&mut predictor, &files).expect("training should succeed");

    // Hash should be non-zero
    assert!(
        dict.hash.iter().any(|&b| b != 0),
        "hash should not be all zeros"
    );
    assert!(!dict.state.is_empty(), "trained state should not be empty");

    // Save to temp file
    let tmp = tempfile::tempdir().unwrap();
    let dict_path = tmp.path().join("order0.aed");
    dict.save(&dict_path).expect("save should succeed");

    // Load back
    let loaded = Dictionary::load(&dict_path).expect("load should succeed");
    assert_eq!(loaded.hash, dict.hash, "hash should match after save/load");
    assert_eq!(
        loaded.state, dict.state,
        "state should match after save/load"
    );
    assert_eq!(loaded.predictor_id, dict.predictor_id);
}

#[test]
fn train_save_load_roundtrip_neural_ssm() {
    let files = sample_files();
    let mut predictor = NeuralSsmPredictor::new();
    let dict = Dictionary::train(&mut predictor, &files).expect("training should succeed");

    let tmp = tempfile::tempdir().unwrap();
    let dict_path = tmp.path().join("neural.aed");
    dict.save(&dict_path).expect("save should succeed");

    let loaded = Dictionary::load(&dict_path).expect("load should succeed");
    assert_eq!(loaded.hash, dict.hash);
    assert_eq!(loaded.state, dict.state);
}

#[test]
fn loaded_dictionary_applies_to_matching_predictor() {
    let files = sample_files();
    let mut predictor = Order0Model::new();
    let dict = Dictionary::train(&mut predictor, &files).unwrap();

    let tmp = tempfile::tempdir().unwrap();
    let dict_path = tmp.path().join("test.aed");
    dict.save(&dict_path).unwrap();

    let loaded = Dictionary::load(&dict_path).unwrap();

    // Apply to a fresh predictor of the same type
    let mut fresh = Order0Model::new();
    loaded
        .apply(&mut fresh)
        .expect("apply should succeed for matching predictor");

    // The pretrained predictor should produce the same predictions as the
    // original trained predictor.
    let p_trained = predictor.predict();
    let p_loaded = fresh.predict();
    for i in 0..256 {
        assert!(
            (p_trained[i] - p_loaded[i]).abs() < 1e-6,
            "prediction mismatch at byte {i}: trained={} loaded={}",
            p_trained[i],
            p_loaded[i],
        );
    }
}

// ── Mismatched predictor ────────────────────────────────────────────────────

#[test]
fn dictionary_with_mismatched_predictor_loads_but_apply_fails() {
    // Train a dictionary with Order0
    let files = sample_files();
    let mut predictor = Order0Model::new();
    let dict = Dictionary::train(&mut predictor, &files).unwrap();

    // Save and reload — loading succeeds because the file format is valid
    let tmp = tempfile::tempdir().unwrap();
    let dict_path = tmp.path().join("order0.aed");
    dict.save(&dict_path).unwrap();
    let loaded = Dictionary::load(&dict_path).unwrap();

    // Applying to a different predictor type should fail (predictor_id mismatch)
    let mut wrong_predictor = NeuralSsmPredictor::new();
    let result = loaded.apply(&mut wrong_predictor);
    assert!(
        result.is_err(),
        "applying Order0 dictionary to NeuralSsm predictor should fail"
    );
}

// ── Dictionary file too large ───────────────────────────────────────────────

#[test]
fn dictionary_state_exceeding_max_size_rejected_on_load() {
    use aether_core::format::PredictorId;
    use byteorder::{LittleEndian, WriteBytesExt};

    // Craft a dictionary header that claims a state_len > 64 MiB
    let huge_len: u32 = 64 * 1024 * 1024 + 1; // one byte over the limit

    let mut buf = Vec::new();
    buf.extend_from_slice(&[0x41, 0x45, 0x44, 0x58]); // "AEDX" magic
    buf.push(1); // version
    buf.write_u16::<LittleEndian>(PredictorId::Order0 as u16)
        .unwrap();
    buf.write_u32::<LittleEndian>(huge_len).unwrap();
    // We don't need to write the full state — read_from should reject
    // before attempting to allocate.

    let result = Dictionary::read_from(&mut Cursor::new(&buf));
    assert!(
        result.is_err(),
        "dictionary with state > 64 MiB should be rejected"
    );
}

#[test]
fn dictionary_state_exceeding_max_size_rejected_on_save() {
    use aether_core::format::PredictorId;

    // Construct a Dictionary with an oversized state vector
    let oversized_state = vec![0u8; 64 * 1024 * 1024 + 1];
    let hash = *blake3::hash(&oversized_state).as_bytes();
    let dict = Dictionary {
        predictor_id: PredictorId::Order0,
        state: oversized_state,
        hash,
    };

    let mut buf = Vec::new();
    let result = dict.write_to(&mut buf);
    assert!(
        result.is_err(),
        "saving dictionary with state > 64 MiB should be rejected"
    );
}

// ── Dictionary train + compress + decompress roundtrip ──────────────────────

#[test]
fn dictionary_compress_decompress_roundtrip() {
    let dir = fixture_dir();
    let files = sample_files();

    // Train a dictionary on the sample files
    let mut train_pred = Order0Model::new();
    let dict = Dictionary::train(&mut train_pred, &files).unwrap();

    // Compress with dictionary
    let compressor = Compressor::new(|| Box::new(Order0Model::new())).with_dictionary(dict.clone());

    let mut archive = Cursor::new(Vec::new());
    compressor
        .compress_to_archive(&dir, &files, &mut archive)
        .expect("compression with dictionary should succeed");

    let archive_bytes = archive.into_inner();
    assert!(!archive_bytes.is_empty());

    // Decompress with the same dictionary
    let decompressor = Decompressor::new(|| Box::new(Order0Model::new())).with_dictionary(dict);

    let tmp = tempfile::tempdir().unwrap();
    let mut cursor = Cursor::new(&archive_bytes[..]);
    decompressor
        .extract_all(&mut cursor, tmp.path())
        .expect("extraction with dictionary should succeed");

    // Verify each file matches the original
    for file_path in &files {
        let rel = file_path
            .strip_prefix(&dir)
            .unwrap()
            .to_string_lossy()
            .replace('\\', "/");
        let original = std::fs::read(file_path).expect("read original");
        let extracted = std::fs::read(tmp.path().join(&rel))
            .unwrap_or_else(|e| panic!("read extracted {rel}: {e}"));
        assert_eq!(
            original,
            extracted,
            "mismatch for {rel} ({} vs {} bytes)",
            original.len(),
            extracted.len(),
        );
    }
}

#[test]
fn dictionary_compressed_without_dict_fails_on_decompress() {
    let dir = fixture_dir();
    let files = sample_files();

    // Train and compress with dictionary
    let mut train_pred = Order0Model::new();
    let dict = Dictionary::train(&mut train_pred, &files).unwrap();

    let compressor = Compressor::new(|| Box::new(Order0Model::new())).with_dictionary(dict);

    let mut archive = Cursor::new(Vec::new());
    compressor
        .compress_to_archive(&dir, &files, &mut archive)
        .expect("compression should succeed");

    let archive_bytes = archive.into_inner();

    // Try to decompress WITHOUT the dictionary — should fail with a mismatch error
    let decompressor = Decompressor::new(|| Box::new(Order0Model::new()));
    let tmp = tempfile::tempdir().unwrap();
    let mut cursor = Cursor::new(&archive_bytes[..]);
    let result = decompressor.extract_all(&mut cursor, tmp.path());
    assert!(
        result.is_err(),
        "decompressing a dictionary archive without providing the dictionary should fail"
    );
}

// ── Corrupted dictionary hash on disk ───────────────────────────────────────

#[test]
fn corrupted_dictionary_file_rejected_on_load() {
    let files = sample_files();
    let mut predictor = Order0Model::new();
    let dict = Dictionary::train(&mut predictor, &files).unwrap();

    let tmp = tempfile::tempdir().unwrap();
    let dict_path = tmp.path().join("corrupt.aed");
    dict.save(&dict_path).unwrap();

    // Corrupt a byte in the middle of the file (in the state region)
    let mut bytes = std::fs::read(&dict_path).unwrap();
    let mid = bytes.len() / 2;
    bytes[mid] ^= 0xFF;
    std::fs::write(&dict_path, &bytes).unwrap();

    let result = Dictionary::load(&dict_path);
    assert!(
        result.is_err(),
        "loading a corrupted dictionary should fail hash validation"
    );
}
