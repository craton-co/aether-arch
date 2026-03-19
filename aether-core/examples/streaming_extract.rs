//! Streaming decompression example: extract from a non-seekable source.
//!
//! Demonstrates the two-phase streaming API where metadata is read first
//! to auto-detect the predictor, then extraction proceeds sequentially.
//!
//! Usage: cargo run --example streaming_extract -p aether-core

use std::io::Cursor;
use std::path::PathBuf;

use aether_core::entropy::NeuralSsmPredictor;
use aether_core::pipeline::compress::Compressor;
use aether_core::pipeline::decompress::Decompressor;

fn main() {
    // Create an archive in memory first
    let fixture_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("tests")
        .join("fixtures")
        .join("sample");

    let files = vec![
        fixture_dir.join("hello.txt"),
        fixture_dir.join("code.rs"),
        fixture_dir.join("data.json"),
    ];

    let compressor = Compressor::new(|| Box::new(NeuralSsmPredictor::new()));
    let mut buf = Cursor::new(Vec::new());
    compressor
        .compress_to_archive(&fixture_dir, &files, &mut buf)
        .expect("compression failed");
    let archive = buf.into_inner();
    println!("Created archive: {} bytes", archive.len());

    // Phase 1: Read metadata from a streaming (non-seekable) source.
    // In practice this could be stdin or a network stream.
    let mut reader = Cursor::new(&archive);
    let metadata =
        Decompressor::read_metadata_streaming(&mut reader).expect("failed to read metadata");

    println!("Archive contains {} files:", metadata.header.file_count);
    for entry in &metadata.file_entries {
        println!("  {} ({} bytes)", entry.path, entry.original_size);
    }

    // Phase 2: Create decompressor with the detected predictor and extract.
    let decompressor = Decompressor::new(|| Box::new(NeuralSsmPredictor::new()));
    let tmp = tempfile::tempdir().expect("failed to create temp dir");
    decompressor
        .extract_with_streaming_metadata(&mut reader, &metadata, tmp.path())
        .expect("streaming extraction failed");

    // Verify extracted files match originals
    println!("\nExtracted to: {}", tmp.path().display());
    for file in &files {
        let name = file.file_name().unwrap().to_str().unwrap();
        let original = std::fs::read(file).unwrap();
        let extracted = std::fs::read(tmp.path().join(name)).unwrap();
        let ok = if original == extracted {
            "OK"
        } else {
            "MISMATCH"
        };
        println!("  {} ({} bytes) [{}]", name, extracted.len(), ok);
    }
}
