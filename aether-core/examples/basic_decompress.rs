//! Basic decompression example: extract and verify an .aet archive.
//!
//! Usage: cargo run --example basic_decompress -p aether-core

use std::io::Cursor;
use std::path::PathBuf;

use aether_core::entropy::NeuralSsmPredictor;
use aether_core::pipeline::compress::Compressor;
use aether_core::pipeline::decompress::Decompressor;

fn main() {
    // First, create an archive to decompress
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

    // Verify the archive integrity (checks BLAKE3 hashes per block)
    let decompressor = Decompressor::new(|| Box::new(NeuralSsmPredictor::new()));
    let mut reader = Cursor::new(&archive);
    let result = decompressor.verify(&mut reader).expect("verify failed");
    println!(
        "Verification: {} blocks OK, {} corrupted",
        result.verified_blocks,
        result.corrupted_blocks.len()
    );

    // Extract all files to a temporary directory
    let tmp = tempfile::tempdir().expect("failed to create temp dir");
    let mut reader = Cursor::new(&archive);
    decompressor
        .extract_all(&mut reader, tmp.path())
        .expect("extraction failed");

    // List extracted files
    println!("Extracted to: {}", tmp.path().display());
    for file in &files {
        let name = file.file_name().unwrap().to_str().unwrap();
        let extracted = tmp.path().join(name);
        let original = std::fs::read(file).unwrap();
        let result = std::fs::read(&extracted).unwrap();
        let ok = if original == result { "OK" } else { "MISMATCH" };
        println!("  {} ({} bytes) [{}]", name, result.len(), ok);
    }
}
