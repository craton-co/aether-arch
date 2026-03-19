//! Basic compression example: compress files into an .aet archive.
//!
//! Usage: cargo run --example basic_compress -p aether-core

use std::io::Cursor;
use std::path::PathBuf;

use aether_core::entropy::NeuralSsmPredictor;
use aether_core::pipeline::compress::Compressor;

fn main() {
    // Locate test fixtures (relative to workspace root)
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

    // Create compressor with NeuralSSM predictor (best compression)
    let compressor = Compressor::new(|| Box::new(NeuralSsmPredictor::new()));

    // Compress to an in-memory buffer (could also use a File)
    let mut output = Cursor::new(Vec::new());
    let (stats, analytics) = compressor
        .compress_to_archive(&fixture_dir, &files, &mut output)
        .expect("compression failed");

    let archive = output.into_inner();

    println!("Compressed {} files:", files.len());
    println!("  Original:   {} bytes", stats.original_size);
    println!("  Compressed: {} bytes", stats.compressed_size);
    println!("  Ratio:      {:.2}%", stats.ratio() * 100.0);
    println!("  Bits/byte:  {:.3}", stats.bits_per_byte());
    println!("  Archive:    {} bytes (in memory)", archive.len());
    println!(
        "  Timing:     {:.2?} compress, {:.2?} write",
        analytics.compression_time, analytics.write_time
    );
}
