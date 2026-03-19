//! Fuzz target for the full decompression pipeline (seekable path).
//!
//! Feeds arbitrary bytes through the complete seekable decompression pipeline:
//! read_metadata → verify → extract_all. This exercises footer parsing,
//! block index reading, block decompression, BLAKE3 verification, and file
//! reassembly with untrusted input.
//!
//! This complements fuzz_streaming_metadata which only exercises the streaming path.

#![no_main]

use libfuzzer_sys::fuzz_target;
use std::io::Cursor;

use aether_core::entropy::{NeuralSsmPredictor, Order0Model};
use aether_core::pipeline::decompress::Decompressor;

fuzz_target!(|data: &[u8]| {
    // Try both Order0 and NeuralSsm predictors to maximize coverage
    for factory_idx in 0..2u8 {
        let decompressor = match factory_idx {
            0 => Decompressor::new(|| Box::new(Order0Model::new())),
            _ => Decompressor::new(|| Box::new(NeuralSsmPredictor::new())),
        };

        let mut cursor = Cursor::new(data);

        // Phase 1: try to read metadata (seekable path — reads footer first)
        let metadata = match decompressor.read_metadata(&mut cursor) {
            Ok(m) => m,
            Err(_) => continue,
        };

        // Phase 2: verify integrity
        cursor.set_position(0);
        let _ = decompressor.verify(&mut cursor);

        // Phase 3: try list_files
        cursor.set_position(0);
        let _ = decompressor.list_files(&mut cursor);

        // Phase 4: try extract_all to a unique temp directory
        if let Ok(tmpdir) = tempfile::tempdir() {
            cursor.set_position(0);
            let _ = decompressor.extract_all(&mut cursor, tmpdir.path());
            // tmpdir is automatically cleaned up on drop
        }

        // Phase 5: try extracting individual files from metadata
        // Cap iterations to avoid OOM from crafted archives with many file entries,
        // each of which creates a new predictor and re-decompresses blocks.
        cursor.set_position(0);
        for entry in metadata.file_entries.iter().take(8) {
            let mut output = Vec::new();
            let _ = decompressor.extract_file(&mut Cursor::new(data), &entry.path, &mut output);
        }
    }
});
