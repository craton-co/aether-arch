//! Fuzz target for streaming metadata parsing.
//!
//! Feeds arbitrary bytes to `Decompressor::read_metadata_streaming` to exercise
//! the archive header, file table, and solid group table parsers with untrusted
//! input. Catches panics, excessive allocations, and parsing errors.
//!
//! The streaming path is especially important to fuzz because it reads from
//! non-seekable sources (pipes, network) where the data cannot be rewound.

#![no_main]

use libfuzzer_sys::fuzz_target;
use std::io::Cursor;

use aether_core::pipeline::decompress::Decompressor;
use aether_core::entropy::NeuralSsmPredictor;

fuzz_target!(|data: &[u8]| {
    let mut cursor = Cursor::new(data);

    // Phase 1: parse streaming metadata (Header + FileTable + GroupTable)
    let metadata = match Decompressor::read_metadata_streaming(&mut cursor) {
        Ok(m) => m,
        Err(_) => return, // Expected for most fuzz inputs
    };

    // Phase 2: if metadata parsed, try to extract/verify with the remaining bytes.
    // This exercises decompress_block_streaming with potentially corrupt payloads.
    // Use a unique temp directory to prevent symlink attacks on predictable paths.
    let decompressor = Decompressor::new(|| Box::new(NeuralSsmPredictor::new()));
    if let Ok(tmpdir) = tempfile::tempdir() {
        let _ = decompressor.extract_with_streaming_metadata(&mut cursor, &metadata, tmpdir.path());
        // tmpdir is automatically cleaned up on drop
    }
});
