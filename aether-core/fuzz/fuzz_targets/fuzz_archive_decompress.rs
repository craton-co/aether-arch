#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Fuzz the full decompression pipeline with crafted archive data.
    // The decompressor must never panic or OOM on any input.
    let decompressor = aether_core::pipeline::decompress::Decompressor::new(|| {
        Box::new(aether_core::entropy::order0::Order0Model::new())
    });

    // Test metadata parsing (seekable path)
    let mut cursor = std::io::Cursor::new(data);
    let _ = decompressor.read_metadata(&mut cursor);

    // Test metadata parsing (streaming path)
    let _ = aether_core::pipeline::decompress::Decompressor::read_metadata_streaming(
        &mut &data[..],
    );

    // Test verify (seekable path)
    let mut cursor = std::io::Cursor::new(data);
    let _ = decompressor.verify(&mut cursor);

    // Test full extraction (streaming path) to a temp dir.
    // Use a cap on input size to keep fuzzing fast.
    if data.len() <= 64 * 1024 {
        if let Ok(tmp) = tempfile::tempdir() {
            let _ = decompressor.extract_all_streaming(&mut &data[..], tmp.path());
        }
    }
});
