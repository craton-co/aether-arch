#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Try decoding with various expected sizes derived from the input.
    // The decoder must never panic, only return Err.
    for expected in [0, 1, 64, 256, 4096, 65536] {
        let _ = aether_core::coding::lz77_preprocess::lz77_decode(data, expected);
    }
    // Also try the size embedded in the header itself (if present).
    if data.len() >= 4 {
        let header_size = u32::from_le_bytes(data[..4].try_into().unwrap()) as usize;
        let capped = header_size.min(1 << 20); // cap to 1 MiB for fuzzing
        let _ = aether_core::coding::lz77_preprocess::lz77_decode(data, capped);
    }
});
