#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // The decompressor must never panic or OOM on crafted input.
    // Cap expected_size to prevent the fuzzer from spending time on
    // legitimate large allocations.
    for expected in [0, 1, 64, 256, 4096] {
        let _ = aether_core::coding::zstd_fallback::decompress(data, expected);
    }
});
