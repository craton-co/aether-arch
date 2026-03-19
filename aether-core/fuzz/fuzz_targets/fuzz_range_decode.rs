#![no_main]
use libfuzzer_sys::fuzz_target;
use aether_core::coding::rans;

fuzz_target!(|data: &[u8]| {
    // Fuzz the range decoder with an Order0 predictor.
    // The decoder must never panic regardless of input.
    for expected in [1, 16, 64, 256] {
        let mut pred = aether_core::entropy::order0::Order0Model::new();
        let _ = rans::decode_block(data, expected, &mut pred);
    }
});
