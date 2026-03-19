#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    for expected in [0, 1, 64, 256, 4096] {
        let _ = aether_core::coding::bwt_preprocess::rle_decode(data, expected);
    }
});
