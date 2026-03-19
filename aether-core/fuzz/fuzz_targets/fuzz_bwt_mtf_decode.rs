#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() < 5 {
        return;
    }
    // The first 4 bytes are the primary index; the rest is MTF data.
    // Cap expected_size to 1 MiB to prevent the fuzzer from spending time
    // on legitimate large allocations.
    let raw_expected = data.len() - 4;
    let expected_size = raw_expected.min(1 << 20);
    let _ = aether_core::coding::bwt_preprocess::bwt_mtf_decode(data, expected_size);
});
