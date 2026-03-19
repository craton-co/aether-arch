//! Fuzz target for the custom range coder (encode + decode roundtrip).
//!
//! Verifies that `encode_block` → `decode_block` always roundtrips correctly
//! for arbitrary input data, and that `decode_block` with arbitrary bytes
//! doesn't panic or corrupt memory.

#![no_main]

use libfuzzer_sys::fuzz_target;
use aether_core::coding::rans;
use aether_core::entropy::NeuralSsmPredictor;

fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }

    // Cap input size to keep fuzzing practical while covering larger inputs
    let input = if data.len() > 65536 { &data[..65536] } else { data };

    // Test 1: encode → decode roundtrip (should always succeed and match)
    let mut enc_pred = NeuralSsmPredictor::new();
    if let Ok(encoded) = rans::encode_block(input, &mut enc_pred) {
        let mut dec_pred = NeuralSsmPredictor::new();
        match rans::decode_block(&encoded, input.len(), &mut dec_pred) {
            Ok(decoded) => {
                assert_eq!(
                    decoded, input,
                    "Range coder roundtrip mismatch: {} bytes",
                    input.len()
                );
            }
            Err(_) => {
                // Encode succeeded but decode failed — this would be a bug
                panic!(
                    "Range coder encode succeeded but decode failed for {} bytes",
                    input.len()
                );
            }
        }
    }

    // Test 2: decode with arbitrary bytes (should not panic)
    let mut dec_pred = NeuralSsmPredictor::new();
    let decode_len = (input.len() % 256).max(1);
    let _ = rans::decode_block(input, decode_len, &mut dec_pred);
});
