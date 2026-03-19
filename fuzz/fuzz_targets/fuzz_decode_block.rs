//! Fuzz target for block decompression via the routing layer.
//!
//! Exercises `router::decompress_chunk` with arbitrary payloads for each
//! compression method and multiple predictor types. This is the primary
//! attack surface: an adversary who controls the archive contents can craft
//! block payloads to trigger panics, OOM, or logic errors in the
//! decompression pipeline.

#![no_main]

use libfuzzer_sys::fuzz_target;
use aether_core::entropy::{NeuralSsmPredictor, Order0Model, ProbabilityPredictor};
use aether_core::format::CompressionMethod;
use aether_core::pipeline::router;

fuzz_target!(|data: &[u8]| {
    if data.len() < 3 {
        return;
    }

    // Use first byte to select compression method, second byte for flags,
    // third byte to select predictor type.
    let method_byte = data[0] % 6;
    let predictor_synced = (data[1] & 1) != 0;
    let predictor_type = data[2] % 2;
    let payload = &data[3..];

    let method = match method_byte {
        0 => CompressionMethod::PredictorRans,
        1 => CompressionMethod::Zstd,
        2 => CompressionMethod::Store,
        3 => CompressionMethod::LzPredictorRans,
        4 => CompressionMethod::Lz77PredictorRans,
        5 => CompressionMethod::BwtPredictorRans,
        _ => unreachable!(), // method_byte = data[0] % 6, so 0..=5 is exhaustive
    };

    // Use a reasonable uncompressed size (capped to avoid OOM in fuzzer)
    // Derive from payload length to keep it bounded
    let uncompressed_size = if payload.len() > 4 {
        let raw = u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]);
        // Cap at 1 MiB to keep fuzzing fast
        (raw as usize % (1024 * 1024)).max(1)
    } else {
        payload.len().max(1)
    };

    // Test with multiple predictor types to find predictor-specific bugs
    let mut predictor: Box<dyn ProbabilityPredictor> = match predictor_type {
        0 => Box::new(Order0Model::new()),
        _ => Box::new(NeuralSsmPredictor::new()),
    };
    let _ = router::decompress_chunk(
        payload,
        method,
        uncompressed_size,
        predictor.as_mut(),
        predictor_synced,
    );
});
