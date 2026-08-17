//! Regression coverage for decode-path predictor agreement.
//!
//! Every predictor-backed block payload is range-coded by `compress_chunk`
//! with a scratch `NeuralSsmPredictor`, never with the group predictor.
//! `decompress_chunk` must build the same predictor. These tests pin that
//! contract for each routed method and for each group predictor the CLI
//! can select, because a mismatch produces archives that compress cleanly
//! and then fail to extract.

use aether_core::chunker;
use aether_core::coding::byteplane_preprocess as bp;
use aether_core::entropy::{NeuralSsmPredictor, Order0Model, ProbabilityPredictor, RlePredictor};
use aether_core::format::{CompressionMethod, ContentType};
use aether_core::pipeline::compress::CompressionProfile;
use aether_core::pipeline::router;

fn fixture(rel: &str) -> Option<Vec<u8>> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join(rel);
    std::fs::read(path).ok()
}

/// Factory for a group predictor, named as the CLI names it.
type PredictorFactory = fn() -> Box<dyn ProbabilityPredictor>;

fn predictors() -> Vec<(&'static str, PredictorFactory)> {
    vec![
        ("order0", || Box::new(Order0Model::new())),
        ("ssm", || Box::new(NeuralSsmPredictor::new())),
        ("rle", || Box::new(RlePredictor::new())),
    ]
}

/// Route every chunk of `data` through compress + decompress with each
/// group predictor, asserting an exact round-trip.
fn assert_routed_roundtrip(label: &str, data: &[u8], content_type: ContentType) {
    let chunks = chunker::chunk_data_refs(data);
    assert!(!chunks.is_empty(), "{label}: no chunks produced");

    for (name, factory) in predictors() {
        let mut methods = Vec::new();
        for (idx, chunk) in chunks.iter().enumerate() {
            let mut enc = factory();
            let compressed = router::compress_chunk(
                chunk,
                enc.as_mut(),
                content_type,
                CompressionProfile::Archival,
            )
            .unwrap_or_else(|e| panic!("{label}/{name} chunk {idx}: compress failed: {e}"));

            let mut dec = factory();
            let original = router::decompress_chunk(
                &compressed.data,
                compressed.method,
                chunk.length,
                dec.as_mut(),
                compressed.predictor_synced,
            )
            .unwrap_or_else(|e| {
                panic!(
                    "{label}/{name} chunk {idx} ({:?}): decompress failed: {e}",
                    compressed.method
                )
            });

            assert_eq!(
                original, chunk.data,
                "{label}/{name} chunk {idx} ({:?}): round-trip mismatch",
                compressed.method,
            );
            methods.push(compressed.method);
        }
        assert!(!methods.is_empty());
    }
}

#[test]
fn text_fixture_round_trips_under_every_group_predictor() {
    let Some(data) = fixture("tests/fixtures/large/english.txt") else {
        eprintln!("fixture missing, skipping");
        return;
    };
    assert_routed_roundtrip("english.txt", &data, ContentType::Text);
}

#[test]
fn numeric_fixtures_round_trip_under_every_group_predictor() {
    for rel in [
        "tests/fixtures/numeric/weights_fp32.bin",
        "tests/fixtures/numeric/weights_bf16.bin",
    ] {
        let Some(data) = fixture(rel) else {
            eprintln!("fixture {rel} missing, skipping");
            continue;
        };
        assert_routed_roundtrip(rel, &data, ContentType::NumericData);
    }
}

/// Byte-plane payloads are range-coded plane-by-plane with `Order0Model`.
/// Skewed exponent planes drive `Order0Model::predict_cdf` into its
/// overshoot fallback, which the encode-side `query_cdf` must reproduce.
#[test]
fn byteplane_payloads_round_trip_on_real_numeric_data() {
    let mut checked = 0usize;
    for rel in [
        "tests/fixtures/numeric/weights_fp32.bin",
        "tests/fixtures/numeric/weights_bf16.bin",
        "tests/fixtures/large/english.txt",
    ] {
        let Some(data) = fixture(rel) else { continue };
        // Byte-plane encoding is O(n log n)-ish per width; a few hundred KiB
        // is enough to reach the skewed-exponent distributions that matter,
        // and keeps this test inside a normal CI budget.
        let data = &data[..data.len().min(384 * 1024)];
        for chunk in chunker::chunk_fixed_refs(data, 96 * 1024) {
            for width in [bp::BytePlaneWidth::Two, bp::BytePlaneWidth::Four] {
                let Some(payload) = bp::byteplane_encode(chunk.data, width) else {
                    continue;
                };
                let decoded = bp::byteplane_decode(&payload, chunk.length).unwrap_or_else(|e| {
                    panic!(
                        "{rel} ({width:?}, {} bytes): byteplane decode failed: {e}",
                        chunk.length
                    )
                });
                assert_eq!(
                    decoded, chunk.data,
                    "{rel} ({width:?}, {} bytes): byteplane round-trip mismatch",
                    chunk.length,
                );
                checked += 1;
            }
        }
    }
    assert!(checked > 0, "no byte-plane payloads exercised");
}

/// Executables route through BCJ+Zstd and the LZ77 predictor path; both
/// must decode identically regardless of the configured group predictor.
#[test]
fn executable_like_data_round_trips_under_every_group_predictor() {
    // Synthesise x86-ish content: repeated code-shaped byte patterns with
    // embedded 32-bit little-endian call targets, plus a high-entropy tail.
    let mut data = Vec::with_capacity(1 << 20);
    let mut x: u32 = 0x9e37_79b9;
    while data.len() < (1 << 20) {
        x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        match x % 8 {
            0 => {
                data.push(0xE8); // call rel32
                data.extend_from_slice(&(x ^ 0x0BAD_F00D).to_le_bytes());
            }
            1 => data.extend_from_slice(&[0x55, 0x48, 0x89, 0xE5]),
            2 => data.extend_from_slice(&[0x48, 0x83, 0xEC, 0x20]),
            3 => data.extend_from_slice(&x.to_le_bytes()),
            _ => data.extend_from_slice(&[0x90, 0x90, 0x0F, 0x1F, 0x44, 0x00, 0x00]),
        }
    }
    assert_routed_roundtrip("synthetic-exe", &data, ContentType::Executable);
}

/// `Store` and `Zstd` blocks carry `predictor_synced = true`; make sure the
/// decode path handles them with every predictor too.
#[test]
fn incompressible_data_round_trips_under_every_group_predictor() {
    let mut data = Vec::with_capacity(512 * 1024);
    let mut x: u64 = 0xDEAD_BEEF_CAFE_1234;
    while data.len() < 512 * 1024 {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        data.extend_from_slice(&x.to_le_bytes());
    }
    assert_routed_roundtrip("random", &data, ContentType::BinaryRandom);
    let _ = CompressionMethod::Store;
}
