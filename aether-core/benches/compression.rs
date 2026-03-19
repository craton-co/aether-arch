//! Criterion benchmarks for AetherArch compression pipeline.
//!
//! Benchmarks cover the key stages:
//! - End-to-end compress + decompress roundtrip
//! - BWT+MTF+RLE preprocessing
//! - Range coder encode/decode
//! - Individual predictor update+predict cycles
//!
//! Run:  cargo bench -p aether-core
//! With enterprise features:  cargo bench -p aether-core --features enterprise

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use std::io::Cursor;
use std::path::PathBuf;

use aether_core::entropy::{NeuralSsmPredictor, Order0Model, ProbabilityPredictor};
use aether_core::pipeline::compress::Compressor;
use aether_core::pipeline::decompress::Decompressor;

// ── Helpers ─────────────────────────────────────────────────────────────────

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("tests")
        .join("fixtures")
}

#[allow(dead_code)]
fn sample_files() -> (PathBuf, Vec<PathBuf>) {
    let dir = fixture_dir().join("sample");
    let files = vec![
        dir.join("hello.txt"),
        dir.join("code.rs"),
        dir.join("data.json"),
    ];
    (dir, files)
}

fn large_files() -> (PathBuf, Vec<PathBuf>) {
    let dir = fixture_dir().join("large");
    let files = vec![
        dir.join("english.txt"),
        dir.join("source.rs"),
        dir.join("mixed.json"),
    ];
    (dir, files)
}

fn total_file_size(files: &[PathBuf]) -> u64 {
    files
        .iter()
        .map(|f| std::fs::metadata(f).unwrap().len())
        .sum()
}

fn compress_to_buf(
    base_dir: &std::path::Path,
    files: &[PathBuf],
    factory: impl Fn() -> Box<dyn ProbabilityPredictor> + Send + Sync + 'static,
) -> Vec<u8> {
    let compressor = Compressor::new(factory);
    let mut buf = Cursor::new(Vec::new());
    compressor
        .compress_to_archive(base_dir, files, &mut buf)
        .expect("compression should succeed");
    buf.into_inner()
}

// ── Roundtrip benchmarks ────────────────────────────────────────────────────

fn bench_roundtrip(c: &mut Criterion) {
    let (base_dir, files) = large_files();
    let total_bytes = total_file_size(&files);

    let mut group = c.benchmark_group("roundtrip");
    group.throughput(Throughput::Bytes(total_bytes));

    // Order0 — fastest predictor
    group.bench_function("compress_order0", |b| {
        b.iter(|| {
            let archive = compress_to_buf(&base_dir, &files, || Box::new(Order0Model::new()));
            black_box(archive.len());
        });
    });

    // NeuralSSM — best compression
    group.bench_function("compress_ssm", |b| {
        b.iter(|| {
            let archive =
                compress_to_buf(&base_dir, &files, || Box::new(NeuralSsmPredictor::new()));
            black_box(archive.len());
        });
    });

    // Decompress (using pre-compressed archive)
    let archive_o0 = compress_to_buf(&base_dir, &files, || Box::new(Order0Model::new()));
    group.bench_function("decompress_order0", |b| {
        b.iter(|| {
            let decompressor = Decompressor::new(|| Box::new(Order0Model::new()));
            let tmp = tempfile::tempdir().unwrap();
            let mut cursor = Cursor::new(&archive_o0[..]);
            decompressor.extract_all(&mut cursor, tmp.path()).unwrap();
        });
    });

    let archive_ssm = compress_to_buf(&base_dir, &files, || Box::new(NeuralSsmPredictor::new()));
    group.bench_function("decompress_ssm", |b| {
        b.iter(|| {
            let decompressor = Decompressor::new(|| Box::new(NeuralSsmPredictor::new()));
            let tmp = tempfile::tempdir().unwrap();
            let mut cursor = Cursor::new(&archive_ssm[..]);
            decompressor.extract_all(&mut cursor, tmp.path()).unwrap();
        });
    });

    group.finish();
}

// ── BWT preprocessing benchmarks ────────────────────────────────────────────

fn bench_bwt(c: &mut Criterion) {
    let text = std::fs::read(fixture_dir().join("large").join("english.txt")).unwrap();

    let mut group = c.benchmark_group("bwt");
    group.throughput(Throughput::Bytes(text.len() as u64));

    group.bench_function("encode", |b| {
        b.iter(|| {
            let result = aether_core::coding::bwt_preprocess::bwt_mtf_encode(&text).unwrap();
            black_box(result.len());
        });
    });

    let encoded = aether_core::coding::bwt_preprocess::bwt_mtf_encode(&text).unwrap();
    group.bench_function("decode", |b| {
        b.iter(|| {
            let result =
                aether_core::coding::bwt_preprocess::bwt_mtf_decode(&encoded, text.len()).unwrap();
            black_box(result.len());
        });
    });

    group.finish();
}

// ── Range coder benchmarks ──────────────────────────────────────────────────

fn bench_range_coder(c: &mut Criterion) {
    use aether_core::coding::rans::{probs_to_cdf, RangeDecoder, RangeEncoder};

    // Generate a realistic byte stream from fixture data
    let data = std::fs::read(fixture_dir().join("large").join("english.txt")).unwrap();

    let mut group = c.benchmark_group("range_coder");
    group.throughput(Throughput::Bytes(data.len() as u64));

    // Build a CDF from actual data distribution
    let mut counts = [1u32; 256];
    for &b in &data {
        counts[b as usize] += 1;
    }
    let total: u32 = counts.iter().sum();
    let mut probs = [0f32; 256];
    for i in 0..256 {
        probs[i] = counts[i] as f32 / total as f32;
    }
    let cdf = probs_to_cdf(&probs);

    group.bench_function("encode", |b| {
        b.iter(|| {
            let mut encoder = RangeEncoder::new();
            for &byte in &data {
                encoder.encode_cdf(byte, &cdf);
            }
            let encoded = encoder.finish().unwrap();
            black_box(encoded.len());
        });
    });

    // Pre-encode for decode benchmark
    let mut encoder = RangeEncoder::new();
    for &byte in &data {
        encoder.encode_cdf(byte, &cdf);
    }
    let encoded = encoder.finish().unwrap();

    group.bench_function("decode", |b| {
        b.iter(|| {
            let mut decoder = RangeDecoder::new(&encoded);
            for _ in 0..data.len() {
                let sym = decoder.decode_cdf(&cdf);
                black_box(sym);
            }
        });
    });

    group.finish();
}

// ── Predictor benchmarks ────────────────────────────────────────────────────

fn bench_predictors(c: &mut Criterion) {
    // Use BWT-encoded data since that's what predictors see in practice
    let text = std::fs::read(fixture_dir().join("large").join("english.txt")).unwrap();
    let bwt_data = aether_core::coding::bwt_preprocess::bwt_mtf_encode(&text).unwrap();

    let mut group = c.benchmark_group("predictor_predict");
    group.throughput(Throughput::Bytes(bwt_data.len() as u64));

    // Order0
    group.bench_function("order0", |b| {
        b.iter(|| {
            let mut pred = Order0Model::new();
            for &byte in &bwt_data {
                let cdf = pred.predict_cdf();
                black_box(cdf[128]); // sample one value to prevent dead-code elimination
                pred.update(byte);
            }
        });
    });

    // NeuralSSM
    group.bench_function("neural_ssm", |b| {
        b.iter(|| {
            let mut pred = NeuralSsmPredictor::new();
            for &byte in &bwt_data {
                let cdf = pred.predict_cdf();
                black_box(cdf[128]);
                pred.update(byte);
            }
        });
    });

    group.finish();
}

// ── Registration ────────────────────────────────────────────────────────────

criterion_group!(
    benches,
    bench_roundtrip,
    bench_bwt,
    bench_range_coder,
    bench_predictors,
);
criterion_main!(benches);
