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

// ── External tool comparison benchmarks ─────────────────────────────────────

fn bench_vs_external(c: &mut Criterion) {
    use std::process::Command;

    // Concatenate all large fixture files into one blob
    let large_dir = fixture_dir().join("large");
    let mut data = Vec::new();
    for name in &["english.txt", "source.rs", "mixed.json"] {
        data.extend_from_slice(&std::fs::read(large_dir.join(name)).unwrap());
    }
    let total_bytes = data.len() as u64;

    // Write to a temp file for external tools
    let tmp_dir = tempfile::tempdir().unwrap();
    let input_path = tmp_dir.path().join("corpus.bin");
    std::fs::write(&input_path, &data).unwrap();

    let mut group = c.benchmark_group("vs_external");
    group.throughput(Throughput::Bytes(total_bytes));
    // External tools are fast; limit samples to keep total time reasonable
    group.sample_size(20);

    // ── AetherArch (compress only, in-memory) ───────────────────────
    let (base_dir, files) = large_files();
    group.bench_function("aetherarch_compress", |b| {
        b.iter(|| {
            let archive = compress_to_buf(&base_dir, &files, || Box::new(Order0Model::new()));
            black_box(archive.len());
        });
    });

    // ── Helper: benchmark an external compressor via file I/O ───────
    // Returns None if the tool is not found on PATH.
    fn bench_external_tool(
        group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
        name: &str,
        program: &str,
        args: &[&str],
        input: &std::path::Path,
        output_ext: &str,
    ) {
        let output_path = input.with_extension(output_ext);
        // Quick check: is the tool available?
        if Command::new(program).arg("--version").output().is_err() {
            eprintln!("  [{}] not found on PATH, skipping", name);
            return;
        }
        group.bench_function(name, |b| {
            b.iter(|| {
                let mut cmd = Command::new(program);
                for a in args {
                    cmd.arg(a);
                }
                let status = cmd.status().expect("failed to run external tool");
                assert!(status.success(), "{} failed", name);
                let out_size = std::fs::metadata(&output_path)
                    .map(|m| m.len())
                    .unwrap_or(0);
                black_box(out_size);
            });
        });
        // Clean up
        let _ = std::fs::remove_file(&output_path);
    }

    let inp = &input_path;

    // gzip -9 (file → file.gz)
    bench_external_tool(
        &mut group,
        "gzip_9",
        "gzip",
        &["-9", "-k", "-f", inp.to_str().unwrap()],
        inp,
        "bin.gz",
    );

    // bzip2 -9 (file → file.bz2)
    bench_external_tool(
        &mut group,
        "bzip2_9",
        "bzip2",
        &["-9", "-k", "-f", inp.to_str().unwrap()],
        inp,
        "bin.bz2",
    );

    // xz -9 (file → file.xz)
    bench_external_tool(
        &mut group,
        "xz_9",
        "xz",
        &["-9", "-k", "-f", inp.to_str().unwrap()],
        inp,
        "bin.xz",
    );

    // zstd -19
    {
        let zst_out = tmp_dir.path().join("corpus.zst");
        let zst_out_str = zst_out.to_str().unwrap().to_string();
        let inp_str = inp.to_str().unwrap().to_string();
        if Command::new("zstd").arg("--version").output().is_ok() {
            group.bench_function("zstd_19", |b| {
                b.iter(|| {
                    let status = Command::new("zstd")
                        .args(["-19", "-f", &inp_str, "-o", &zst_out_str, "--no-progress"])
                        .status()
                        .expect("zstd failed");
                    assert!(status.success());
                    let out_size = std::fs::metadata(&zst_out).map(|m| m.len()).unwrap_or(0);
                    black_box(out_size);
                });
            });
            let _ = std::fs::remove_file(&zst_out);
        }
    }

    // brotli -q 11
    {
        let br_out = tmp_dir.path().join("corpus.br");
        let br_out_str = br_out.to_str().unwrap().to_string();
        let inp_str = inp.to_str().unwrap().to_string();
        if Command::new("brotli").arg("--version").output().is_ok() {
            group.bench_function("brotli_11", |b| {
                b.iter(|| {
                    let status = Command::new("brotli")
                        .args(["-q", "11", "-f", &inp_str, "-o", &br_out_str])
                        .status()
                        .expect("brotli failed");
                    assert!(status.success());
                    let out_size = std::fs::metadata(&br_out).map(|m| m.len()).unwrap_or(0);
                    black_box(out_size);
                });
            });
            let _ = std::fs::remove_file(&br_out);
        }
    }

    // lz4 -9
    {
        let lz4_out = tmp_dir.path().join("corpus.lz4");
        let lz4_out_str = lz4_out.to_str().unwrap().to_string();
        let inp_str = inp.to_str().unwrap().to_string();
        if Command::new("lz4").arg("--version").output().is_ok() {
            group.bench_function("lz4_9", |b| {
                b.iter(|| {
                    let status = Command::new("lz4")
                        .args(["-9", "-f", &inp_str, &lz4_out_str])
                        .stderr(std::process::Stdio::null())
                        .status()
                        .expect("lz4 failed");
                    assert!(status.success());
                    let out_size = std::fs::metadata(&lz4_out).map(|m| m.len()).unwrap_or(0);
                    black_box(out_size);
                });
            });
            let _ = std::fs::remove_file(&lz4_out);
        }
    }

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
    bench_vs_external,
    bench_predictors,
);
criterion_main!(benches);
