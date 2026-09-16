//! Criterion benchmarks for the AetherArch **decompression** path.
//!
//! `compression.rs` covers the encode side and a coarse roundtrip; this file
//! isolates the stages a reader actually pays for, so a change can be
//! attributed to a stage instead of being lost in end-to-end noise:
//!
//! - `decode/*` — the range-decoder hot loop per predictor: the stage that
//!   dominates every predictor-backed block
//! - `method/*` — `router::decompress_chunk` per compression method, on
//!   payloads produced by the real encoder
//! - `transform/*` — the inverse transforms (BWT+MTF, RLE, LZ77, byte-plane)
//! - `archive/*` — metadata parsing, whole-archive extraction, verify
//!
//! Run:  cargo bench -p aether-core --bench decompression
//! A/B:  cargo bench -p aether-core --bench decompression -- --save-baseline before
//!       cargo bench -p aether-core --bench decompression -- --baseline before

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use std::io::Cursor;
use std::path::PathBuf;

use aether_core::chunker::{self, ChunkRef};
use aether_core::coding::{bwt_preprocess, lz77_preprocess, rans};
use aether_core::entropy::{NeuralSsmPredictor, Order0Model, ProbabilityPredictor};
use aether_core::format::{CompressionMethod, ContentType};
use aether_core::pipeline::compress::{CompressionProfile, Compressor};
use aether_core::pipeline::decompress::Decompressor;
use aether_core::pipeline::router;

// ── Fixtures ────────────────────────────────────────────────────────────────

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("tests")
        .join("fixtures")
}

fn english() -> Vec<u8> {
    std::fs::read(fixture_dir().join("large").join("english.txt")).expect("english.txt fixture")
}

/// A slice small enough to keep per-iteration cost sane while still being
/// large enough for the predictors to leave warm-up behind.
fn text_sample(len: usize) -> Vec<u8> {
    let data = english();
    data[..len.min(data.len())].to_vec()
}

/// Deterministic high-entropy bytes — routes to `Store`.
fn random_bytes(len: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(len);
    let mut x: u64 = 0x0123_4567_89AB_CDEF;
    while out.len() < len {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        out.extend_from_slice(&x.to_le_bytes());
    }
    out.truncate(len);
    out
}

/// Deterministic float32 array — routes to the byte-plane path.
fn float_bytes(len: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(len);
    let mut x: u32 = 0x9E37_79B9;
    let mut i = 0f32;
    while out.len() < len {
        x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        i += 0.001;
        let v = i + (x >> 26) as f32;
        out.extend_from_slice(&v.to_le_bytes());
    }
    out.truncate(len);
    out
}

/// Deterministic x86-shaped bytes — routes through BCJ+Zstd.
fn code_bytes(len: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(len);
    let mut x: u32 = 0xDEAD_BEEF;
    while out.len() < len {
        x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        match x % 8 {
            0 => {
                out.push(0xE8);
                out.extend_from_slice(&(x ^ 0x0BAD_F00D).to_le_bytes());
            }
            1 => out.extend_from_slice(&[0x55, 0x48, 0x89, 0xE5]),
            2 => out.extend_from_slice(&[0x48, 0x83, 0xEC, 0x20]),
            3 => out.extend_from_slice(&x.to_le_bytes()),
            _ => out.extend_from_slice(&[0x90, 0x90, 0x0F, 0x1F, 0x44, 0x00, 0x00]),
        }
    }
    out.truncate(len);
    out
}

fn chunk_of(data: &[u8]) -> ChunkRef<'_> {
    chunker::chunk_fixed_refs(data, data.len())
        .into_iter()
        .next()
        .expect("non-empty chunk")
}

/// Run the real encoder so the benchmarked payload is exactly what an
/// archive would contain.
fn encode(
    data: &[u8],
    content_type: ContentType,
    profile: CompressionProfile,
) -> (CompressionMethod, Vec<u8>, bool) {
    let chunk = chunk_of(data);
    let mut predictor: Box<dyn ProbabilityPredictor> = Box::new(NeuralSsmPredictor::new());
    let compressed = router::compress_chunk(&chunk, predictor.as_mut(), content_type, profile)
        .expect("compress_chunk");
    (
        compressed.method,
        compressed.data,
        compressed.predictor_synced,
    )
}

// ── decode/* — the range-decoder hot loop ───────────────────────────────────

/// The stream a `BwtPredictorRans` block actually range-codes: BWT+MTF+RLE
/// of English text. This is where a NeuralSSM archive spends its decode time.
fn bwt_rle_stream(text: &[u8]) -> Vec<u8> {
    let (_, mtf) = bwt_preprocess::bwt_mtf_encode_parts(text).expect("bwt encode");
    bwt_preprocess::rle_encode(&mtf).unwrap_or(mtf)
}

fn bench_decode_loop(c: &mut Criterion) {
    let text = text_sample(512 * 1024);
    let stream = bwt_rle_stream(&text);

    let mut group = c.benchmark_group("decode");
    group.throughput(Throughput::Bytes(stream.len() as u64));
    group.sample_size(10);

    let mut ssm = NeuralSsmPredictor::new();
    let ssm_encoded = rans::encode_block(&stream, &mut ssm).expect("ssm encode");
    group.bench_function("ssm_bwt_stream", |b| {
        b.iter(|| {
            let mut predictor = NeuralSsmPredictor::new();
            let out = rans::decode_block(&ssm_encoded, stream.len(), &mut predictor).unwrap();
            black_box(out.len());
        });
    });

    let mut o0 = Order0Model::new();
    let o0_encoded = rans::encode_block(&stream, &mut o0).expect("order0 encode");
    group.bench_function("order0_bwt_stream", |b| {
        b.iter(|| {
            let mut predictor = Order0Model::new();
            let out = rans::decode_block(&o0_encoded, stream.len(), &mut predictor).unwrap();
            black_box(out.len());
        });
    });

    group.finish();
}

// ── method/* — router::decompress_chunk per compression method ──────────────

struct MethodCase {
    name: String,
    method: CompressionMethod,
    payload: Vec<u8>,
    original_len: usize,
    predictor_synced: bool,
}

/// One case per method the router actually produces for a representative
/// input. The method is taken from the encoder rather than asserted, so a
/// routing change shows up as a renamed case instead of a failed bench.
fn method_cases() -> Vec<MethodCase> {
    let inputs: Vec<(&str, Vec<u8>, ContentType, CompressionProfile)> = vec![
        (
            "random",
            random_bytes(256 * 1024),
            ContentType::BinaryRandom,
            CompressionProfile::Archival,
        ),
        (
            "text_fast",
            text_sample(256 * 1024),
            ContentType::Text,
            CompressionProfile::Fast,
        ),
        (
            "code_fast",
            code_bytes(256 * 1024),
            ContentType::Executable,
            CompressionProfile::Fast,
        ),
        (
            "text_archival",
            text_sample(256 * 1024),
            ContentType::Text,
            CompressionProfile::Archival,
        ),
        (
            "floats",
            float_bytes(256 * 1024),
            ContentType::NumericData,
            CompressionProfile::Archival,
        ),
    ];

    inputs
        .into_iter()
        .map(|(label, data, content_type, profile)| {
            let (method, payload, predictor_synced) = encode(&data, content_type, profile);
            MethodCase {
                name: format!("{}_{:?}", label, method),
                method,
                payload,
                original_len: data.len(),
                predictor_synced,
            }
        })
        .collect()
}

fn bench_methods(c: &mut Criterion) {
    let cases = method_cases();

    let mut group = c.benchmark_group("method");
    group.sample_size(10);

    for case in &cases {
        group.throughput(Throughput::Bytes(case.original_len as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(case.name.clone()),
            case,
            |b, case| {
                b.iter(|| {
                    let mut predictor: Box<dyn ProbabilityPredictor> =
                        Box::new(NeuralSsmPredictor::new());
                    let out = router::decompress_chunk(
                        &case.payload,
                        case.method,
                        case.original_len,
                        predictor.as_mut(),
                        case.predictor_synced,
                    )
                    .unwrap();
                    black_box(out.len());
                });
            },
        );
    }

    group.finish();
}

// ── transform/* — inverse transforms in isolation ───────────────────────────

fn bench_transforms(c: &mut Criterion) {
    let text = text_sample(512 * 1024);

    let mut group = c.benchmark_group("transform");
    group.sample_size(20);

    // Inverse BWT + MTF, at sizes straddling the point where the LF table
    // stops fitting in cache. The walk changes shape there — a single size
    // would measure only half of it.
    for &size in &[128 * 1024usize, 512 * 1024, 2 * 1024 * 1024] {
        let sample = text_sample(size);
        if sample.len() < size {
            continue; // fixture too small for this case
        }
        let (primary_index, mtf) =
            bwt_preprocess::bwt_mtf_encode_parts(&sample).expect("bwt encode");
        group.throughput(Throughput::Bytes(sample.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("bwt_mtf_decode", format!("{}KiB", size / 1024)),
            &(primary_index, mtf, sample.len()),
            |b, (primary_index, mtf, len)| {
                b.iter(|| {
                    let out =
                        bwt_preprocess::bwt_mtf_decode_parts(*primary_index, mtf, *len).unwrap();
                    black_box(out.len());
                });
            },
        );
    }

    let (primary_index, mtf) = bwt_preprocess::bwt_mtf_encode_parts(&text).expect("bwt encode");
    let _ = primary_index;

    // RLE decode over the same MTF stream
    if let Some(rle) = bwt_preprocess::rle_encode(&mtf) {
        group.throughput(Throughput::Bytes(mtf.len() as u64));
        group.bench_function("rle_decode", |b| {
            b.iter(|| {
                let out = bwt_preprocess::rle_decode(&rle, mtf.len()).unwrap();
                black_box(out.len());
            });
        });
    }

    // LZ77 decode
    if let Some(lz) = lz77_preprocess::lz77_encode(&text) {
        group.throughput(Throughput::Bytes(text.len() as u64));
        group.bench_function("lz77_decode", |b| {
            b.iter(|| {
                let out = lz77_preprocess::lz77_decode(&lz, text.len()).unwrap();
                black_box(out.len());
            });
        });
    }

    // Byte-plane decode. Built directly rather than through the router: on
    // this input BWT often wins the routing cascade, but the byte-plane
    // decoder still has to be fast for the archives where it does win.
    let floats = float_bytes(512 * 1024);
    let payload = aether_core::coding::byteplane_preprocess::byteplane_encode(
        &floats,
        aether_core::coding::byteplane_preprocess::BytePlaneWidth::Four,
    )
    .expect("byteplane encode");
    group.throughput(Throughput::Bytes(floats.len() as u64));
    group.bench_function("byteplane_decode", |b| {
        b.iter(|| {
            let out =
                aether_core::coding::byteplane_preprocess::byteplane_decode(&payload, floats.len())
                    .unwrap();
            black_box(out.len());
        });
    });

    group.finish();
}

// ── archive/* — whole-archive paths ─────────────────────────────────────────

fn build_archive(files: &[(&str, Vec<u8>)]) -> (tempfile::TempDir, Vec<u8>, u64) {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut paths = Vec::new();
    let mut total = 0u64;
    for (name, data) in files {
        let path = dir.path().join(name);
        std::fs::write(&path, data).expect("write fixture");
        total += data.len() as u64;
        paths.push(path);
    }
    let compressor = Compressor::new(|| Box::new(NeuralSsmPredictor::new()));
    let mut buf = Cursor::new(Vec::new());
    compressor
        .compress_to_archive(dir.path(), &paths, &mut buf)
        .expect("compress_to_archive");
    (dir, buf.into_inner(), total)
}

fn bench_archive(c: &mut Criterion) {
    // Deliberately small: `archive/*` measures the whole pipeline including
    // metadata and file writes, so it must stay inside a bench budget.
    let text_files = vec![
        ("english.txt", text_sample(192 * 1024)),
        ("floats.bin", float_bytes(64 * 1024)),
        ("random.bin", random_bytes(64 * 1024)),
    ];
    let (_keep, archive, total_bytes) = build_archive(&text_files);

    let mut group = c.benchmark_group("archive");
    group.sample_size(10);
    group.throughput(Throughput::Bytes(total_bytes));

    group.bench_function("read_metadata", |b| {
        b.iter(|| {
            let decompressor = Decompressor::new(|| Box::new(NeuralSsmPredictor::new()));
            let mut cursor = Cursor::new(&archive[..]);
            let meta = decompressor.read_metadata(&mut cursor).unwrap();
            black_box(meta.block_index.len());
        });
    });

    group.bench_function("extract_all", |b| {
        b.iter(|| {
            let decompressor = Decompressor::new(|| Box::new(NeuralSsmPredictor::new()));
            let out = tempfile::tempdir().unwrap();
            let mut cursor = Cursor::new(&archive[..]);
            decompressor.extract_all(&mut cursor, out.path()).unwrap();
        });
    });

    group.bench_function("verify", |b| {
        b.iter(|| {
            let decompressor = Decompressor::new(|| Box::new(NeuralSsmPredictor::new()));
            let mut cursor = Cursor::new(&archive[..]);
            let result = decompressor.verify(&mut cursor).unwrap();
            black_box(result.verified_blocks);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_decode_loop,
    bench_methods,
    bench_transforms,
    bench_archive
);
criterion_main!(benches);
