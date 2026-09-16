//! Parallel decompression must be indistinguishable from sequential.
//!
//! The seekable path decodes one rayon task per block, relying on blocks
//! being independently decodable. These tests pin the observable consequence:
//! whatever the thread count, extraction produces the same bytes.

#![cfg(feature = "threading")]

use std::io::Cursor;

use aether_core::entropy::{NeuralSsmPredictor, Order0Model, ProbabilityPredictor};
use aether_core::pipeline::compress::Compressor;
use aether_core::pipeline::decompress::Decompressor;

/// Files chosen to produce many blocks of very uneven size — the case where a
/// fixed partition of blocks across workers would leave lanes idle, and the
/// case most likely to expose an ordering assumption.
fn corpus() -> Vec<(String, Vec<u8>)> {
    let mut files = Vec::new();

    // Compressible text of varying length.
    for i in 0..12 {
        let mut text = String::new();
        let words = [
            "archive",
            "predictor",
            "entropy",
            "block",
            "chunk",
            "decode",
            "stream",
        ];
        for j in 0..(200 * (i + 1)) {
            text.push_str(words[(i * 7 + j) % words.len()]);
            text.push(if j % 11 == 0 { '\n' } else { ' ' });
        }
        files.push((format!("text_{i:02}.txt"), text.into_bytes()));
    }

    // Incompressible blobs (route to Store/Zstd).
    for i in 0..4 {
        let mut blob = Vec::new();
        let mut x: u64 = 0xA5A5_0000 + i as u64;
        while blob.len() < 40_000 * (i + 1) {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            blob.extend_from_slice(&x.to_le_bytes());
        }
        files.push((format!("blob_{i}.bin"), blob));
    }

    // Float arrays (route through the byte-plane / BWT cascade).
    for i in 0..3 {
        let mut data = Vec::new();
        let mut x: u32 = 0x9E37_79B9 ^ (i as u32);
        let mut t = 0f32;
        while data.len() < 30_000 * (i + 1) {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            t += 0.01;
            data.extend_from_slice(&(t + (x >> 27) as f32).to_le_bytes());
        }
        files.push((format!("floats_{i}.bin"), data));
    }

    files
}

fn build_archive(
    files: &[(String, Vec<u8>)],
    factory: fn() -> Box<dyn ProbabilityPredictor>,
) -> (tempfile::TempDir, Vec<u8>) {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut paths = Vec::new();
    for (name, data) in files {
        let path = dir.path().join(name);
        std::fs::write(&path, data).expect("write input");
        paths.push(path);
    }
    let compressor = Compressor::new(factory);
    let mut buf = Cursor::new(Vec::new());
    compressor
        .compress_to_archive(dir.path(), &paths, &mut buf)
        .expect("compress_to_archive");
    (dir, buf.into_inner())
}

fn extract_with_threads(
    archive: &[u8],
    threads: usize,
    factory: fn() -> Box<dyn ProbabilityPredictor>,
) -> Vec<(String, Vec<u8>)> {
    let out = tempfile::tempdir().expect("tempdir");
    let decompressor = Decompressor::new(factory).with_max_threads(threads);
    let mut cursor = Cursor::new(archive);
    decompressor
        .extract_all(&mut cursor, out.path())
        .unwrap_or_else(|e| panic!("extract_all failed with {threads} thread(s): {e}"));

    let mut extracted: Vec<(String, Vec<u8>)> = std::fs::read_dir(out.path())
        .expect("read_dir")
        .map(|entry| {
            let entry = entry.expect("dir entry");
            (
                entry.file_name().to_string_lossy().into_owned(),
                std::fs::read(entry.path()).expect("read extracted"),
            )
        })
        .collect();
    extracted.sort_by(|a, b| a.0.cmp(&b.0));
    extracted
}

fn assert_thread_counts_agree(factory: fn() -> Box<dyn ProbabilityPredictor>) {
    let mut files = corpus();
    files.sort_by(|a, b| a.0.cmp(&b.0));
    let (_keep, archive) = build_archive(&files, factory);

    // Sanity: the archive must actually have enough blocks for the thread
    // counts below to mean anything.
    let block_count = {
        let decompressor = Decompressor::new(factory);
        let mut cursor = Cursor::new(&archive[..]);
        decompressor
            .read_metadata(&mut cursor)
            .expect("read_metadata")
            .block_index
            .len()
    };
    assert!(
        block_count >= files.len(),
        "expected at least one block per file, got {block_count}",
    );

    let reference = extract_with_threads(&archive, 1, factory);
    assert_eq!(reference, files, "sequential extraction lost data");

    for threads in [2usize, 3, 4, 8, 0] {
        let parallel = extract_with_threads(&archive, threads, factory);
        assert_eq!(
            parallel, reference,
            "extraction with {threads} thread(s) differs from sequential",
        );
    }
}

#[test]
fn parallel_and_sequential_extraction_match_ssm() {
    assert_thread_counts_agree(|| Box::new(NeuralSsmPredictor::new()));
}

#[test]
fn parallel_and_sequential_extraction_match_order0() {
    assert_thread_counts_agree(|| Box::new(Order0Model::new()));
}

#[test]
fn verify_reports_ok_under_every_thread_count() {
    let files = corpus();
    let (_keep, archive) = build_archive(&files, || Box::new(NeuralSsmPredictor::new()));

    for threads in [1usize, 4, 0] {
        let decompressor =
            Decompressor::new(|| Box::new(NeuralSsmPredictor::new())).with_max_threads(threads);
        let mut cursor = Cursor::new(&archive[..]);
        let result = decompressor.verify(&mut cursor).expect("verify");
        assert!(
            result.is_ok(),
            "verify reported corruption with {threads} thread(s): {:?}",
            result.corrupted_blocks,
        );
        assert_eq!(result.verified_blocks, result.total_blocks);
    }
}

/// Verification must report corruption, not fail, and must report the *same*
/// corruption whatever the thread count.
///
/// Extraction stops at the first bad block; verification must not — finding
/// bad blocks is its job, and one corrupt block must not hide the state of
/// everything after it. The parallel path reads leniently for exactly this
/// reason, so it needs its own coverage.
#[test]
fn verify_reports_the_same_corruption_at_every_thread_count() {
    let files = corpus();
    let (_keep, archive) = build_archive(&files, || Box::new(NeuralSsmPredictor::new()));

    // Find the block payloads by their index entries, then flip bits inside
    // two of them. Corrupting the payload (rather than a header) keeps the
    // archive structurally walkable, so verification should reach every
    // block and report precisely the two that were damaged.
    let metadata = {
        let decompressor = Decompressor::new(|| Box::new(NeuralSsmPredictor::new()));
        let mut cursor = Cursor::new(&archive[..]);
        decompressor
            .read_metadata(&mut cursor)
            .expect("read_metadata")
    };
    assert!(
        metadata.block_index.len() >= 4,
        "need several blocks to make this test meaningful",
    );

    let mut corrupted = archive.clone();
    let targets = [1usize, metadata.block_index.len() - 1];
    for &i in &targets {
        let entry = &metadata.block_index[i];
        // Land inside the compressed payload: past the block header, and
        // short of the trailer.
        let offset = entry.archive_offset as usize + 40;
        assert!(offset < corrupted.len());
        corrupted[offset] ^= 0xFF;
        corrupted[offset + 1] ^= 0x0F;
    }

    let mut reports = Vec::new();
    for threads in [1usize, 2, 4, 0] {
        let decompressor =
            Decompressor::new(|| Box::new(NeuralSsmPredictor::new())).with_max_threads(threads);
        let mut cursor = Cursor::new(&corrupted[..]);
        let result = decompressor.verify(&mut cursor).unwrap_or_else(|e| {
            panic!("verify errored with {threads} thread(s) instead of reporting: {e}")
        });

        assert!(
            !result.is_ok(),
            "verify missed the corruption with {threads} thread(s)",
        );
        assert_eq!(result.total_blocks, metadata.block_index.len());

        let mut corrupt = result.corrupted_blocks.clone();
        corrupt.sort_unstable();
        reports.push((threads, corrupt, result.verified_blocks));
    }

    let (_, reference_corrupt, reference_verified) = reports[0].clone();
    for (threads, corrupt, verified) in &reports[1..] {
        assert_eq!(
            corrupt, &reference_corrupt,
            "verify with {threads} thread(s) reported different corrupt blocks",
        );
        assert_eq!(
            verified, &reference_verified,
            "verify with {threads} thread(s) counted a different number of good blocks",
        );
    }
}
