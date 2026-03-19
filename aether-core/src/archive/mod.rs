//! Archive assembly and reading.
//!
//! This module provides convenience re-exports of the compression and
//! decompression orchestration types. The actual logic lives in
//! [`crate::pipeline::compress`] and [`crate::pipeline::decompress`].

pub use crate::pipeline::compress::{CompressionStats, Compressor};
pub use crate::pipeline::decompress::{ArchiveMetadata, Decompressor, VerificationResult};
