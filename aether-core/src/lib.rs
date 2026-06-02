//! # AetherArch Core Library
//!
//! Next-generation file archiving with neural-probabilistic prediction,
//! adaptive entropy coding, and semantic solid grouping.
//!
//! ## Architecture
//!
//! - **`format`**: Constants, enums, and the `.aet` binary format specification.
//! - **`header`**: Serialization for archive headers, file entries, solid groups, and footers.
//! - **`block`**: Block headers, trailers, and index entries.
//! - **`chunker`**: Content-defined chunking via FastCDC.
//! - **`analyzer`**: Entropy analysis and content-type detection.
//! - **`grouper`**: Semantic solid grouping by file type.
//! - **`entropy`**: Probability predictors (`Order0`, `ContextMixer`, `NeuralSsm`).
//! - **`coding`**: Custom byte-aligned range coding and Zstandard fallback.
//! - **`pipeline`**: High-level compress/decompress orchestration.
//! - **`archive`**: Archive assembly, reading, and random-access indexing.

pub mod analyzer;
pub mod archive;
pub mod block;
pub mod builtin_dicts;
pub mod chunker;
#[cfg(feature = "cloud")]
pub mod cloud;
pub mod coding;
pub mod dictionary;
pub mod entropy;
pub mod error;
pub mod format;
pub mod grouper;
pub mod header;
pub mod pipeline;

#[cfg(feature = "enterprise")]
pub mod crypto;

// Re-exports for convenience
pub use error::{AetherError, Result};
pub use format::{CompressionMethod, ContentType, PredictorId};
