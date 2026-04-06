//! Entropy analysis, content-type detection, and compression routing.
//!
//! Analyzes chunks to determine:
//! - What type of content they contain (text, binary, image, etc.)
//! - How random/structured they are (Shannon entropy)
//! - Which compression method is optimal (predictor+range, zstd, or store)

use crate::format::ContentType;

// ── Entropy Thresholds ───────────────────────────────────────────────────────

/// Above this entropy (bits/byte), data is near-random → use zstd fallback.
pub const HIGH_ENTROPY_THRESHOLD: f64 = 7.5;

/// Above this, data is incompressible → store raw.
pub const INCOMPRESSIBLE_THRESHOLD: f64 = 7.95;

// ── Analysis Types ───────────────────────────────────────────────────────────

/// Recommended compression method for a chunk or file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecommendedMethod {
    /// Low/medium entropy: structured data → predictor + range coding.
    PredictorRans,
    /// High entropy but still somewhat compressible → zstd.
    Zstd,
    /// Incompressible (encrypted, already compressed) → store verbatim.
    Store,
}

/// Result of analyzing a chunk or file.
#[derive(Debug, Clone)]
pub struct AnalysisResult {
    /// Detected content type (text, binary, image, executable, etc.).
    pub content_type: ContentType,
    /// Mean Shannon entropy in bits per byte (0.0 = uniform, 8.0 = random).
    pub mean_entropy: f64,
    /// Recommended compression method based on entropy and content type.
    pub recommended_method: RecommendedMethod,
}

// ── Content-Type Detection ───────────────────────────────────────────────────

/// Detect content type from file extension and/or magic bytes.
pub fn detect_content_type(path: &str, first_bytes: &[u8]) -> ContentType {
    // Try magic bytes first
    if let Some(ct) = detect_from_magic(first_bytes) {
        return ct;
    }

    // Fall back to extension
    detect_from_extension(path)
}

fn detect_from_magic(data: &[u8]) -> Option<ContentType> {
    if data.len() < 4 {
        return None;
    }

    // NumPy .npy format: "\x93NUMPY"
    if data.len() >= 6 && data.starts_with(&[0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59]) {
        return Some(ContentType::NumericData);
    }
    // Safetensors: starts with '{' followed by JSON metadata (8-byte LE length prefix + '{')
    // The first 8 bytes are the metadata length, then JSON starting with '{'
    if data.len() >= 9 {
        let meta_len = u64::from_le_bytes(data[0..8].try_into().unwrap_or([0; 8]));
        if meta_len > 0 && meta_len < 100_000_000 && data[8] == b'{' {
            // Heuristic: looks like safetensors header
            return Some(ContentType::NumericData);
        }
    }
    // GGUF format: "GGUF" magic
    if data.starts_with(b"GGUF") {
        return Some(ContentType::NumericData);
    }

    // ELF binary
    if data.starts_with(b"\x7fELF") {
        return Some(ContentType::Executable);
    }
    // PE executable (MZ header)
    if data.starts_with(b"MZ") {
        return Some(ContentType::Executable);
    }
    // JPEG
    if data.starts_with(&[0xFF, 0xD8, 0xFF]) {
        return Some(ContentType::Image);
    }
    // PNG
    if data.starts_with(&[0x89, 0x50, 0x4E, 0x47]) {
        return Some(ContentType::Image);
    }
    // GIF
    if data.starts_with(b"GIF8") {
        return Some(ContentType::Image);
    }
    // WEBP (RIFF....WEBP)
    if data.len() >= 12 && data.starts_with(b"RIFF") && &data[8..12] == b"WEBP" {
        return Some(ContentType::Image);
    }
    // PDF
    if data.starts_with(b"%PDF") {
        return Some(ContentType::BinaryStructured);
    }
    // ZIP / JAR / DOCX / XLSX (already compressed)
    if data.starts_with(&[0x50, 0x4B, 0x03, 0x04]) {
        return Some(ContentType::BinaryRandom);
    }
    // GZIP
    if data.starts_with(&[0x1F, 0x8B]) {
        return Some(ContentType::BinaryRandom);
    }
    // ZSTD
    if data.starts_with(&[0x28, 0xB5, 0x2F, 0xFD]) {
        return Some(ContentType::BinaryRandom);
    }
    // XZ
    if data.starts_with(&[0xFD, 0x37, 0x7A, 0x58, 0x5A, 0x00]) {
        return Some(ContentType::BinaryRandom);
    }
    // Mach-O (macOS binary)
    if data.starts_with(&[0xCF, 0xFA, 0xED, 0xFE]) || data.starts_with(&[0xFE, 0xED, 0xFA, 0xCF]) {
        return Some(ContentType::Executable);
    }

    // Check if it looks like text (high proportion of printable ASCII)
    let printable_count = data
        .iter()
        .take(512)
        .filter(|&&b| b == b'\n' || b == b'\r' || b == b'\t' || (0x20..=0x7E).contains(&b))
        .count();
    let sample_len = data.len().min(512);
    if sample_len > 0 && (printable_count * 100 / sample_len) > 85 {
        return Some(ContentType::Text);
    }

    None
}

fn detect_from_extension(path: &str) -> ContentType {
    let lower = path.to_lowercase();
    let ext = lower.rsplit('.').next().unwrap_or("");

    match ext {
        // Numeric / tensor data
        "npy" | "npz" | "safetensors" | "gguf" | "ggml" | "bin" | "weight" | "weights" | "pt"
        | "pth" | "ot" | "tflite" | "pb" | "onnx" => ContentType::NumericData,

        // Text / Code
        "txt" | "md" | "rst" | "csv" | "tsv" | "log" => ContentType::Text,
        "rs" | "py" | "js" | "ts" | "c" | "cpp" | "h" | "hpp" | "java" | "go" | "rb" | "php"
        | "swift" | "kt" | "scala" | "zig" | "nim" | "lua" | "sh" | "bash" | "zsh" | "fish"
        | "ps1" | "bat" | "cmd" => ContentType::Text,
        "json" | "yaml" | "yml" | "toml" | "xml" | "html" | "htm" | "css" | "scss" | "less"
        | "sql" | "graphql" | "proto" | "ini" | "cfg" | "conf" => ContentType::Text,

        // Images
        "jpg" | "jpeg" | "png" | "gif" | "bmp" | "tiff" | "tif" | "webp" | "ico" | "svg"
        | "avif" | "heic" | "heif" => ContentType::Image,

        // Already-compressed / encrypted → random
        "zip" | "gz" | "bz2" | "xz" | "zst" | "lz4" | "br" | "7z" | "rar" | "tar" | "aet" => {
            ContentType::BinaryRandom
        }
        "mp3" | "mp4" | "aac" | "ogg" | "flac" | "wav" | "avi" | "mkv" | "webm" | "mov" | "m4a"
        | "m4v" => ContentType::BinaryRandom,

        // Executables
        "exe" | "dll" | "so" | "dylib" | "o" | "a" | "lib" | "elf" | "wasm" => {
            ContentType::Executable
        }

        // Structured binary
        "pdf" | "doc" | "docx" | "xls" | "xlsx" | "ppt" | "pptx" | "odt" | "ods" | "db"
        | "sqlite" | "sqlite3" => ContentType::BinaryStructured,

        // Default
        _ => ContentType::Mixed,
    }
}

// ── Routing ──────────────────────────────────────────────────────────────────

/// Decide the best compression method based on entropy.
pub fn recommend_method(entropy: f64) -> RecommendedMethod {
    if entropy > INCOMPRESSIBLE_THRESHOLD {
        RecommendedMethod::Store
    } else if entropy > HIGH_ENTROPY_THRESHOLD {
        RecommendedMethod::Zstd
    } else {
        RecommendedMethod::PredictorRans
    }
}

/// Decide the best compression method, also considering content type.
/// Already-compressed formats should always be stored or use zstd at most.
pub fn recommend_method_for(entropy: f64, content_type: ContentType) -> RecommendedMethod {
    match content_type {
        // Already compressed: don't waste time with predictor
        ContentType::BinaryRandom => {
            if entropy > INCOMPRESSIBLE_THRESHOLD {
                RecommendedMethod::Store
            } else {
                RecommendedMethod::Zstd
            }
        }
        // Images: usually already compressed (JPEG/PNG), use zstd at best
        ContentType::Image => {
            if entropy > HIGH_ENTROPY_THRESHOLD {
                RecommendedMethod::Store
            } else {
                RecommendedMethod::Zstd
            }
        }
        // Numeric data: predictor with byte-plane splitting
        ContentType::NumericData => recommend_method(entropy),
        // Text, code, structured: predictor shines
        _ => recommend_method(entropy),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_elf() {
        let data = b"\x7fELF\x02\x01\x01\x00";
        assert_eq!(detect_content_type("binary", data), ContentType::Executable);
    }

    #[test]
    fn detect_jpeg() {
        let data = [0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10];
        assert_eq!(detect_content_type("photo.jpg", &data), ContentType::Image);
    }

    #[test]
    fn detect_png() {
        let data = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
        assert_eq!(detect_content_type("icon.png", &data), ContentType::Image);
    }

    #[test]
    fn detect_text_from_content() {
        let data = b"Hello, world! This is a plain text file.\nWith multiple lines.\n";
        assert_eq!(detect_content_type("unknown.dat", data), ContentType::Text);
    }

    #[test]
    fn detect_from_ext() {
        assert_eq!(detect_content_type("main.rs", &[0u8; 4]), ContentType::Text);
        assert_eq!(
            detect_content_type("data.zip", &[0u8; 4]),
            ContentType::BinaryRandom
        );
        assert_eq!(
            detect_content_type("program.exe", &[0u8; 4]),
            ContentType::Executable
        );
    }

    #[test]
    fn routing_thresholds() {
        assert_eq!(recommend_method(3.0), RecommendedMethod::PredictorRans);
        assert_eq!(recommend_method(7.0), RecommendedMethod::PredictorRans);
        assert_eq!(recommend_method(7.6), RecommendedMethod::Zstd);
        assert_eq!(recommend_method(7.96), RecommendedMethod::Store);
    }

    #[test]
    fn routing_already_compressed() {
        // JPEG with high entropy → store
        assert_eq!(
            recommend_method_for(7.8, ContentType::Image),
            RecommendedMethod::Store
        );
        // ZIP data → zstd (might still compress slightly)
        assert_eq!(
            recommend_method_for(7.0, ContentType::BinaryRandom),
            RecommendedMethod::Zstd
        );
        // Text → predictor
        assert_eq!(
            recommend_method_for(4.5, ContentType::Text),
            RecommendedMethod::PredictorRans
        );
    }
}
