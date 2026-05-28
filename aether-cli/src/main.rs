use std::io::{BufReader, Cursor};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use humansize::{format_size as fmt_size, BINARY};
use zeroize::Zeroizing;

use aether_core::entropy::context_mixer::ContextMixerConfig;
use aether_core::entropy::{
    ContextMixer, Lz4AwarePredictor, NeuralSsmPredictor, Order0Model, ProbabilityPredictor,
};
use aether_core::format::PredictorId;
use aether_core::header::ArchiveHeader;
use aether_core::pipeline::compress::Compressor;
use aether_core::pipeline::decompress::Decompressor;

#[derive(Parser)]
#[command(
    name = "aet",
    about = "AetherArch — next-generation file archiver with neural-probabilistic compression",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Compress files into a .aet archive
    #[command(alias = "c")]
    Compress {
        /// Input files or directories
        #[arg(required = true)]
        inputs: Vec<PathBuf>,

        /// Output archive path
        #[arg(short, long)]
        output: PathBuf,

        /// Predictor: order0, cm (context-mixer, default)
        #[arg(short, long, default_value = "cm")]
        predictor: String,

        /// Encrypt the archive with a password (requires enterprise feature).
        /// Use --password without a value to be prompted securely.
        #[arg(long, num_args = 0..=1, default_missing_value = "")]
        password: Option<String>,

        /// Encryption cipher: aes (AES-256-GCM, default), chacha (ChaCha20-Poly1305)
        #[arg(long, default_value = "aes")]
        cipher: String,

        /// Show detailed compression analytics (method breakdown, group stats, timing)
        #[arg(long)]
        analytics: bool,

        /// Path to a pretrained dictionary (.aed) for improved compression
        #[arg(long)]
        dictionary: Option<PathBuf>,

        /// Overwrite output file if it already exists
        #[arg(long)]
        force: bool,
    },

    /// Decompress a .aet archive (use "-" for stdin streaming)
    #[command(alias = "x")]
    Extract {
        /// Input .aet archive (use "-" to read from stdin)
        input: PathBuf,

        /// Output directory
        #[arg(short, long, default_value = ".")]
        output: PathBuf,

        /// Extract a single file by path (not supported in streaming mode)
        #[arg(short, long)]
        file: Option<String>,

        /// Override predictor (auto-detected from archive header by default)
        #[arg(short, long)]
        predictor: Option<String>,

        /// Password for encrypted archives.
        /// Use --password without a value to be prompted securely.
        #[arg(long, num_args = 0..=1, default_missing_value = "")]
        password: Option<String>,

        /// Decompression threads (enterprise): 0=all cores, 1=sequential (default), N=bounded
        #[arg(short = 't', long, default_value = "1")]
        threads: usize,

        /// Path to a pretrained dictionary (.aed) for dictionary-compressed archives
        #[arg(long)]
        dictionary: Option<PathBuf>,
    },

    /// List contents of a .aet archive (use "-" for stdin streaming)
    #[command(alias = "l")]
    List {
        /// Input .aet archive (use "-" to read from stdin)
        input: PathBuf,

        /// Show detailed info (sizes, hashes, groups)
        #[arg(short, long)]
        long: bool,

        /// Override predictor (auto-detected from archive header by default)
        #[arg(short, long)]
        predictor: Option<String>,
    },

    /// Verify integrity of a .aet archive (use "-" for stdin streaming)
    #[command(alias = "v")]
    Verify {
        /// Input .aet archive (use "-" to read from stdin)
        input: PathBuf,

        /// Override predictor (auto-detected from archive header by default)
        #[arg(short, long)]
        predictor: Option<String>,

        /// Password for encrypted archives.
        /// Use --password without a value to be prompted securely.
        #[arg(long, num_args = 0..=1, default_missing_value = "")]
        password: Option<String>,
    },

    /// Benchmark compression on files
    Bench {
        /// Input files
        #[arg(required = true)]
        inputs: Vec<PathBuf>,

        /// Predictors to benchmark (comma-separated)
        #[arg(short = 'P', long, default_value = "order0,cm")]
        predictors: String,

        /// Compare against external compressors (gzip, bzip2, xz, zstd, brotli, lz4) if available on PATH
        #[arg(long)]
        compare: bool,
    },

    /// Train a dictionary from a corpus of files
    Train {
        /// Input training files or directories
        #[arg(required = true)]
        inputs: Vec<PathBuf>,

        /// Output dictionary file (.aed)
        #[arg(short, long)]
        output: PathBuf,

        /// Predictor to train: order0, cm, ssm, rle
        #[arg(short, long, default_value = "ssm")]
        predictor: String,

        /// Overwrite output file if it already exists
        #[arg(long)]
        force: bool,
    },

    /// Migrate an archive: decompress and recompress with new settings
    Migrate {
        /// Source archive file
        input: PathBuf,

        /// Output archive file
        #[arg(short, long)]
        output: PathBuf,

        /// Target predictor: order0, cm, cm-light, lz4, ssm, rle
        #[arg(short, long)]
        predictor: Option<String>,

        /// Source dictionary file (.aed), if source was compressed with one
        #[arg(long = "source-dictionary")]
        source_dictionary: Option<PathBuf>,

        /// Target dictionary file (.aed) to apply
        #[arg(long = "target-dictionary")]
        target_dictionary: Option<PathBuf>,

        /// Password for decrypting encrypted source archive (enterprise feature).
        /// Use without a value to be prompted securely.
        #[arg(long = "source-password", num_args = 0..=1, default_missing_value = "")]
        source_password: Option<String>,

        /// Password for encrypting the target archive (enterprise feature).
        /// Use without a value to be prompted securely.
        #[arg(long = "target-password", num_args = 0..=1, default_missing_value = "")]
        target_password: Option<String>,

        /// Encryption cipher for target archive: aes (AES-256-GCM, default), chacha (ChaCha20-Poly1305)
        #[arg(long = "target-cipher", default_value = "aes")]
        target_cipher: String,

        /// Overwrite output file if it already exists
        #[arg(long)]
        force: bool,
    },
}

fn make_predictor_factory(
    name: &str,
) -> Result<Box<dyn Fn() -> Box<dyn ProbabilityPredictor> + Send + Sync>> {
    match name {
        "order0" | "o0" => Ok(Box::new(|| Box::new(Order0Model::new()))),
        "cm" | "context-mixer" => Ok(Box::new(|| {
            Box::new(ContextMixer::with_config(ContextMixerConfig::default()))
        })),
        "cm-light" => Ok(Box::new(|| {
            Box::new(ContextMixer::with_config(ContextMixerConfig::lightweight()))
        })),
        "lz4" | "lz4-aware" => Ok(Box::new(|| Box::new(Lz4AwarePredictor::new()))),
        "ssm" | "neural-ssm" => Ok(Box::new(|| Box::new(NeuralSsmPredictor::new()))),
        "rle" => Ok(Box::new(|| {
            Box::new(aether_core::entropy::RlePredictor::new())
        })),
        other => anyhow::bail!(
            "Unknown predictor: {other}. Use: order0, cm, cm-light, lz4-aware, ssm, rle"
        ),
    }
}

fn make_predictor_factory_from_id(
    id: PredictorId,
) -> Box<dyn Fn() -> Box<dyn ProbabilityPredictor> + Send + Sync> {
    match id {
        PredictorId::Order0 => Box::new(|| Box::new(Order0Model::new())),
        PredictorId::ContextMixer => {
            Box::new(|| Box::new(ContextMixer::with_config(ContextMixerConfig::default())))
        }
        PredictorId::ContextMixerLight => {
            Box::new(|| Box::new(ContextMixer::with_config(ContextMixerConfig::lightweight())))
        }
        PredictorId::Lz4Aware => Box::new(|| Box::new(Lz4AwarePredictor::new())),
        PredictorId::NeuralSsm => Box::new(|| Box::new(NeuralSsmPredictor::new())),
        PredictorId::Rle => Box::new(|| Box::new(aether_core::entropy::RlePredictor::new())),
        PredictorId::ZstdOnly => {
            // Zstd-only mode doesn't need a predictor, but provide one for API compatibility
            Box::new(|| Box::new(Order0Model::new()))
        }
        _ => {
            // Future predictor IDs: fall back to NeuralSsm as the best general-purpose predictor
            eprintln!("Warning: Unknown predictor ID in archive, falling back to NeuralSsm");
            Box::new(|| Box::new(NeuralSsmPredictor::new()))
        }
    }
}

/// Read the archive header to determine which predictor was used during compression.
fn detect_predictor(path: &Path) -> Result<PredictorId> {
    let mut f =
        std::fs::File::open(path).with_context(|| format!("Cannot open {}", path.display()))?;
    let header =
        ArchiveHeader::read_from(&mut f).with_context(|| "Failed to read archive header")?;
    Ok(header.predictor_id)
}

/// Returns `true` if the input path is `"-"`, indicating stdin streaming mode.
fn is_streaming(input: &Path) -> bool {
    input.as_os_str() == "-"
}

/// Maximum number of files to collect for compression (prevents resource exhaustion).
const MAX_COLLECT_FILES: usize = 1_000_000;

/// Maximum total input size for bench --compare external tool comparison (2 GiB).
const MAX_BENCH_COMPARE_BYTES: u64 = 2 * 1024 * 1024 * 1024;

/// Maximum reasonable thread count for parallel decompression.
const MAX_THREADS: usize = 1024;

/// Minimum password length for encryption.
const MIN_PASSWORD_LENGTH: usize = 8;

/// Maximum recursion depth for directory traversal (prevents stack overflow
/// from deeply nested or circular directory structures).
const MAX_DIR_DEPTH: usize = 256;

fn collect_files(inputs: &[PathBuf]) -> Result<(PathBuf, Vec<PathBuf>)> {
    let mut files = Vec::new();

    // Find the common base directory
    let base_dir = if inputs.len() == 1 && inputs[0].is_dir() {
        inputs[0].clone()
    } else {
        inputs[0]
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf()
    };

    for input in inputs {
        // S-H3: use symlink_metadata() to avoid following symlinks during input
        // collection. This prevents archiving arbitrary files via symlinks placed
        // in the input directory.
        let meta = std::fs::symlink_metadata(input)
            .with_context(|| format!("Cannot stat: {}", input.display()))?;
        if meta.file_type().is_symlink() {
            eprintln!("Warning: skipping symlink: {}", input.display());
            continue;
        }
        if meta.is_file() {
            files.push(input.clone());
        } else if meta.is_dir() {
            collect_dir_recursive(input, &mut files, 0)?;
        } else {
            anyhow::bail!("Not a file or directory: {}", input.display());
        }
        if files.len() > MAX_COLLECT_FILES {
            anyhow::bail!(
                "Too many files (>{MAX_COLLECT_FILES}). Use smaller input sets or archive in batches."
            );
        }
    }

    Ok((base_dir, files))
}

fn collect_dir_recursive(dir: &Path, files: &mut Vec<PathBuf>, depth: usize) -> Result<()> {
    if depth > MAX_DIR_DEPTH {
        anyhow::bail!(
            "Directory recursion depth exceeded ({MAX_DIR_DEPTH}). \
             Possible circular directory structure at: {}",
            dir.display()
        );
    }
    for entry in std::fs::read_dir(dir)
        .with_context(|| format!("Failed to read directory: {}", dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        // S-H3: skip symlinks in recursive directory traversal.
        // We re-check with symlink_metadata to mitigate the TOCTOU window —
        // while not fully atomic, it narrows the race.
        let meta = std::fs::symlink_metadata(&path)
            .with_context(|| format!("Cannot stat: {}", path.display()))?;
        if meta.file_type().is_symlink() {
            eprintln!("Warning: skipping symlink: {}", path.display());
            continue;
        }
        if meta.is_file() {
            files.push(path);
        } else if meta.is_dir() {
            collect_dir_recursive(&path, files, depth + 1)?;
        }
        if files.len() > MAX_COLLECT_FILES {
            anyhow::bail!(
                "Too many files (>{MAX_COLLECT_FILES}). Use smaller input sets or archive in batches."
            );
        }
    }
    Ok(())
}

/// Resolve the absolute path of a command by searching PATH, similar to `which`.
fn which_tool(tool: &str) -> Option<PathBuf> {
    std::env::var_os("PATH").and_then(|paths| {
        std::env::split_paths(&paths).find_map(|dir| {
            let candidate = dir.join(tool);
            // On Windows, also check with common executable extensions
            if candidate.is_file() {
                return Some(candidate);
            }
            #[cfg(target_os = "windows")]
            for ext in &["exe", "cmd", "bat"] {
                let with_ext = candidate.with_extension(ext);
                if with_ext.is_file() {
                    return Some(with_ext);
                }
            }
            None
        })
    })
}

/// S-S6: External tool execution — tool names are hardcoded, not user-supplied.
/// We resolve the absolute path of each tool via `which` and execute it by
/// absolute path to prevent PATH hijacking between resolution and execution.
/// stderr is captured (not nulled) so security warnings from external tools
/// are not swallowed.
fn run_external_compressor(
    tool: &str,
    args: &[&str],
    input: &[u8],
) -> Option<(usize, std::time::Duration)> {
    use std::io::Write;
    use std::process::{Command, Stdio};

    // Resolve absolute path to prevent PATH hijacking
    let resolved = which_tool(tool)?;
    eprintln!("  [{tool}] using: {}", resolved.display());

    let mut child = Command::new(&resolved)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .ok()?;

    let start = Instant::now();

    {
        let stdin = child.stdin.as_mut()?;
        stdin.write_all(input).ok()?;
    }

    let output = child.wait_with_output().ok()?;
    let elapsed = start.elapsed();

    if !output.stderr.is_empty() {
        let stderr_msg = String::from_utf8_lossy(&output.stderr);
        eprintln!("  [{tool}] {}", stderr_msg.trim());
    }

    if output.status.success() {
        Some((output.stdout.len(), elapsed))
    } else {
        None
    }
}

/// Minimum number of distinct character classes required in passwords.
/// Classes: lowercase, uppercase, digits, symbols/other.
const MIN_PASSWORD_CHAR_CLASSES: usize = 2;

/// Minimum number of distinct characters in a password.
const MIN_PASSWORD_DISTINCT_CHARS: usize = 5;

/// Validate password strength: length, distinct characters, and character class diversity.
fn validate_password(pw: &str) -> Result<()> {
    if pw.is_empty() {
        anyhow::bail!("Password cannot be empty");
    }
    if pw.len() < MIN_PASSWORD_LENGTH {
        anyhow::bail!(
            "Password too short (minimum {MIN_PASSWORD_LENGTH} characters). \
             Use a stronger password for meaningful encryption."
        );
    }
    // Reject passwords with too few distinct characters (catches "aaaaaaaa", "abababab", etc.)
    let distinct: std::collections::HashSet<char> = pw.chars().collect();
    if distinct.len() < MIN_PASSWORD_DISTINCT_CHARS {
        anyhow::bail!(
            "Password too weak: use at least {MIN_PASSWORD_DISTINCT_CHARS} distinct characters \
             for meaningful encryption."
        );
    }
    // Require at least 2 character classes (lowercase, uppercase, digits, symbols)
    let has_lower = pw.chars().any(|c| c.is_ascii_lowercase());
    let has_upper = pw.chars().any(|c| c.is_ascii_uppercase());
    let has_digit = pw.chars().any(|c| c.is_ascii_digit());
    let has_symbol = pw.chars().any(|c| !c.is_ascii_alphanumeric());
    let class_count = [has_lower, has_upper, has_digit, has_symbol]
        .iter()
        .filter(|&&v| v)
        .count();
    if class_count < MIN_PASSWORD_CHAR_CLASSES {
        anyhow::bail!(
            "Password too weak: use at least {MIN_PASSWORD_CHAR_CLASSES} character classes \
             (lowercase, uppercase, digits, symbols) for meaningful encryption."
        );
    }
    Ok(())
}

/// S-S2/S3: Resolve a password argument, with strength validation and secure prompting.
///
/// Resolution order:
/// 1. `Some(value)` — use the CLI-provided value (with a warning about process list exposure)
/// 2. `Some("")` — check the `AET_PASSWORD` env var; if unset, prompt interactively
/// 3. `None` — no password
///
/// Returns `Zeroizing<String>` so the password is automatically zeroized on drop,
/// regardless of how the caller handles it.
///
/// Using `AET_PASSWORD` env var or `--password` (no value) for interactive prompting
/// is preferred over passing the password directly on the command line, which is
/// visible in the process list to other users on the system.
fn resolve_password(
    password_arg: Option<String>,
    label: &str,
) -> Result<Option<Zeroizing<String>>> {
    match password_arg {
        None => Ok(None),
        Some(pw) if pw.is_empty() => {
            // Check env var first (safer than CLI args — not visible in process list)
            if let Ok(env_pw) = std::env::var("AET_PASSWORD") {
                if !env_pw.is_empty() {
                    let env_pw = Zeroizing::new(env_pw);
                    validate_password(&env_pw)?;
                    return Ok(Some(env_pw));
                }
            }
            let prompt = format!("{label}: ");
            let pw = Zeroizing::new(
                rpassword::prompt_password(&prompt)
                    .with_context(|| "Failed to read password from terminal")?,
            );
            validate_password(&pw)?;
            Ok(Some(pw))
        }
        Some(pw) => {
            eprintln!("Warning: passing passwords via command line is insecure (visible in process list).");
            eprintln!("         Use --password without a value (prompted) or AET_PASSWORD env var instead.");
            let pw = Zeroizing::new(pw);
            validate_password(&pw)?;
            Ok(Some(pw))
        }
    }
}

/// S-S1: Validate that an extraction target path does not escape the output directory.
/// Prevents "Zip Slip" path traversal attacks via crafted archive entries like
/// `../../etc/passwd` or absolute paths.
fn safe_join(output_dir: &Path, file_path: &str) -> Result<PathBuf> {
    // Reject archive entries that contain absolute path components — these can
    // never be safe to join because they reset the path root.
    let entry = Path::new(file_path);
    for component in entry.components() {
        match component {
            std::path::Component::RootDir | std::path::Component::Prefix(_) => {
                anyhow::bail!(
                    "Path traversal detected: \"{}\" contains an absolute path component",
                    file_path
                );
            }
            _ => {}
        }
    }

    let joined = output_dir.join(file_path);

    // Normalize components to resolve ".." without requiring the path to exist yet
    let mut normalized = PathBuf::new();
    for component in joined.components() {
        match component {
            std::path::Component::ParentDir => {
                if !normalized.pop() {
                    anyhow::bail!(
                        "Path traversal detected: \"{}\" escapes output directory",
                        file_path
                    );
                }
            }
            std::path::Component::Normal(c) => normalized.push(c),
            std::path::Component::RootDir => normalized.push(component),
            std::path::Component::Prefix(p) => normalized.push(p.as_os_str()),
            std::path::Component::CurDir => {}
        }
    }

    // Also normalize the output dir the same way for comparison
    let mut norm_output = PathBuf::new();
    for component in output_dir.components() {
        match component {
            std::path::Component::ParentDir => {
                norm_output.pop();
            }
            std::path::Component::Normal(c) => norm_output.push(c),
            std::path::Component::RootDir => norm_output.push(component),
            std::path::Component::Prefix(p) => norm_output.push(p.as_os_str()),
            std::path::Component::CurDir => {}
        }
    }

    if !normalized.starts_with(&norm_output) {
        anyhow::bail!(
            "Path traversal detected: \"{}\" escapes output directory \"{}\"",
            file_path,
            output_dir.display()
        );
    }

    // Return the normalized path, not the raw join — the raw join may resolve
    // differently on the OS (e.g. via symlink resolution) than our component walk.
    Ok(normalized)
}

/// S-S7: Atomically create the output file, failing if it exists and --force is not set.
/// Returns the opened file handle to eliminate the TOCTOU race between existence check
/// and file creation. When `force` is true, the file is truncated if it exists.
fn create_output_file(path: &Path, force: bool) -> Result<std::fs::File> {
    if force {
        std::fs::File::create(path).with_context(|| format!("Cannot create {}", path.display()))
    } else {
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .map_err(|e| {
                if e.kind() == std::io::ErrorKind::AlreadyExists {
                    anyhow::anyhow!(
                        "Output file already exists: {}. Use --force to overwrite.",
                        path.display()
                    )
                } else {
                    anyhow::anyhow!("Cannot create {}: {}", path.display(), e)
                }
            })
    }
}

/// S-S4: Require enterprise feature when a password is provided.
#[cfg(not(feature = "enterprise"))]
fn require_enterprise_for_password(
    password: &Option<Zeroizing<String>>,
    action: &str,
) -> Result<()> {
    if password.is_some() {
        anyhow::bail!(
            "Encryption/decryption for {action} requires the 'enterprise' feature. \
             Recompile with --features enterprise."
        );
    }
    Ok(())
}

/// Q2: Build a Decompressor with common setup (password, dictionary).
fn build_decompressor(
    factory: Box<dyn Fn() -> Box<dyn ProbabilityPredictor> + Send + Sync>,
    #[allow(unused_variables)] resolved_pw: &Option<Zeroizing<String>>,
    dictionary: &Option<PathBuf>,
) -> Result<Decompressor> {
    let mut decompressor = Decompressor::new(move || factory());

    #[cfg(feature = "enterprise")]
    if let Some(ref pw) = resolved_pw {
        decompressor = decompressor.with_password(pw);
    }

    if let Some(ref dict_path) = dictionary {
        let dict = aether_core::dictionary::Dictionary::load(dict_path)
            .with_context(|| format!("Cannot load dictionary: {}", dict_path.display()))?;
        decompressor = decompressor.with_dictionary(dict);
        eprintln!("Using dictionary: {}", dict_path.display());
    }

    Ok(decompressor)
}

/// Q2: Resolve the predictor factory — either from an explicit name or auto-detected from the archive.
fn resolve_predictor_factory(
    predictor_name: &Option<String>,
    archive_path: &Path,
) -> Result<Box<dyn Fn() -> Box<dyn ProbabilityPredictor> + Send + Sync>> {
    if let Some(ref name) = predictor_name {
        make_predictor_factory(name)
    } else {
        let id = detect_predictor(archive_path)?;
        eprintln!("Auto-detected predictor: {:?}", id);
        Ok(make_predictor_factory_from_id(id))
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Compress {
            inputs,
            output,
            predictor,
            mut password,
            cipher,
            analytics: show_analytics,
            dictionary,
            force,
        } => {
            // S10: warn if --cipher was explicitly set without enterprise feature
            #[cfg(not(feature = "enterprise"))]
            if cipher != "aes" {
                eprintln!(
                    "Warning: --cipher '{cipher}' ignored without the 'enterprise' feature. \
                     Recompile with --features enterprise to enable encryption."
                );
            }
            // S7: atomic overwrite protection — obtain the file handle early so
            // there is no TOCTOU gap between the existence check and create.
            let out_file = create_output_file(&output, force)?;

            let factory = make_predictor_factory(&predictor)?;
            let mut compressor = Compressor::new(move || factory());

            if let Some(ref dict_path) = dictionary {
                let dict = aether_core::dictionary::Dictionary::load(dict_path)
                    .with_context(|| format!("Cannot load dictionary: {}", dict_path.display()))?;
                eprintln!(
                    "Dictionary: {} ({:?})",
                    dict_path.display(),
                    dict.predictor_id
                );
                compressor = compressor.with_dictionary(dict);
            }

            let resolved_pw = resolve_password(password.take(), "Password")?;

            // S4: require enterprise for encryption
            #[cfg(feature = "enterprise")]
            if let Some(ref pw) = resolved_pw {
                let cipher_id = match cipher.as_str() {
                    "aes" | "aes-gcm" | "aes256" => aether_core::crypto::CipherId::Aes256Gcm,
                    "chacha" | "chacha20" => aether_core::crypto::CipherId::ChaCha20Poly1305,
                    other => anyhow::bail!("Unknown cipher: {other}. Use: aes, chacha"),
                };
                compressor = compressor.with_encryption(pw, cipher_id);
                eprintln!("Encryption: {:?}", cipher_id);
            }
            #[cfg(not(feature = "enterprise"))]
            require_enterprise_for_password(&resolved_pw, "compression")?;

            let (base_dir, files) = collect_files(&inputs)?;

            if files.is_empty() {
                anyhow::bail!("No files to compress");
            }

            eprintln!(
                "Compressing {} file(s) with {} predictor...",
                files.len(),
                predictor
            );

            let start = Instant::now();
            let mut out_file = out_file;

            let (stats, analytics) =
                compressor.compress_to_archive(&base_dir, &files, &mut out_file)?;
            let elapsed = start.elapsed();

            let archive_size = std::fs::metadata(&output)?.len();

            // S2: password automatically zeroized on drop via Zeroizing<String>
            drop(resolved_pw);

            eprintln!();
            eprintln!("  Archive:     {}", output.display());
            eprintln!("  Files:       {}", stats.file_count);
            eprintln!("  Blocks:      {}", stats.block_count);
            eprintln!("  Groups:      {}", stats.group_count);
            eprintln!("  Original:    {}", fmt_size(stats.original_size, BINARY));
            eprintln!("  Compressed:  {}", fmt_size(archive_size, BINARY));
            eprintln!("  Ratio:       {:.2}%", stats.ratio() * 100.0);
            eprintln!("  Bits/byte:   {:.3}", stats.bits_per_byte());
            eprintln!("  Time:        {:.2?}", elapsed);

            if stats.original_size > 0 {
                let speed = stats.original_size as f64 / elapsed.as_secs_f64() / (1024.0 * 1024.0);
                eprintln!("  Speed:       {speed:.1} MiB/s");
            }

            if show_analytics {
                eprintln!();
                eprintln!("  --- Analytics ---");
                eprintln!("  Compression phase: {:.2?}", analytics.compression_time);
                eprintln!("  Write phase:       {:.2?}", analytics.write_time);
                eprintln!();
                eprintln!("  Method breakdown:");
                let mut methods: Vec<_> = analytics.method_counts.iter().collect();
                methods.sort_by(|a, b| b.1.cmp(a.1));
                for (method, count) in &methods {
                    let orig = analytics
                        .method_bytes_original
                        .get(method)
                        .copied()
                        .unwrap_or(0);
                    let comp = analytics
                        .method_bytes_compressed
                        .get(method)
                        .copied()
                        .unwrap_or(0);
                    let ratio = if orig > 0 {
                        comp as f64 / orig as f64 * 100.0
                    } else {
                        0.0
                    };
                    eprintln!(
                        "    {:?}: {} block(s), {} -> {} ({:.1}%)",
                        method,
                        count,
                        fmt_size(orig, BINARY),
                        fmt_size(comp, BINARY),
                        ratio,
                    );
                }

                if analytics.group_stats.len() > 1 {
                    eprintln!();
                    eprintln!("  Group breakdown:");
                    for g in &analytics.group_stats {
                        let ratio = if g.original_size > 0 {
                            g.compressed_size as f64 / g.original_size as f64 * 100.0
                        } else {
                            0.0
                        };
                        eprintln!(
                            "    Group {} ({:?}): {} block(s), {} -> {} ({:.1}%)",
                            g.group_id,
                            g.content_type,
                            g.block_count,
                            fmt_size(g.original_size, BINARY),
                            fmt_size(g.compressed_size, BINARY),
                            ratio,
                        );
                    }
                }
            }
        }

        Commands::Extract {
            input,
            output,
            file,
            predictor,
            mut password,
            threads,
            dictionary,
        } => {
            let resolved_pw = resolve_password(password.take(), "Password")?;

            // S4: require enterprise for decryption
            #[cfg(not(feature = "enterprise"))]
            require_enterprise_for_password(&resolved_pw, "extraction")?;

            // Q7: validate thread count
            if threads > MAX_THREADS {
                anyhow::bail!(
                    "Thread count {threads} exceeds maximum ({MAX_THREADS}). \
                     Use 0 for all cores or a reasonable value."
                );
            }

            if is_streaming(&input) {
                // ── Streaming mode: read from stdin ─────────────────────
                if file.is_some() {
                    anyhow::bail!("Single-file extraction (-f) is not supported in streaming mode.\nUse a file path instead of \"-\" for single-file extraction.");
                }

                let mut stdin = BufReader::new(std::io::stdin());
                let start = Instant::now();

                let metadata = Decompressor::read_metadata_streaming(&mut stdin)
                    .with_context(|| "Failed to read streaming metadata")?;

                let factory = if let Some(ref name) = predictor {
                    make_predictor_factory(name)?
                } else {
                    let id = metadata.header.predictor_id;
                    eprintln!("Auto-detected predictor: {:?}", id);
                    make_predictor_factory_from_id(id)
                };
                let decompressor = build_decompressor(factory, &resolved_pw, &dictionary)?;

                std::fs::create_dir_all(&output)?;
                decompressor.extract_with_streaming_metadata(&mut stdin, &metadata, &output)?;

                let elapsed = start.elapsed();
                eprintln!("Extracted to: {} (streaming)", output.display());
                eprintln!("Time: {:.2?}", elapsed);
            } else {
                // ── Seekable mode: read from file ───────────────────────
                let factory = resolve_predictor_factory(&predictor, &input)?;
                #[allow(unused_mut)]
                let mut decompressor = build_decompressor(factory, &resolved_pw, &dictionary)?;

                #[cfg(feature = "enterprise")]
                if threads != 1 {
                    decompressor = decompressor.with_max_threads(threads);
                    eprintln!(
                        "Parallel decompression: {} thread(s)",
                        if threads == 0 {
                            "all".to_string()
                        } else {
                            threads.to_string()
                        }
                    );
                }
                #[cfg(not(feature = "enterprise"))]
                if threads != 1 {
                    eprintln!("Warning: --threads requires the 'enterprise' feature. Using sequential decompression.");
                }

                let mut archive = std::fs::File::open(&input)
                    .with_context(|| format!("Cannot open {}", input.display()))?;

                let start = Instant::now();

                if let Some(ref file_path) = file {
                    // S1: validate extraction path against directory traversal
                    let out_path = safe_join(&output, file_path)?;
                    if let Some(parent) = out_path.parent() {
                        std::fs::create_dir_all(parent)?;
                    }
                    let mut out_file = std::fs::File::create(&out_path)?;
                    decompressor.extract_file(&mut archive, file_path, &mut out_file)?;
                    eprintln!("Extracted: {}", out_path.display());
                } else {
                    std::fs::create_dir_all(&output)?;
                    decompressor.extract_all(&mut archive, &output)?;
                    eprintln!("Extracted to: {}", output.display());
                }

                let elapsed = start.elapsed();
                let archive_size = std::fs::metadata(&input)?.len();
                eprintln!("Time: {:.2?}", elapsed);
                if archive_size > 0 {
                    let speed = archive_size as f64 / elapsed.as_secs_f64() / (1024.0 * 1024.0);
                    eprintln!("Speed: {speed:.1} MiB/s (archive throughput)");
                }
            }

            // S2: password automatically zeroized on drop via Zeroizing<String>
            drop(resolved_pw);
        }

        Commands::List {
            input,
            long,
            predictor,
        } => {
            let entries = if is_streaming(&input) {
                let mut stdin = BufReader::new(std::io::stdin());
                Decompressor::list_files_streaming(&mut stdin)
                    .with_context(|| "Failed to list files (streaming)")?
            } else {
                let factory = resolve_predictor_factory(&predictor, &input)?;
                let decompressor = Decompressor::new(move || factory());

                let mut archive = std::fs::File::open(&input)
                    .with_context(|| format!("Cannot open {}", input.display()))?;

                decompressor.list_files(&mut archive)?
            };

            if long {
                println!("{:<10}  {:<8}  {:<40}  Path", "Size", "Group", "BLAKE3");
                println!("{}", "-".repeat(80));
                for entry in &entries {
                    let hash_short: String = entry.blake3_hash[..4]
                        .iter()
                        .map(|b| format!("{b:02x}"))
                        .collect();
                    println!(
                        "{:<10}  {:<8}  {:<40}  {}",
                        fmt_size(entry.original_size, BINARY),
                        entry.solid_group_id,
                        format!("{hash_short}..."),
                        entry.path
                    );
                }
            } else {
                for entry in &entries {
                    println!(
                        "{:>10}  {}",
                        fmt_size(entry.original_size, BINARY),
                        entry.path
                    );
                }
            }

            let total: u64 = entries.iter().map(|e| e.original_size).sum();
            println!();
            println!(
                "{} file(s), {} total",
                entries.len(),
                fmt_size(total, BINARY)
            );
        }

        Commands::Verify {
            input,
            predictor,
            mut password,
        } => {
            let resolved_pw = resolve_password(password.take(), "Password")?;

            // S4: require enterprise for decryption
            #[cfg(not(feature = "enterprise"))]
            require_enterprise_for_password(&resolved_pw, "verification")?;

            let result = if is_streaming(&input) {
                // ── Streaming verify ────────────────────────────────────
                let mut stdin = BufReader::new(std::io::stdin());

                let metadata = Decompressor::read_metadata_streaming(&mut stdin)
                    .with_context(|| "Failed to read streaming metadata")?;

                let factory = if let Some(ref name) = predictor {
                    make_predictor_factory(name)?
                } else {
                    let id = metadata.header.predictor_id;
                    eprintln!("Auto-detected predictor: {:?}", id);
                    make_predictor_factory_from_id(id)
                };
                let decompressor = build_decompressor(factory, &resolved_pw, &None)?;

                eprintln!("Verifying (streaming)...");
                decompressor.verify_with_streaming_metadata(&mut stdin, &metadata)?
            } else {
                // ── Seekable verify ─────────────────────────────────────
                let factory = resolve_predictor_factory(&predictor, &input)?;
                let decompressor = build_decompressor(factory, &resolved_pw, &None)?;

                let mut archive = std::fs::File::open(&input)
                    .with_context(|| format!("Cannot open {}", input.display()))?;

                eprintln!("Verifying {}...", input.display());
                decompressor.verify(&mut archive)?
            };

            // S2: password automatically zeroized on drop via Zeroizing<String>
            drop(resolved_pw);

            if result.is_ok() {
                eprintln!(
                    "OK: {}/{} blocks verified",
                    result.verified_blocks, result.total_blocks
                );
            } else {
                eprintln!(
                    "CORRUPTED: {} block(s) failed verification",
                    result.corrupted_blocks.len()
                );
                for block_id in &result.corrupted_blocks {
                    eprintln!("  Block {block_id}: FAILED");
                }
                std::process::exit(1);
            }
        }

        Commands::Bench {
            inputs,
            predictors,
            compare,
        } => {
            let (base_dir, files) = collect_files(&inputs)?;

            if files.is_empty() {
                anyhow::bail!("No files to benchmark");
            }

            let total_size: u64 = files
                .iter()
                .map(|f| std::fs::metadata(f).map(|m| m.len()).unwrap_or(0))
                .sum();

            // S5: cap memory allocation for external tool comparison
            let concatenated: Vec<u8> = if compare {
                if total_size > MAX_BENCH_COMPARE_BYTES {
                    anyhow::bail!(
                        "Input too large for --compare ({} > {}). \
                         External tool comparison buffers all input in memory. \
                         Use a smaller input set or omit --compare.",
                        fmt_size(total_size, BINARY),
                        fmt_size(MAX_BENCH_COMPARE_BYTES, BINARY),
                    );
                }
                let alloc_size = usize::try_from(total_size)
                    .with_context(|| "Input size exceeds addressable memory")?;
                let mut data = Vec::with_capacity(alloc_size);
                for f in &files {
                    data.extend_from_slice(&std::fs::read(f)?);
                }
                data
            } else {
                Vec::new()
            };

            println!(
                "Benchmarking {} file(s), {} total",
                files.len(),
                fmt_size(total_size, BINARY),
            );
            println!();
            println!(
                "{:<16} {:>10} {:>12} {:>10} {:>10} {:>10}",
                "Predictor", "Comp MB/s", "Decomp MB/s", "Ratio", "Bits/byte", "Time"
            );
            println!("{}", "-".repeat(75));

            for pred_name in predictors.split(',') {
                let pred_name = pred_name.trim();

                // ── Compression pass ─────────────────────────────────
                let factory = make_predictor_factory(pred_name)?;
                let compressor = Compressor::new(move || factory());
                let start = Instant::now();
                let mut buf = Cursor::new(Vec::new());
                let (stats, _analytics) =
                    compressor.compress_to_archive(&base_dir, &files, &mut buf)?;
                let elapsed = start.elapsed();
                let comp_speed =
                    stats.original_size as f64 / elapsed.as_secs_f64() / (1024.0 * 1024.0);

                // ── Decompression pass ───────────────────────────────
                buf.set_position(0);
                let decomp_factory = make_predictor_factory(pred_name)?;
                let decompressor = Decompressor::new(move || decomp_factory());
                let decomp_start = Instant::now();
                let _ = decompressor.verify(&mut buf)?;
                let decomp_elapsed = decomp_start.elapsed();
                let decomp_speed =
                    stats.original_size as f64 / decomp_elapsed.as_secs_f64() / (1024.0 * 1024.0);

                println!(
                    "{:<16} {:>9.1} {:>11.1} {:>9.2}% {:>10.3} {:>9.2?}",
                    pred_name,
                    comp_speed,
                    decomp_speed,
                    stats.ratio() * 100.0,
                    stats.bits_per_byte(),
                    elapsed,
                );
            }

            // ── External tool comparison ─────────────────────────────
            if compare {
                println!();
                println!("External compressor comparison:");
                println!(
                    "{:<16} {:>10} {:>12} {:>10} {:>10}",
                    "Tool", "Comp MB/s", "Size", "Ratio", "Bits/byte"
                );
                println!("{}", "-".repeat(62));

                let external_tools: &[(&str, &[&str])] = &[
                    ("gzip -9", &["gzip", "-9", "-c"]),
                    ("bzip2 -9", &["bzip2", "-9", "-c"]),
                    ("xz -9", &["xz", "-9", "-c"]),
                    ("zstd -19", &["zstd", "-19", "-c", "--no-progress"]),
                    ("brotli -q 11", &["brotli", "-q", "11", "-c"]),
                    ("lz4 -9", &["lz4", "-9", "-c", "--no-frame-crc"]),
                ];

                for (label, args) in external_tools {
                    match run_external_compressor(args[0], &args[1..], &concatenated) {
                        Some((compressed_size, duration)) => {
                            let ratio = compressed_size as f64 / total_size as f64;
                            let bpb = ratio * 8.0;
                            let speed =
                                total_size as f64 / duration.as_secs_f64() / (1024.0 * 1024.0);
                            println!(
                                "{:<16} {:>9.1} {:>12} {:>9.2}% {:>10.3}",
                                label,
                                speed,
                                fmt_size(compressed_size as u64, BINARY),
                                ratio * 100.0,
                                bpb,
                            );
                        }
                        None => {
                            println!("{:<16} (not found on PATH)", label);
                        }
                    }
                }
            }
        }

        Commands::Train {
            inputs,
            output,
            predictor,
            force,
        } => {
            // S7: atomic overwrite protection (consistent with Compress and Migrate)
            let out_file = create_output_file(&output, force)?;

            let factory = make_predictor_factory(&predictor)?;
            let mut predictor_instance = factory();
            let predictor_id = predictor_instance.predictor_id();

            let (_base_dir, training_files) = collect_files(&inputs)?;

            if training_files.is_empty() {
                anyhow::bail!("No training files found");
            }

            // NeuralSSM dictionaries are used as the BWT coding path's
            // per-block reset baseline (Stage A), so they must be trained on
            // the same BWT+MTF+RLE-transformed stream the coder sees. Other
            // predictors seed the raw-byte plain path, so train on raw bytes.
            let transformed = matches!(predictor_id, PredictorId::NeuralSsm);
            eprintln!(
                "Training {:?} dictionary on {} file(s){}...",
                predictor_id,
                training_files.len(),
                if transformed {
                    " (BWT+MTF+RLE transformed)"
                } else {
                    ""
                }
            );

            let dict = if transformed {
                aether_core::dictionary::Dictionary::train_transformed(
                    predictor_instance.as_mut(),
                    &training_files,
                )?
            } else {
                aether_core::dictionary::Dictionary::train(
                    predictor_instance.as_mut(),
                    &training_files,
                )?
            };
            let mut out_file = out_file;
            dict.write_to(&mut out_file)?;

            eprintln!("Dictionary saved: {}", output.display());
            eprintln!("  Predictor: {:?}", predictor_id);
            eprintln!("  State size: {}", fmt_size(dict.state.len(), BINARY));
            eprintln!(
                "  Hash: {}",
                dict.hash
                    .iter()
                    .map(|b| format!("{b:02x}"))
                    .collect::<String>()
            );
        }

        Commands::Migrate {
            input,
            output,
            predictor,
            source_dictionary,
            target_dictionary,
            mut source_password,
            mut target_password,
            target_cipher,
            force,
        } => {
            // S10: warn if --target-cipher was explicitly set without enterprise feature
            #[cfg(not(feature = "enterprise"))]
            if target_cipher != "aes" {
                eprintln!(
                    "Warning: --target-cipher '{target_cipher}' ignored without the 'enterprise' feature. \
                     Recompile with --features enterprise to enable encryption."
                );
            }
            use aether_core::pipeline::migrate::Migrator;

            // S7: atomic overwrite protection
            let out_file = create_output_file(&output, force)?;

            let source_id = detect_predictor(&input)?;
            eprintln!("Source predictor: {:?}", source_id);
            let source_factory = make_predictor_factory_from_id(source_id);

            let target_factory = if let Some(ref name) = predictor {
                eprintln!("Target predictor: {name}");
                make_predictor_factory(name)?
            } else {
                eprintln!("Target predictor: {:?} (same as source)", source_id);
                make_predictor_factory_from_id(source_id)
            };

            let mut migrator = Migrator::new(move || source_factory(), move || target_factory());

            if let Some(ref dict_path) = source_dictionary {
                let dict =
                    aether_core::dictionary::Dictionary::load(dict_path).with_context(|| {
                        format!("Cannot load source dictionary: {}", dict_path.display())
                    })?;
                migrator = migrator.with_source_dictionary(dict);
                eprintln!("Source dictionary: {}", dict_path.display());
            }

            if let Some(ref dict_path) = target_dictionary {
                let dict =
                    aether_core::dictionary::Dictionary::load(dict_path).with_context(|| {
                        format!("Cannot load target dictionary: {}", dict_path.display())
                    })?;
                migrator = migrator.with_target_dictionary(dict);
                eprintln!("Target dictionary: {}", dict_path.display());
            }

            let resolved_src_pw = resolve_password(source_password.take(), "Source password")?;
            let resolved_tgt_pw = resolve_password(target_password.take(), "Target password")?;

            // S4: require enterprise for encryption/decryption
            #[cfg(feature = "enterprise")]
            {
                if let Some(ref pw) = resolved_src_pw {
                    migrator = migrator.with_source_password(pw);
                }
                if let Some(ref pw) = resolved_tgt_pw {
                    let cipher_id = match target_cipher.as_str() {
                        "aes" | "aes-gcm" | "aes256" => aether_core::crypto::CipherId::Aes256Gcm,
                        "chacha" | "chacha20" => aether_core::crypto::CipherId::ChaCha20Poly1305,
                        other => anyhow::bail!("Unknown cipher: {other}. Use: aes, chacha"),
                    };
                    migrator = migrator.with_target_password(pw, cipher_id);
                    eprintln!("Target encryption: {:?}", cipher_id);
                }
            }
            #[cfg(not(feature = "enterprise"))]
            {
                require_enterprise_for_password(&resolved_src_pw, "migration (source decryption)")?;
                require_enterprise_for_password(&resolved_tgt_pw, "migration (target encryption)")?;
            }

            let start = Instant::now();

            let mut source = std::fs::File::open(&input)
                .with_context(|| format!("Cannot open {}", input.display()))?;
            let mut out_file = out_file;

            let file_count = migrator.migrate(&mut source, &mut out_file)?;

            let elapsed = start.elapsed();
            let out_size = std::fs::metadata(&output)?.len();

            // S2: passwords automatically zeroized on drop via Zeroizing<String>
            drop((resolved_src_pw, resolved_tgt_pw));

            eprintln!(
                "Migrated {} file(s) → {} ({:.2?})",
                file_count,
                output.display(),
                elapsed,
            );
            eprintln!("Output size: {}", fmt_size(out_size, BINARY));
        }
    }

    Ok(())
}
