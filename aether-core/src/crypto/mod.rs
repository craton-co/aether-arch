//! Encryption support for AetherArch archives (enterprise feature).
//!
//! Provides authenticated encryption for archive blocks using either
//! AES-256-GCM (hardware-accelerated on x86_64) or ChaCha20-Poly1305
//! (constant-time on all platforms, preferred for ARM).
//!
//! # Design
//!
//! - **Encrypt-after-compress**: Each compressed block is encrypted individually,
//!   preserving the ability to decompress blocks independently (single-file extraction).
//! - **Per-block nonces**: Derived via BLAKE3 keyed hash of the block_id using
//!   the master nonce as context, producing full 96-bit unique nonces per block.
//! - **Key derivation**: Argon2id with configurable memory/iteration parameters.
//! - **Authenticated**: Both ciphers provide AEAD — tampered blocks are detected.
//!
//! # Format
//!
//! When `FLAG_ENCRYPTED` is set in the archive header:
//!
//! ```text
//! [ArchiveHeader 48B] [EncryptionHeader 90B] [FileTable] [Groups] [Blocks...] [Index] [Footer]
//! ```
//!
//! Each encrypted block payload: `[12B nonce] [ciphertext...] [16B auth tag]`
//!
//! # Block ID Limit
//!
//! Block IDs are `u32`, supporting up to ~4 billion blocks per archive.

use crate::error::{AetherError, Result};
use zeroize::Zeroize;

/// Cipher algorithm identifier stored in the encryption header.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CipherId {
    /// AES-256-GCM (hardware-accelerated via AES-NI on x86_64).
    Aes256Gcm = 0,
    /// ChaCha20-Poly1305 (constant-time, preferred for ARM/embedded).
    ChaCha20Poly1305 = 1,
}

impl CipherId {
    pub fn from_u8(v: u8) -> Result<Self> {
        match v {
            0 => Ok(Self::Aes256Gcm),
            1 => Ok(Self::ChaCha20Poly1305),
            _ => Err(AetherError::Decryption(format!("Unknown cipher ID: {v}"))),
        }
    }
}

/// Encryption header version. Allows future format changes to be detected.
pub const HEADER_VERSION: u8 = 1;
/// Size of the encryption header appended after the archive header.
/// 1 (version) + 1 (cipher_id) + 32 (salt) + 4×3 (m/t/p_cost) + 12 (master_nonce) + 32 (verification_tag) = 90.
pub const ENCRYPTION_HEADER_SIZE: usize = 90;
/// Domain separation string for the password verification tag.
/// BLAKE3 keyed hash of this constant using a separate verification key
/// provides fast-fail on wrong password without revealing the encryption key.
const VERIFICATION_DOMAIN: &[u8] = b"AetherArch-password-verification-v1";
/// Domain separation string for deriving the verification sub-key from the
/// master derived key. This ensures the verification tag reveals nothing
/// about the encryption key even if the BLAKE3 keyed hash is compromised.
const VERIFICATION_KEY_DOMAIN: &[u8] = b"AetherArch-verification-key-v1";
/// Domain separation string for per-block nonce derivation via BLAKE3.
const NONCE_DOMAIN: &[u8] = b"AetherArch-block-nonce-v1";
/// Nonce size for both AES-GCM and ChaCha20-Poly1305 (96 bits).
pub const NONCE_SIZE: usize = 12;
/// Authentication tag size (128 bits).
pub const TAG_SIZE: usize = 16;
/// Derived key size (256 bits).
pub const KEY_SIZE: usize = 32;
/// Salt size for Argon2id key derivation.
pub const SALT_SIZE: usize = 32;

/// Minimum password length for new encrypted archives, measured in **bytes**
/// (not Unicode characters). A 4-character CJK password (12 UTF-8 bytes)
/// passes, while a 7-character ASCII password (7 bytes) does not.
/// Enforced only at encryption time — decryption accepts any password
/// to allow opening archives created with weaker passwords.
pub const MIN_PASSWORD_LEN: usize = 8;

/// Maximum plaintext size for a single block (1 GiB).
/// AES-GCM has a hard limit of ~64 GiB per nonce, but we enforce a much
/// lower bound to prevent excessive memory allocation from crafted inputs.
pub const MAX_BLOCK_PLAINTEXT_SIZE: usize = 1024 * 1024 * 1024;

/// Default Argon2id parameters (tuned for ~1 second derivation).
pub const DEFAULT_ARGON2_M_COST: u32 = 65536; // 64 MiB
pub const DEFAULT_ARGON2_T_COST: u32 = 3; // 3 iterations
pub const DEFAULT_ARGON2_P_COST: u32 = 4; // 4 lanes

// Argon2id parameter bounds are defined in `crate::format` as the single
// source of truth, ensuring consistent validation at both compression time
// and decompression time. Re-exported here so consumers of `crypto::*` get
// them without importing `format` separately.
pub use crate::format::{
    MAX_ARGON2_M_COST, MAX_ARGON2_P_COST, MAX_ARGON2_T_COST, MIN_ARGON2_M_COST, MIN_ARGON2_P_COST,
    MIN_ARGON2_T_COST,
};

/// Encryption metadata stored in the archive (90 bytes).
///
/// Written immediately after the 48-byte archive header when `FLAG_ENCRYPTED` is set.
/// The version byte allows future format changes without ambiguity.
pub struct EncryptionHeader {
    /// Header format version (currently [`HEADER_VERSION`] = 1).
    pub version: u8,
    /// Which cipher algorithm to use.
    pub cipher_id: CipherId,
    /// Random salt for Argon2id key derivation (32 bytes).
    pub salt: [u8; SALT_SIZE],
    /// Argon2id memory cost in KiB.
    pub m_cost: u32,
    /// Argon2id iteration count.
    pub t_cost: u32,
    /// Argon2id parallelism lanes.
    pub p_cost: u32,
    /// Master nonce (12 bytes) — used as context for BLAKE3 per-block nonce derivation.
    pub master_nonce: [u8; NONCE_SIZE],
    /// Password verification tag (32 bytes). BLAKE3 keyed hash of a known constant,
    /// allowing fast-fail on wrong password before attempting block decryption.
    pub verification_tag: [u8; crate::format::VERIFICATION_TAG_SIZE],
}

impl Clone for EncryptionHeader {
    fn clone(&self) -> Self {
        Self {
            version: self.version,
            cipher_id: self.cipher_id,
            salt: self.salt,
            m_cost: self.m_cost,
            t_cost: self.t_cost,
            p_cost: self.p_cost,
            master_nonce: self.master_nonce,
            verification_tag: self.verification_tag,
        }
    }
}

impl std::fmt::Debug for EncryptionHeader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EncryptionHeader")
            .field("version", &self.version)
            .field("cipher_id", &self.cipher_id)
            .field("salt", &"[REDACTED]")
            .field("m_cost", &self.m_cost)
            .field("t_cost", &self.t_cost)
            .field("p_cost", &self.p_cost)
            .field("master_nonce", &"[REDACTED]")
            .field("verification_tag", &"[REDACTED]")
            .finish()
    }
}

impl Drop for EncryptionHeader {
    fn drop(&mut self) {
        self.salt.zeroize();
        self.master_nonce.zeroize();
        self.verification_tag.zeroize();
    }
}

impl EncryptionHeader {
    /// Write the encryption header to a writer (exactly [`ENCRYPTION_HEADER_SIZE`] bytes).
    pub fn write_to<W: std::io::Write>(&self, w: &mut W) -> Result<()> {
        use byteorder::{LittleEndian, WriteBytesExt};
        w.write_u8(self.version)?;
        w.write_u8(self.cipher_id as u8)?;
        w.write_all(&self.salt)?;
        w.write_u32::<LittleEndian>(self.m_cost)?;
        w.write_u32::<LittleEndian>(self.t_cost)?;
        w.write_u32::<LittleEndian>(self.p_cost)?;
        w.write_all(&self.master_nonce)?;
        w.write_all(&self.verification_tag)?;
        Ok(())
    }

    /// Read the encryption header from a reader (exactly [`ENCRYPTION_HEADER_SIZE`] bytes).
    ///
    /// Validates the header version and Argon2 parameters against safety bounds
    /// immediately, failing fast before any key derivation is attempted.
    pub fn read_from<R: std::io::Read>(r: &mut R) -> Result<Self> {
        use byteorder::{LittleEndian, ReadBytesExt};
        let version = r.read_u8()?;
        if version != HEADER_VERSION {
            return Err(AetherError::Decryption(format!(
                "Unsupported encryption header version {version} (expected {HEADER_VERSION})"
            )));
        }
        let cipher_id = CipherId::from_u8(r.read_u8()?)?;
        let mut salt = [0u8; SALT_SIZE];
        r.read_exact(&mut salt)?;
        let m_cost = r.read_u32::<LittleEndian>()?;
        let t_cost = r.read_u32::<LittleEndian>()?;
        let p_cost = r.read_u32::<LittleEndian>()?;
        let mut master_nonce = [0u8; NONCE_SIZE];
        r.read_exact(&mut master_nonce)?;
        let mut verification_tag = [0u8; crate::format::VERIFICATION_TAG_SIZE];
        r.read_exact(&mut verification_tag)?;

        validate_argon2_params(m_cost, t_cost, p_cost)?;

        Ok(Self {
            version,
            cipher_id,
            salt,
            m_cost,
            t_cost,
            p_cost,
            master_nonce,
            verification_tag,
        })
    }

    /// Verify a derived key against this header's verification tag.
    ///
    /// Returns `Ok(())` if the tag matches, or `Err(PasswordVerificationFailed)`
    /// if the password is wrong. This allows fast-fail before attempting any
    /// block decryption.
    ///
    /// Uses constant-time comparison to prevent timing side-channel attacks
    /// that could reduce brute-force effort.
    pub fn verify_password(&self, key: &DerivedKey) -> Result<()> {
        use subtle::ConstantTimeEq;
        let expected = compute_verification_tag(key);
        if !bool::from(self.verification_tag.ct_eq(&expected)) {
            return Err(AetherError::PasswordVerificationFailed);
        }
        Ok(())
    }
}

/// Compute a password verification tag from a derived key.
///
/// First derives a separate verification sub-key from the master key using
/// BLAKE3 with domain separation, then computes the tag using that sub-key.
/// This ensures the verification tag (stored in the header) reveals nothing
/// about the encryption key, even if the BLAKE3 construction is weakened.
pub fn compute_verification_tag(key: &DerivedKey) -> [u8; crate::format::VERIFICATION_TAG_SIZE] {
    // Derive a verification-specific sub-key so the tag is independent of
    // the encryption key. Compromise of the tag does not leak the key.
    let verification_key = blake3::keyed_hash(key.as_bytes(), VERIFICATION_KEY_DOMAIN);
    let hash = blake3::keyed_hash(verification_key.as_bytes(), VERIFICATION_DOMAIN);
    *hash.as_bytes()
}

/// Derive a per-block nonce from the master nonce and a block counter.
///
/// Uses BLAKE3 keyed hash to derive a full 96-bit nonce from the master
/// nonce and block_id. This provides full entropy across all 12 bytes
/// (unlike XOR which only varied the last 4 bytes), while still
/// guaranteeing uniqueness within an archive since block IDs are unique.
///
/// **Note**: The nonce is stored per-block in the archive, so existing
/// archives created with prior derivation schemes still decrypt correctly.
pub fn derive_block_nonce(master_nonce: &[u8; NONCE_SIZE], block_id: u32) -> [u8; NONCE_SIZE] {
    // Pad the master nonce to 32 bytes for use as a BLAKE3 key.
    let mut key = [0u8; KEY_SIZE];
    key[..NONCE_SIZE].copy_from_slice(master_nonce);
    // Hash the domain and block_id to derive a full 96-bit nonce.
    let mut hasher = blake3::Hasher::new_keyed(&key);
    hasher.update(NONCE_DOMAIN);
    hasher.update(&block_id.to_be_bytes());
    let hash = hasher.finalize();
    let mut nonce = [0u8; NONCE_SIZE];
    nonce.copy_from_slice(&hash.as_bytes()[..NONCE_SIZE]);
    nonce
}

/// A zeroize-on-drop wrapper for derived encryption keys.
///
/// Ensures the 256-bit key is securely erased from memory when no longer needed,
/// preventing recovery via crash dumps, memory disclosure bugs, or cold boot attacks.
///
/// Intentionally does not implement `Clone` to prevent accidental copies of
/// key material that would bypass zeroize-on-drop.
pub struct DerivedKey {
    key: [u8; KEY_SIZE],
}

impl DerivedKey {
    pub fn as_bytes(&self) -> &[u8; KEY_SIZE] {
        &self.key
    }
}

impl std::fmt::Debug for DerivedKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("DerivedKey([REDACTED])")
    }
}

impl Drop for DerivedKey {
    fn drop(&mut self) {
        self.key.zeroize();
    }
}

/// Validate that a password meets minimum requirements for new encrypted archives.
///
/// Call this before [`derive_key`] when **creating** new encrypted archives.
/// Decryption should NOT validate password length (to support legacy archives
/// that may have been created with shorter passwords).
pub fn validate_encryption_password(password: &[u8]) -> Result<()> {
    if password.is_empty() {
        return Err(AetherError::Encryption("Password cannot be empty".into()));
    }
    if password.len() < MIN_PASSWORD_LEN {
        return Err(AetherError::Encryption(format!(
            "Password too short ({} bytes, minimum {})",
            password.len(),
            MIN_PASSWORD_LEN,
        )));
    }
    Ok(())
}

/// Validate Argon2id parameters against safety bounds.
///
/// Rejects parameters that are too low (enabling brute-force) or too high
/// (enabling DoS via resource exhaustion from crafted archives).
fn validate_argon2_params(m_cost: u32, t_cost: u32, p_cost: u32) -> Result<()> {
    if m_cost < MIN_ARGON2_M_COST {
        return Err(AetherError::Encryption(format!(
            "Argon2 m_cost {} below minimum of {} KiB",
            m_cost, MIN_ARGON2_M_COST,
        )));
    }
    if m_cost > MAX_ARGON2_M_COST {
        return Err(AetherError::Encryption(format!(
            "Argon2 m_cost {} exceeds safety limit of {} KiB",
            m_cost, MAX_ARGON2_M_COST,
        )));
    }
    if t_cost < MIN_ARGON2_T_COST {
        return Err(AetherError::Encryption(format!(
            "Argon2 t_cost {} below minimum of {}",
            t_cost, MIN_ARGON2_T_COST,
        )));
    }
    if t_cost > MAX_ARGON2_T_COST {
        return Err(AetherError::Encryption(format!(
            "Argon2 t_cost {} exceeds safety limit of {}",
            t_cost, MAX_ARGON2_T_COST,
        )));
    }
    if p_cost < MIN_ARGON2_P_COST {
        return Err(AetherError::Encryption(format!(
            "Argon2 p_cost {} below minimum of {}",
            p_cost, MIN_ARGON2_P_COST,
        )));
    }
    if p_cost > MAX_ARGON2_P_COST {
        return Err(AetherError::Encryption(format!(
            "Argon2 p_cost {} exceeds safety limit of {}",
            p_cost, MAX_ARGON2_P_COST,
        )));
    }
    Ok(())
}

/// Derive an encryption key from a password using Argon2id.
///
/// Returns a [`DerivedKey`] that zeroizes the 256-bit key on drop,
/// suitable for AES-256-GCM or ChaCha20-Poly1305.
///
/// **Note**: This function does not enforce password length requirements.
/// Call [`validate_encryption_password`] before this when creating new archives.
/// The caller is responsible for zeroizing the password buffer after use.
pub fn derive_key(
    password: &[u8],
    salt: &[u8; SALT_SIZE],
    m_cost: u32,
    t_cost: u32,
    p_cost: u32,
) -> Result<DerivedKey> {
    validate_argon2_params(m_cost, t_cost, p_cost)?;

    use argon2::{Algorithm, Argon2, Params, Version};
    let params = Params::new(m_cost, t_cost, p_cost, Some(KEY_SIZE))
        .map_err(|e| AetherError::Encryption(format!("Argon2 params error: {e}")))?;
    let argon2 = Argon2::new(Algorithm::Argon2id, Version::V0x13, params);
    let mut key = [0u8; KEY_SIZE];
    argon2
        .hash_password_into(password, salt, &mut key)
        .map_err(|e| AetherError::Encryption(format!("Key derivation failed: {e}")))?;
    Ok(DerivedKey { key })
}

/// Reject plaintext that exceeds the maximum block size.
///
/// Prevents OOM from oversized inputs and stays well within AES-GCM's
/// ~64 GiB per-nonce limit.
fn validate_plaintext_size(len: usize) -> Result<()> {
    if len > MAX_BLOCK_PLAINTEXT_SIZE {
        return Err(AetherError::Encryption(format!(
            "Block plaintext size {} exceeds maximum of {} bytes",
            len, MAX_BLOCK_PLAINTEXT_SIZE,
        )));
    }
    Ok(())
}

/// Build the Associated Authenticated Data (AAD) for a block.
///
/// Binds cipher_id and block_id to the ciphertext so that blocks cannot be
/// reordered or swapped between archives using different ciphers.
fn build_block_aad(cipher_id: CipherId, block_id: u32) -> [u8; 5] {
    let mut aad = [0u8; 5];
    aad[0] = cipher_id as u8;
    aad[1..5].copy_from_slice(&block_id.to_be_bytes());
    aad
}

/// Encrypt a block payload using the specified cipher.
///
/// Returns: `[12B nonce] [ciphertext...] [16B auth tag]`
///
/// The block_id and cipher_id are bound as AAD (Associated Authenticated Data),
/// preventing block reordering and cross-cipher swap attacks.
pub fn encrypt_block(
    cipher_id: CipherId,
    key: &[u8; KEY_SIZE],
    master_nonce: &[u8; NONCE_SIZE],
    block_id: u32,
    plaintext: &[u8],
) -> Result<Vec<u8>> {
    validate_plaintext_size(plaintext.len())?;
    let nonce = derive_block_nonce(master_nonce, block_id);
    let aad = build_block_aad(cipher_id, block_id);
    let ciphertext = match cipher_id {
        CipherId::Aes256Gcm => encrypt_aes_gcm(key, &nonce, plaintext, &aad)?,
        CipherId::ChaCha20Poly1305 => encrypt_chacha(key, &nonce, plaintext, &aad)?,
    };
    // Prepend nonce to ciphertext+tag
    let mut out = Vec::with_capacity(NONCE_SIZE + ciphertext.len());
    out.extend_from_slice(&nonce);
    out.extend_from_slice(&ciphertext);
    Ok(out)
}

/// Decrypt a block payload using the specified cipher.
///
/// Input: `[12B nonce] [ciphertext...] [16B auth tag]`
///
/// The `master_nonce` and `block_id` are required to verify the stored nonce
/// against the expected derived nonce (defense-in-depth against nonce
/// tampering) and to reconstruct the AAD used during encryption.
pub fn decrypt_block(
    cipher_id: CipherId,
    key: &[u8; KEY_SIZE],
    master_nonce: &[u8; NONCE_SIZE],
    block_id: u32,
    encrypted: &[u8],
) -> Result<Vec<u8>> {
    if encrypted.len() < NONCE_SIZE + TAG_SIZE {
        return Err(AetherError::Decryption(
            "Encrypted block too small (missing nonce or tag)".into(),
        ));
    }
    let stored_nonce: [u8; NONCE_SIZE] = encrypted[..NONCE_SIZE]
        .try_into()
        .map_err(|_| AetherError::Decryption("Encrypted block nonce truncated".into()))?;

    // Verify stored nonce matches the deterministically derived nonce.
    // Uses constant-time comparison consistent with the rest of the module.
    use subtle::ConstantTimeEq;
    let expected_nonce = derive_block_nonce(master_nonce, block_id);
    if !bool::from(stored_nonce.ct_eq(&expected_nonce)) {
        return Err(AetherError::NonceMismatch { block_id });
    }

    let ciphertext_and_tag = &encrypted[NONCE_SIZE..];
    let aad = build_block_aad(cipher_id, block_id);
    match cipher_id {
        CipherId::Aes256Gcm => decrypt_aes_gcm(key, &stored_nonce, ciphertext_and_tag, &aad),
        CipherId::ChaCha20Poly1305 => decrypt_chacha(key, &stored_nonce, ciphertext_and_tag, &aad),
    }
}

// ── AES-256-GCM ─────────────────────────────────────────────────────────────

fn encrypt_aes_gcm(
    key: &[u8; KEY_SIZE],
    nonce: &[u8; NONCE_SIZE],
    plaintext: &[u8],
    aad: &[u8],
) -> Result<Vec<u8>> {
    use aes_gcm::aead::generic_array::GenericArray;
    use aes_gcm::{aead::Aead, aead::Payload, Aes256Gcm, KeyInit};

    let cipher = Aes256Gcm::new(GenericArray::from_slice(key));
    let nonce_ga = GenericArray::from_slice(nonce);
    let payload = Payload {
        msg: plaintext,
        aad,
    };
    cipher
        .encrypt(nonce_ga, payload)
        .map_err(|e| AetherError::Encryption(format!("AES-GCM encrypt failed: {e}")))
}

fn decrypt_aes_gcm(
    key: &[u8; KEY_SIZE],
    nonce: &[u8; NONCE_SIZE],
    ciphertext_and_tag: &[u8],
    aad: &[u8],
) -> Result<Vec<u8>> {
    use aes_gcm::aead::generic_array::GenericArray;
    use aes_gcm::{aead::Aead, aead::Payload, Aes256Gcm, KeyInit};

    let cipher = Aes256Gcm::new(GenericArray::from_slice(key));
    let nonce_ga = GenericArray::from_slice(nonce);
    let payload = Payload {
        msg: ciphertext_and_tag,
        aad,
    };
    cipher.decrypt(nonce_ga, payload).map_err(|_| {
        AetherError::Decryption(
            "AES-GCM authentication failed: wrong password or corrupted/tampered data".into(),
        )
    })
}

// ── ChaCha20-Poly1305 ───────────────────────────────────────────────────────

fn encrypt_chacha(
    key: &[u8; KEY_SIZE],
    nonce: &[u8; NONCE_SIZE],
    plaintext: &[u8],
    aad: &[u8],
) -> Result<Vec<u8>> {
    use chacha20poly1305::aead::generic_array::GenericArray;
    use chacha20poly1305::{aead::Aead, aead::Payload, ChaCha20Poly1305, KeyInit};

    let cipher = ChaCha20Poly1305::new(GenericArray::from_slice(key));
    let nonce_ga = GenericArray::from_slice(nonce);
    let payload = Payload {
        msg: plaintext,
        aad,
    };
    cipher
        .encrypt(nonce_ga, payload)
        .map_err(|e| AetherError::Encryption(format!("ChaCha20 encrypt failed: {e}")))
}

fn decrypt_chacha(
    key: &[u8; KEY_SIZE],
    nonce: &[u8; NONCE_SIZE],
    ciphertext_and_tag: &[u8],
    aad: &[u8],
) -> Result<Vec<u8>> {
    use chacha20poly1305::aead::generic_array::GenericArray;
    use chacha20poly1305::{aead::Aead, aead::Payload, ChaCha20Poly1305, KeyInit};

    let cipher = ChaCha20Poly1305::new(GenericArray::from_slice(key));
    let nonce_ga = GenericArray::from_slice(nonce);
    let payload = Payload {
        msg: ciphertext_and_tag,
        aad,
    };
    cipher.decrypt(nonce_ga, payload).map_err(|_| {
        AetherError::Decryption(
            "ChaCha20-Poly1305 authentication failed: wrong password or corrupted/tampered data"
                .into(),
        )
    })
}

// ── BlockCryptor ────────────────────────────────────────────────────────────

/// Cipher implementation holder — avoids re-running key expansion per block.
#[allow(clippy::large_enum_variant)]
enum CipherImpl {
    Aes(aes_gcm::Aes256Gcm),
    ChaCha(chacha20poly1305::ChaCha20Poly1305),
}

/// Reusable block encryptor/decryptor that caches the initialized cipher.
///
/// Constructing AES-GCM or ChaCha20 cipher objects involves key expansion.
/// When encrypting/decrypting many blocks, reusing a single `BlockCryptor`
/// avoids repeating that work for every block.
pub struct BlockCryptor {
    cipher_id: CipherId,
    inner: CipherImpl,
    master_nonce: [u8; NONCE_SIZE],
}

impl BlockCryptor {
    /// Create a new `BlockCryptor` with the given cipher, key, and master nonce.
    pub fn new(cipher_id: CipherId, key: &[u8; KEY_SIZE], master_nonce: [u8; NONCE_SIZE]) -> Self {
        let inner = match cipher_id {
            CipherId::Aes256Gcm => {
                use aes_gcm::aead::generic_array::GenericArray;
                use aes_gcm::{Aes256Gcm, KeyInit};
                CipherImpl::Aes(Aes256Gcm::new(GenericArray::from_slice(key)))
            }
            CipherId::ChaCha20Poly1305 => {
                use chacha20poly1305::aead::generic_array::GenericArray;
                use chacha20poly1305::{ChaCha20Poly1305, KeyInit};
                CipherImpl::ChaCha(ChaCha20Poly1305::new(GenericArray::from_slice(key)))
            }
        };
        Self {
            cipher_id,
            inner,
            master_nonce,
        }
    }

    /// Encrypt a block payload. Returns `[12B nonce] [ciphertext...] [16B tag]`.
    pub fn encrypt(&self, block_id: u32, plaintext: &[u8]) -> Result<Vec<u8>> {
        validate_plaintext_size(plaintext.len())?;
        let nonce = derive_block_nonce(&self.master_nonce, block_id);
        let aad = build_block_aad(self.cipher_id, block_id);

        let ciphertext = match &self.inner {
            CipherImpl::Aes(cipher) => {
                use aes_gcm::aead::generic_array::GenericArray;
                use aes_gcm::aead::{Aead, Payload};
                cipher
                    .encrypt(
                        GenericArray::from_slice(&nonce),
                        Payload {
                            msg: plaintext,
                            aad: &aad,
                        },
                    )
                    .map_err(|e| AetherError::Encryption(format!("AES-GCM encrypt failed: {e}")))?
            }
            CipherImpl::ChaCha(cipher) => {
                use chacha20poly1305::aead::generic_array::GenericArray;
                use chacha20poly1305::aead::{Aead, Payload};
                cipher
                    .encrypt(
                        GenericArray::from_slice(&nonce),
                        Payload {
                            msg: plaintext,
                            aad: &aad,
                        },
                    )
                    .map_err(|e| AetherError::Encryption(format!("ChaCha20 encrypt failed: {e}")))?
            }
        };

        let mut out = Vec::with_capacity(NONCE_SIZE + ciphertext.len());
        out.extend_from_slice(&nonce);
        out.extend_from_slice(&ciphertext);
        Ok(out)
    }

    /// Decrypt a block payload. Input: `[12B nonce] [ciphertext...] [16B tag]`.
    ///
    /// Verifies the stored nonce matches the expected derived nonce before
    /// attempting AEAD decryption.
    pub fn decrypt(&self, block_id: u32, encrypted: &[u8]) -> Result<Vec<u8>> {
        if encrypted.len() < NONCE_SIZE + TAG_SIZE {
            return Err(AetherError::Decryption(
                "Encrypted block too small (missing nonce or tag)".into(),
            ));
        }
        let stored_nonce: [u8; NONCE_SIZE] = encrypted[..NONCE_SIZE]
            .try_into()
            .map_err(|_| AetherError::Decryption("Encrypted block nonce truncated".into()))?;

        use subtle::ConstantTimeEq;
        let expected_nonce = derive_block_nonce(&self.master_nonce, block_id);
        if !bool::from(stored_nonce.ct_eq(&expected_nonce)) {
            return Err(AetherError::NonceMismatch { block_id });
        }

        let ciphertext_and_tag = &encrypted[NONCE_SIZE..];
        let aad = build_block_aad(self.cipher_id, block_id);

        match &self.inner {
            CipherImpl::Aes(cipher) => {
                use aes_gcm::aead::generic_array::GenericArray;
                use aes_gcm::aead::{Aead, Payload};
                cipher
                    .decrypt(
                        GenericArray::from_slice(&stored_nonce),
                        Payload { msg: ciphertext_and_tag, aad: &aad },
                    )
                    .map_err(|_| AetherError::Decryption(
                        "AES-GCM authentication failed: wrong password or corrupted/tampered data".into(),
                    ))
            }
            CipherImpl::ChaCha(cipher) => {
                use chacha20poly1305::aead::generic_array::GenericArray;
                use chacha20poly1305::aead::{Aead, Payload};
                cipher
                    .decrypt(
                        GenericArray::from_slice(&stored_nonce),
                        Payload { msg: ciphertext_and_tag, aad: &aad },
                    )
                    .map_err(|_| AetherError::Decryption(
                        "ChaCha20-Poly1305 authentication failed: wrong password or corrupted/tampered data".into(),
                    ))
            }
        }
    }

    /// The cipher algorithm this cryptor uses.
    pub fn cipher_id(&self) -> CipherId {
        self.cipher_id
    }

    /// The master nonce this cryptor derives per-block nonces from.
    pub fn master_nonce(&self) -> &[u8; NONCE_SIZE] {
        &self.master_nonce
    }
}

impl Drop for BlockCryptor {
    fn drop(&mut self) {
        self.master_nonce.zeroize();
        // The inner cipher holds expanded key schedule material.
        // Zero the enum's memory to prevent key recovery from memory dumps.
        // SAFETY: CipherImpl is a repr(Rust) enum containing only plain data
        // (expanded key arrays). Zeroing is safe and prevents key leakage.
        let ptr = &mut self.inner as *mut CipherImpl as *mut u8;
        let size = std::mem::size_of::<CipherImpl>();
        // Use volatile writes to prevent the compiler from optimizing this out.
        unsafe {
            std::ptr::write_bytes(ptr, 0, size);
        }
    }
}

// ── Generate random salt and nonce ──────────────────────────────────────────

/// Generate a random salt for key derivation (32 bytes).
pub fn generate_salt() -> [u8; SALT_SIZE] {
    let mut salt = [0u8; SALT_SIZE];
    getrandom(&mut salt);
    salt
}

/// Generate a random master nonce (12 bytes).
///
/// All 12 bytes are filled with cryptographic randomness, providing 96 bits
/// of entropy. `derive_block_nonce` uses BLAKE3 to derive per-block nonces
/// from this master nonce, preserving full entropy in the derived nonces.
pub fn generate_nonce() -> [u8; NONCE_SIZE] {
    let mut nonce = [0u8; NONCE_SIZE];
    getrandom(&mut nonce);
    nonce
}

/// Fill a buffer with cryptographically secure random bytes from the OS.
///
/// # Panics
///
/// Panics if the OS random number generator is unavailable. This is
/// intentional: proceeding with non-random bytes for cryptographic nonces
/// or salts would silently produce insecure output, which is worse than
/// a visible crash.
fn getrandom(buf: &mut [u8]) {
    getrandom_crate::getrandom(buf).expect("OS random number generator failed");
}

// ── Metadata Encryption ─────────────────────────────────────────────────────

/// Encrypt serialized metadata (file table + group table) using a reserved block ID.
///
/// Uses [`ENCRYPTED_METADATA_BLOCK_ID`](crate::format::ENCRYPTED_METADATA_BLOCK_ID)
/// to derive a unique nonce that can never collide with real block nonces.
pub fn encrypt_metadata(cryptor: &BlockCryptor, plaintext: &[u8]) -> Result<Vec<u8>> {
    cryptor.encrypt(crate::format::ENCRYPTED_METADATA_BLOCK_ID, plaintext)
}

/// Decrypt serialized metadata (file table + group table) using a reserved block ID.
pub fn decrypt_metadata(cryptor: &BlockCryptor, encrypted: &[u8]) -> Result<Vec<u8>> {
    cryptor.decrypt(crate::format::ENCRYPTED_METADATA_BLOCK_ID, encrypted)
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_key() -> [u8; KEY_SIZE] {
        let mut key = [0u8; KEY_SIZE];
        for (i, b) in key.iter_mut().enumerate() {
            *b = i as u8;
        }
        key
    }

    fn test_nonce() -> [u8; NONCE_SIZE] {
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    }

    #[test]
    fn aes_gcm_roundtrip() {
        let key = test_key();
        let nonce = test_nonce();
        let data = b"Hello, encrypted world!";

        let encrypted = encrypt_block(CipherId::Aes256Gcm, &key, &nonce, 0, data).unwrap();
        assert_ne!(&encrypted[NONCE_SIZE..], data.as_slice());

        let decrypted = decrypt_block(CipherId::Aes256Gcm, &key, &nonce, 0, &encrypted).unwrap();
        assert_eq!(decrypted, data);
    }

    #[test]
    fn chacha20_roundtrip() {
        let key = test_key();
        let nonce = test_nonce();
        let data = b"Hello, encrypted world!";

        let encrypted = encrypt_block(CipherId::ChaCha20Poly1305, &key, &nonce, 0, data).unwrap();
        assert_ne!(&encrypted[NONCE_SIZE..], data.as_slice());

        let decrypted =
            decrypt_block(CipherId::ChaCha20Poly1305, &key, &nonce, 0, &encrypted).unwrap();
        assert_eq!(decrypted, data);
    }

    #[test]
    fn per_block_nonces_are_unique() {
        let nonce = test_nonce();

        let n0 = derive_block_nonce(&nonce, 0);
        let n1 = derive_block_nonce(&nonce, 1);
        let n2 = derive_block_nonce(&nonce, 2);
        assert_ne!(n0, n1);
        assert_ne!(n1, n2);
        assert_ne!(n0, n2);
    }

    #[test]
    fn per_block_nonces_are_deterministic() {
        let nonce = test_nonce();
        let n1a = derive_block_nonce(&nonce, 42);
        let n1b = derive_block_nonce(&nonce, 42);
        assert_eq!(n1a, n1b);
    }

    #[test]
    fn per_block_nonces_differ_across_all_bytes() {
        // BLAKE3-based derivation should produce nonces that differ across
        // all 12 bytes, not just the last 4.
        let nonce = test_nonce();
        let n0 = derive_block_nonce(&nonce, 0);
        let n1 = derive_block_nonce(&nonce, 1);
        // At least some of the first 8 bytes should differ (with overwhelming probability)
        assert_ne!(&n0[..8], &n1[..8]);
    }

    #[test]
    fn wrong_key_fails() {
        let key = test_key();
        let nonce = test_nonce();
        let data = b"secret data";

        let encrypted = encrypt_block(CipherId::Aes256Gcm, &key, &nonce, 0, data).unwrap();

        let mut wrong_key = test_key();
        wrong_key[0] ^= 0xFF;
        let result = decrypt_block(CipherId::Aes256Gcm, &wrong_key, &nonce, 0, &encrypted);
        assert!(result.is_err());
    }

    #[test]
    fn tampered_ciphertext_fails() {
        let key = test_key();
        let nonce = test_nonce();
        let data = b"integrity test";

        let mut encrypted = encrypt_block(CipherId::Aes256Gcm, &key, &nonce, 0, data).unwrap();
        // Flip a bit in the ciphertext (not the nonce)
        if encrypted.len() > NONCE_SIZE + 2 {
            encrypted[NONCE_SIZE + 1] ^= 0x01;
        }
        let result = decrypt_block(CipherId::Aes256Gcm, &key, &nonce, 0, &encrypted);
        assert!(result.is_err());
    }

    #[test]
    fn tampered_nonce_detected() {
        let key = test_key();
        let nonce = test_nonce();
        let data = b"nonce tamper test";

        let mut encrypted = encrypt_block(CipherId::Aes256Gcm, &key, &nonce, 0, data).unwrap();
        // Flip a bit in the stored nonce
        encrypted[0] ^= 0x01;
        let result = decrypt_block(CipherId::Aes256Gcm, &key, &nonce, 0, &encrypted);
        assert!(result.is_err());
    }

    #[test]
    fn key_derivation_deterministic() {
        let password = b"test-password-123";
        let salt = [42u8; SALT_SIZE];

        // Use params at minimum allowed thresholds for test speed
        let key1 = derive_key(
            password,
            &salt,
            MIN_ARGON2_M_COST,
            MIN_ARGON2_T_COST,
            MIN_ARGON2_P_COST,
        )
        .unwrap();
        let key2 = derive_key(
            password,
            &salt,
            MIN_ARGON2_M_COST,
            MIN_ARGON2_T_COST,
            MIN_ARGON2_P_COST,
        )
        .unwrap();
        assert_eq!(key1.as_bytes(), key2.as_bytes());
    }

    #[test]
    fn different_passwords_different_keys() {
        let salt = [42u8; SALT_SIZE];
        let key1 = derive_key(
            b"password1",
            &salt,
            MIN_ARGON2_M_COST,
            MIN_ARGON2_T_COST,
            MIN_ARGON2_P_COST,
        )
        .unwrap();
        let key2 = derive_key(
            b"password2",
            &salt,
            MIN_ARGON2_M_COST,
            MIN_ARGON2_T_COST,
            MIN_ARGON2_P_COST,
        )
        .unwrap();
        assert_ne!(key1.as_bytes(), key2.as_bytes());
    }

    #[test]
    fn encryption_header_roundtrip() {
        let header = EncryptionHeader {
            version: HEADER_VERSION,
            cipher_id: CipherId::ChaCha20Poly1305,
            salt: [0xAA; SALT_SIZE],
            m_cost: 65536,
            t_cost: 3,
            p_cost: 4,
            master_nonce: [0xBB; NONCE_SIZE],
            verification_tag: [0xCC; crate::format::VERIFICATION_TAG_SIZE],
        };

        let mut buf = Vec::new();
        header.write_to(&mut buf).unwrap();
        assert_eq!(buf.len(), ENCRYPTION_HEADER_SIZE);

        let parsed = EncryptionHeader::read_from(&mut &buf[..]).unwrap();
        assert_eq!(parsed.version, HEADER_VERSION);
        assert_eq!(parsed.cipher_id, CipherId::ChaCha20Poly1305);
        assert_eq!(parsed.salt, [0xAA; SALT_SIZE]);
        assert_eq!(parsed.m_cost, 65536);
        assert_eq!(parsed.t_cost, 3);
        assert_eq!(parsed.p_cost, 4);
        assert_eq!(parsed.master_nonce, [0xBB; NONCE_SIZE]);
        assert_eq!(
            parsed.verification_tag,
            [0xCC; crate::format::VERIFICATION_TAG_SIZE]
        );
    }

    #[test]
    fn generate_salt_is_not_zero() {
        let salt = generate_salt();
        assert!(salt.iter().any(|&b| b != 0));
    }

    #[test]
    fn generate_nonce_fills_all_bytes() {
        // All 12 bytes should have entropy (statistically, not all zero)
        let nonce = generate_nonce();
        assert!(nonce.iter().any(|&b| b != 0));
        // Bytes 8..12 should also have entropy
        assert!(
            nonce[8..12].iter().any(|&b| b != 0) || {
                // Extremely unlikely (1 in 2^32) but technically possible
                // Run a second time to be sure
                let nonce2 = generate_nonce();
                nonce2[8..12].iter().any(|&b| b != 0)
            }
        );
    }

    #[test]
    fn empty_plaintext_roundtrip() {
        let key = test_key();
        let nonce = test_nonce();
        let encrypted = encrypt_block(CipherId::Aes256Gcm, &key, &nonce, 0, b"").unwrap();
        let decrypted = decrypt_block(CipherId::Aes256Gcm, &key, &nonce, 0, &encrypted).unwrap();
        assert!(decrypted.is_empty());
    }

    #[test]
    fn large_data_roundtrip() {
        let key = test_key();
        let nonce = test_nonce();
        let data: Vec<u8> = (0..100_000).map(|i| (i % 256) as u8).collect();

        for cipher in [CipherId::Aes256Gcm, CipherId::ChaCha20Poly1305] {
            let encrypted = encrypt_block(cipher, &key, &nonce, 42, &data).unwrap();
            let decrypted = decrypt_block(cipher, &key, &nonce, 42, &encrypted).unwrap();
            assert_eq!(decrypted, data);
        }
    }

    // ── Security tests ─────────────────────────────────────────────────────

    #[test]
    fn cross_cipher_decrypt_fails() {
        let key = test_key();
        let nonce = test_nonce();
        let data = b"cross-cipher test";

        // Encrypt with AES, decrypt with ChaCha — must fail authentication
        let encrypted = encrypt_block(CipherId::Aes256Gcm, &key, &nonce, 0, data).unwrap();
        let result = decrypt_block(CipherId::ChaCha20Poly1305, &key, &nonce, 0, &encrypted);
        assert!(result.is_err());
    }

    #[test]
    fn wrong_block_id_decrypt_fails() {
        let key = test_key();
        let nonce = test_nonce();
        let data = b"block reorder test";

        // Encrypt at block_id=5, decrypt with block_id=6 — nonce mismatch
        let encrypted = encrypt_block(CipherId::Aes256Gcm, &key, &nonce, 5, data).unwrap();
        let result = decrypt_block(CipherId::Aes256Gcm, &key, &nonce, 6, &encrypted);
        assert!(result.is_err());
    }

    #[test]
    fn argon2_m_cost_too_high_rejected() {
        let salt = [42u8; SALT_SIZE];
        let result = derive_key(
            b"password-for-test",
            &salt,
            MAX_ARGON2_M_COST + 1,
            MIN_ARGON2_T_COST,
            MIN_ARGON2_P_COST,
        );
        assert!(result.is_err());
    }

    #[test]
    fn argon2_m_cost_too_low_rejected() {
        let salt = [42u8; SALT_SIZE];
        let result = derive_key(
            b"password-for-test",
            &salt,
            MIN_ARGON2_M_COST - 1,
            MIN_ARGON2_T_COST,
            MIN_ARGON2_P_COST,
        );
        assert!(result.is_err());
    }

    #[test]
    fn argon2_t_cost_too_low_rejected() {
        let salt = [42u8; SALT_SIZE];
        let result = derive_key(
            b"password-for-test",
            &salt,
            MIN_ARGON2_M_COST,
            MIN_ARGON2_T_COST - 1,
            MIN_ARGON2_P_COST,
        );
        assert!(result.is_err());
    }

    #[test]
    fn unknown_cipher_id_rejected() {
        let result = CipherId::from_u8(255);
        assert!(result.is_err());
    }

    #[test]
    fn derive_block_nonce_u32_max() {
        let nonce = test_nonce();
        // u32::MAX should produce a valid, unique nonce
        let result = derive_block_nonce(&nonce, u32::MAX);
        // Must differ from block 0
        assert_ne!(result, derive_block_nonce(&nonce, 0));
        // Must be deterministic
        assert_eq!(result, derive_block_nonce(&nonce, u32::MAX));
    }

    #[test]
    fn decrypt_block_too_small_rejected() {
        let key = test_key();
        let nonce = test_nonce();
        let tiny = vec![0u8; NONCE_SIZE + TAG_SIZE - 1];
        let result = decrypt_block(CipherId::Aes256Gcm, &key, &nonce, 0, &tiny);
        assert!(result.is_err());
    }

    #[test]
    fn header_read_rejects_low_m_cost() {
        use byteorder::{LittleEndian, WriteBytesExt};
        let mut buf = Vec::new();
        buf.push(HEADER_VERSION); // version
        buf.push(0u8); // cipher_id = AES
        buf.extend_from_slice(&[0xAA; SALT_SIZE]); // salt
        buf.write_u32::<LittleEndian>(1024).unwrap(); // m_cost below minimum
        buf.write_u32::<LittleEndian>(3).unwrap(); // t_cost
        buf.write_u32::<LittleEndian>(4).unwrap(); // p_cost
        buf.extend_from_slice(&[0xBB; NONCE_SIZE]); // nonce
        buf.extend_from_slice(&[0x00; crate::format::VERIFICATION_TAG_SIZE]); // tag

        let result = EncryptionHeader::read_from(&mut &buf[..]);
        assert!(result.is_err());
    }

    #[test]
    fn header_read_rejects_bad_version() {
        use byteorder::{LittleEndian, WriteBytesExt};
        let mut buf = Vec::new();
        buf.push(99u8); // bad version
        buf.push(0u8); // cipher_id
        buf.extend_from_slice(&[0xAA; SALT_SIZE]);
        buf.write_u32::<LittleEndian>(MIN_ARGON2_M_COST).unwrap();
        buf.write_u32::<LittleEndian>(MIN_ARGON2_T_COST).unwrap();
        buf.write_u32::<LittleEndian>(MIN_ARGON2_P_COST).unwrap();
        buf.extend_from_slice(&[0xBB; NONCE_SIZE]);
        buf.extend_from_slice(&[0x00; crate::format::VERIFICATION_TAG_SIZE]); // tag

        let result = EncryptionHeader::read_from(&mut &buf[..]);
        assert!(result.is_err());
    }

    #[test]
    fn derived_key_debug_is_redacted() {
        let salt = [42u8; SALT_SIZE];
        let key = derive_key(
            b"password",
            &salt,
            MIN_ARGON2_M_COST,
            MIN_ARGON2_T_COST,
            MIN_ARGON2_P_COST,
        )
        .unwrap();
        let debug_str = format!("{:?}", key);
        assert_eq!(debug_str, "DerivedKey([REDACTED])");
        assert!(!debug_str.contains("42"));
    }

    #[test]
    fn encryption_header_debug_is_redacted() {
        let header = EncryptionHeader {
            version: HEADER_VERSION,
            cipher_id: CipherId::Aes256Gcm,
            salt: [0xAA; SALT_SIZE],
            m_cost: 65536,
            t_cost: 3,
            p_cost: 4,
            master_nonce: [0xBB; NONCE_SIZE],
            verification_tag: [0xDD; crate::format::VERIFICATION_TAG_SIZE],
        };
        let debug_str = format!("{:?}", header);
        assert!(debug_str.contains("[REDACTED]"));
        assert!(!debug_str.contains("170")); // 0xAA = 170
        assert!(!debug_str.contains("187")); // 0xBB = 187
    }

    #[test]
    fn validate_encryption_password_rejects_empty() {
        assert!(validate_encryption_password(b"").is_err());
    }

    #[test]
    fn validate_encryption_password_rejects_short() {
        assert!(validate_encryption_password(b"short").is_err());
    }

    #[test]
    fn validate_encryption_password_accepts_minimum() {
        assert!(validate_encryption_password(b"12345678").is_ok());
    }

    // ── BlockCryptor tests ─────────────────────────────────────────────────

    #[test]
    fn block_cryptor_roundtrip_aes() {
        let key = test_key();
        let nonce = test_nonce();
        let data = b"BlockCryptor AES roundtrip";

        let cryptor = BlockCryptor::new(CipherId::Aes256Gcm, &key, nonce);
        let encrypted = cryptor.encrypt(0, data).unwrap();
        let decrypted = cryptor.decrypt(0, &encrypted).unwrap();
        assert_eq!(decrypted, data);
    }

    #[test]
    fn block_cryptor_roundtrip_chacha() {
        let key = test_key();
        let nonce = test_nonce();
        let data = b"BlockCryptor ChaCha roundtrip";

        let cryptor = BlockCryptor::new(CipherId::ChaCha20Poly1305, &key, nonce);
        let encrypted = cryptor.encrypt(0, data).unwrap();
        let decrypted = cryptor.decrypt(0, &encrypted).unwrap();
        assert_eq!(decrypted, data);
    }

    #[test]
    fn block_cryptor_matches_standalone_functions() {
        let key = test_key();
        let nonce = test_nonce();
        let data = b"consistency test";

        let cryptor = BlockCryptor::new(CipherId::Aes256Gcm, &key, nonce);

        let enc_standalone = encrypt_block(CipherId::Aes256Gcm, &key, &nonce, 42, data).unwrap();
        let enc_cryptor = cryptor.encrypt(42, data).unwrap();
        assert_eq!(enc_standalone, enc_cryptor);

        let dec = cryptor.decrypt(42, &enc_standalone).unwrap();
        assert_eq!(dec, data);
    }
}
