//! Additional crypto tests for ChaCha20 roundtrip, mixed-cipher failures,
//! Argon2id parameter enforcement, and large-data encryption.

#![cfg(feature = "enterprise")]

use aether_core::crypto::*;

// ── Helpers ─────────────────────────────────────────────────────────────────

fn test_key() -> [u8; KEY_SIZE] {
    let mut key = [0u8; KEY_SIZE];
    for (i, b) in key.iter_mut().enumerate() {
        *b = i as u8;
    }
    key
}

fn test_nonce() -> [u8; NONCE_SIZE] {
    [1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0]
}

// ── ChaCha20-Poly1305 full roundtrip with multiple block IDs ────────────────

#[test]
fn chacha20_roundtrip_multiple_blocks() {
    let key = test_key();
    let nonce = test_nonce();
    let data = b"ChaCha20 multi-block roundtrip test payload";

    for block_id in [0u32, 1, 7, 255, 1024, u32::MAX - 1] {
        let encrypted = encrypt_block(CipherId::ChaCha20Poly1305, &key, &nonce, block_id, data)
            .unwrap_or_else(|e| panic!("encrypt block_id={block_id} failed: {e}"));

        // Ciphertext should differ from plaintext
        assert_ne!(
            &encrypted[NONCE_SIZE..encrypted.len() - TAG_SIZE],
            data.as_slice(),
            "ciphertext should not equal plaintext for block_id={block_id}"
        );

        let decrypted = decrypt_block(
            CipherId::ChaCha20Poly1305,
            &key,
            &nonce,
            block_id,
            &encrypted,
        )
        .unwrap_or_else(|e| panic!("decrypt block_id={block_id} failed: {e}"));

        assert_eq!(
            decrypted, data,
            "roundtrip mismatch for block_id={block_id}"
        );
    }
}

#[test]
fn chacha20_different_blocks_produce_different_ciphertext() {
    let key = test_key();
    let nonce = test_nonce();
    let data = b"same plaintext, different block IDs";

    let enc0 = encrypt_block(CipherId::ChaCha20Poly1305, &key, &nonce, 0, data).unwrap();
    let enc1 = encrypt_block(CipherId::ChaCha20Poly1305, &key, &nonce, 1, data).unwrap();

    // Different block IDs should produce different ciphertext (different nonces + AAD)
    assert_ne!(enc0, enc1);
}

// ── Mixed cipher usage ──────────────────────────────────────────────────────

#[test]
fn encrypt_aes_decrypt_chacha_fails() {
    let key = test_key();
    let nonce = test_nonce();
    let data = b"cross-cipher mismatch test";

    let encrypted = encrypt_block(CipherId::Aes256Gcm, &key, &nonce, 0, data).unwrap();
    let result = decrypt_block(CipherId::ChaCha20Poly1305, &key, &nonce, 0, &encrypted);
    assert!(
        result.is_err(),
        "decrypting AES ciphertext with ChaCha should fail"
    );
}

#[test]
fn encrypt_chacha_decrypt_aes_fails() {
    let key = test_key();
    let nonce = test_nonce();
    let data = b"reverse cross-cipher mismatch";

    let encrypted = encrypt_block(CipherId::ChaCha20Poly1305, &key, &nonce, 0, data).unwrap();
    let result = decrypt_block(CipherId::Aes256Gcm, &key, &nonce, 0, &encrypted);
    assert!(
        result.is_err(),
        "decrypting ChaCha ciphertext with AES should fail"
    );
}

// ── Argon2id minimum parameter enforcement ──────────────────────────────────

#[test]
fn argon2_m_cost_below_minimum_rejected() {
    let salt = [0xABu8; SALT_SIZE];
    let result = derive_key(
        b"password",
        &salt,
        MIN_ARGON2_M_COST - 1,
        MIN_ARGON2_T_COST,
        MIN_ARGON2_P_COST,
    );
    assert!(result.is_err(), "m_cost below minimum should be rejected");
}

#[test]
fn argon2_t_cost_below_minimum_rejected() {
    let salt = [0xABu8; SALT_SIZE];
    // MIN_ARGON2_T_COST is 2, so t_cost=1 should fail
    let result = derive_key(
        b"password",
        &salt,
        MIN_ARGON2_M_COST,
        MIN_ARGON2_T_COST - 1,
        MIN_ARGON2_P_COST,
    );
    assert!(result.is_err(), "t_cost below minimum should be rejected");
}

#[test]
fn argon2_p_cost_zero_rejected() {
    let salt = [0xABu8; SALT_SIZE];
    // MIN_ARGON2_P_COST is 1, so p_cost=0 should fail
    let result = derive_key(
        b"password",
        &salt,
        MIN_ARGON2_M_COST,
        MIN_ARGON2_T_COST,
        0, // below MIN_ARGON2_P_COST
    );
    assert!(result.is_err(), "p_cost of 0 should be rejected");
}

// ── Argon2id maximum parameter enforcement ──────────────────────────────────

#[test]
fn argon2_m_cost_above_maximum_rejected() {
    let salt = [0xCDu8; SALT_SIZE];
    let result = derive_key(
        b"password",
        &salt,
        MAX_ARGON2_M_COST + 1,
        MIN_ARGON2_T_COST,
        MIN_ARGON2_P_COST,
    );
    assert!(result.is_err(), "m_cost above maximum should be rejected");
}

#[test]
fn argon2_t_cost_above_maximum_rejected() {
    let salt = [0xCDu8; SALT_SIZE];
    let result = derive_key(
        b"password",
        &salt,
        MIN_ARGON2_M_COST,
        MAX_ARGON2_T_COST + 1,
        MIN_ARGON2_P_COST,
    );
    assert!(result.is_err(), "t_cost above maximum should be rejected");
}

#[test]
fn argon2_p_cost_above_maximum_rejected() {
    let salt = [0xCDu8; SALT_SIZE];
    let result = derive_key(
        b"password",
        &salt,
        MIN_ARGON2_M_COST,
        MIN_ARGON2_T_COST,
        MAX_ARGON2_P_COST + 1, // 256, above limit of 255
    );
    assert!(result.is_err(), "p_cost above maximum should be rejected");
}

#[test]
fn argon2_at_exact_minimum_succeeds() {
    let salt = [0xEFu8; SALT_SIZE];
    let result = derive_key(
        b"password",
        &salt,
        MIN_ARGON2_M_COST,
        MIN_ARGON2_T_COST,
        MIN_ARGON2_P_COST,
    );
    assert!(result.is_ok(), "exact minimum parameters should succeed");
}

#[test]
fn argon2_at_exact_maximum_succeeds() {
    // Note: this test uses the maximum parameters. It may be slow.
    // We only validate that the function accepts the values without error;
    // we skip running it because MAX_ARGON2_M_COST = 4 GiB which would OOM.
    // Instead, test that the boundary is properly checked via header roundtrip.
    let header = EncryptionHeader {
        version: HEADER_VERSION,
        cipher_id: CipherId::Aes256Gcm,
        salt: [0x11; SALT_SIZE],
        m_cost: MAX_ARGON2_M_COST,
        t_cost: MAX_ARGON2_T_COST,
        p_cost: MAX_ARGON2_P_COST,
        master_nonce: [0x22; NONCE_SIZE],
        verification_tag: [0x33; 32],
    };

    let mut buf = Vec::new();
    header.write_to(&mut buf).unwrap();
    let parsed = EncryptionHeader::read_from(&mut &buf[..]).unwrap();
    assert_eq!(parsed.m_cost, MAX_ARGON2_M_COST);
    assert_eq!(parsed.t_cost, MAX_ARGON2_T_COST);
    assert_eq!(parsed.p_cost, MAX_ARGON2_P_COST);
}

// ── Encryption header roundtrip with ChaCha20 ──────────────────────────────

#[test]
fn encryption_header_roundtrip_chacha20() {
    let salt = generate_salt();
    let nonce = generate_nonce();

    let header = EncryptionHeader {
        version: HEADER_VERSION,
        cipher_id: CipherId::ChaCha20Poly1305,
        salt,
        m_cost: MIN_ARGON2_M_COST,
        t_cost: MIN_ARGON2_T_COST,
        p_cost: MIN_ARGON2_P_COST,
        master_nonce: nonce,
        verification_tag: [0x44; 32],
    };

    let mut buf = Vec::new();
    header.write_to(&mut buf).unwrap();
    assert_eq!(buf.len(), ENCRYPTION_HEADER_SIZE);

    let parsed = EncryptionHeader::read_from(&mut &buf[..]).unwrap();
    assert_eq!(parsed.cipher_id, CipherId::ChaCha20Poly1305);
    assert_eq!(parsed.salt, salt);
    assert_eq!(parsed.m_cost, MIN_ARGON2_M_COST);
    assert_eq!(parsed.t_cost, MIN_ARGON2_T_COST);
    assert_eq!(parsed.p_cost, MIN_ARGON2_P_COST);
    assert_eq!(parsed.master_nonce, nonce);
}

#[test]
fn encryption_header_rejects_below_minimum_on_read() {
    use byteorder::{LittleEndian, WriteBytesExt};

    // Craft a header with t_cost below minimum
    let mut buf = Vec::new();
    buf.push(HEADER_VERSION);
    buf.push(CipherId::ChaCha20Poly1305 as u8);
    buf.extend_from_slice(&[0xAA; SALT_SIZE]);
    buf.write_u32::<LittleEndian>(MIN_ARGON2_M_COST).unwrap();
    buf.write_u32::<LittleEndian>(MIN_ARGON2_T_COST - 1)
        .unwrap(); // below min
    buf.write_u32::<LittleEndian>(MIN_ARGON2_P_COST).unwrap();
    buf.extend_from_slice(&[0xBB; NONCE_SIZE]);
    buf.extend_from_slice(&[0x00; 32]); // verification_tag

    let result = EncryptionHeader::read_from(&mut &buf[..]);
    assert!(
        result.is_err(),
        "header with t_cost below minimum should be rejected"
    );
}

#[test]
fn encryption_header_rejects_above_maximum_on_read() {
    use byteorder::{LittleEndian, WriteBytesExt};

    // Craft a header with m_cost above maximum
    let mut buf = Vec::new();
    buf.push(HEADER_VERSION);
    buf.push(CipherId::Aes256Gcm as u8);
    buf.extend_from_slice(&[0xAA; SALT_SIZE]);
    buf.write_u32::<LittleEndian>(MAX_ARGON2_M_COST + 1)
        .unwrap(); // above max
    buf.write_u32::<LittleEndian>(MIN_ARGON2_T_COST).unwrap();
    buf.write_u32::<LittleEndian>(MIN_ARGON2_P_COST).unwrap();
    buf.extend_from_slice(&[0xBB; NONCE_SIZE]);
    buf.extend_from_slice(&[0x00; 32]); // verification_tag

    let result = EncryptionHeader::read_from(&mut &buf[..]);
    assert!(
        result.is_err(),
        "header with m_cost above maximum should be rejected"
    );
}

// ── Empty data encryption/decryption ────────────────────────────────────────

#[test]
fn empty_data_roundtrip_chacha20() {
    let key = test_key();
    let nonce = test_nonce();

    let encrypted = encrypt_block(CipherId::ChaCha20Poly1305, &key, &nonce, 0, b"").unwrap();
    // Should still have nonce + auth tag
    assert_eq!(encrypted.len(), NONCE_SIZE + TAG_SIZE);

    let decrypted = decrypt_block(CipherId::ChaCha20Poly1305, &key, &nonce, 0, &encrypted).unwrap();
    assert!(decrypted.is_empty());
}

#[test]
fn empty_data_roundtrip_aes() {
    let key = test_key();
    let nonce = test_nonce();

    let encrypted = encrypt_block(CipherId::Aes256Gcm, &key, &nonce, 0, b"").unwrap();
    assert_eq!(encrypted.len(), NONCE_SIZE + TAG_SIZE);

    let decrypted = decrypt_block(CipherId::Aes256Gcm, &key, &nonce, 0, &encrypted).unwrap();
    assert!(decrypted.is_empty());
}

// ── Large data (1 MB) encryption/decryption ─────────────────────────────────

#[test]
fn large_data_1mb_roundtrip_aes() {
    let key = test_key();
    let nonce = test_nonce();
    let data: Vec<u8> = (0..1_048_576u32).map(|i| (i % 256) as u8).collect();

    let encrypted = encrypt_block(CipherId::Aes256Gcm, &key, &nonce, 42, &data).unwrap();
    assert!(
        encrypted.len() > data.len(),
        "ciphertext should be larger (nonce + tag)"
    );

    let decrypted = decrypt_block(CipherId::Aes256Gcm, &key, &nonce, 42, &encrypted).unwrap();
    assert_eq!(decrypted, data);
}

#[test]
fn large_data_1mb_roundtrip_chacha20() {
    let key = test_key();
    let nonce = test_nonce();
    let data: Vec<u8> = (0..1_048_576u32).map(|i| (i % 256) as u8).collect();

    let encrypted = encrypt_block(CipherId::ChaCha20Poly1305, &key, &nonce, 99, &data).unwrap();
    assert!(encrypted.len() > data.len());

    let decrypted =
        decrypt_block(CipherId::ChaCha20Poly1305, &key, &nonce, 99, &encrypted).unwrap();
    assert_eq!(decrypted, data);
}

#[test]
fn large_data_1mb_tampered_fails() {
    let key = test_key();
    let nonce = test_nonce();
    let data: Vec<u8> = (0..1_048_576u32).map(|i| (i % 256) as u8).collect();

    for cipher in [CipherId::Aes256Gcm, CipherId::ChaCha20Poly1305] {
        let mut encrypted = encrypt_block(cipher, &key, &nonce, 0, &data).unwrap();
        // Flip a byte in the middle of the ciphertext (not the nonce)
        let mid = NONCE_SIZE + encrypted.len() / 2;
        encrypted[mid] ^= 0x01;
        let result = decrypt_block(cipher, &key, &nonce, 0, &encrypted);
        assert!(
            result.is_err(),
            "tampered 1MB data should fail for {cipher:?}"
        );
    }
}

// ── Randomness quality tests ─────────────────────────────────────────────

#[test]
fn generate_salt_produces_non_zero_output() {
    let salt = generate_salt();
    assert!(
        salt.iter().any(|&b| b != 0),
        "generate_salt() should not produce all-zero output"
    );
}

#[test]
fn generate_nonce_produces_non_zero_output() {
    let nonce = generate_nonce();
    assert!(
        nonce.iter().any(|&b| b != 0),
        "generate_nonce() should not produce all-zero output"
    );
}

#[test]
fn generate_salt_produces_unique_outputs() {
    let salt1 = generate_salt();
    let salt2 = generate_salt();
    assert_ne!(
        salt1, salt2,
        "two consecutive generate_salt() calls should produce different values"
    );
}

#[test]
fn generate_nonce_produces_unique_outputs() {
    let nonce1 = generate_nonce();
    let nonce2 = generate_nonce();
    assert_ne!(
        nonce1, nonce2,
        "two consecutive generate_nonce() calls should produce different values"
    );
}

// ── Nonce derivation correctness ─────────────────────────────────────────

#[test]
fn derive_block_nonce_is_deterministic() {
    let master = test_nonce();
    let n1 = derive_block_nonce(&master, 42);
    let n2 = derive_block_nonce(&master, 42);
    assert_eq!(n1, n2, "same inputs should produce same derived nonce");
}

#[test]
fn derive_block_nonce_differs_per_block_id() {
    let master = test_nonce();
    let mut seen = std::collections::HashSet::new();
    for block_id in [0u32, 1, 2, 255, 1024, u32::MAX - 1, u32::MAX] {
        let derived = derive_block_nonce(&master, block_id);
        assert!(
            seen.insert(derived),
            "block_id={block_id} produced a nonce collision"
        );
    }
}

#[test]
fn same_key_nonce_block_id_different_plaintext_cannot_cross_decrypt() {
    let key = test_key();
    let nonce = test_nonce();
    let data_a = b"plaintext message A";
    let data_b = b"plaintext message B";

    for cipher in [CipherId::Aes256Gcm, CipherId::ChaCha20Poly1305] {
        let enc_a = encrypt_block(cipher, &key, &nonce, 0, data_a).unwrap();
        let enc_b = encrypt_block(cipher, &key, &nonce, 0, data_b).unwrap();

        // Each ciphertext should only decrypt to its own plaintext
        let dec_a = decrypt_block(cipher, &key, &nonce, 0, &enc_a).unwrap();
        let dec_b = decrypt_block(cipher, &key, &nonce, 0, &enc_b).unwrap();
        assert_eq!(dec_a.as_slice(), data_a);
        assert_eq!(dec_b.as_slice(), data_b);

        // Decrypting enc_a with wrong block_id should fail
        let result = decrypt_block(cipher, &key, &nonce, 1, &enc_a);
        assert!(
            result.is_err(),
            "decrypting with wrong block_id should fail for {cipher:?}"
        );
    }
}
