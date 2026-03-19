//! Fuzz target for encryption header parsing.
//!
//! Exercises `EncryptionHeader::read_from` with arbitrary bytes to catch
//! panics, invalid Argon2 parameter acceptance, and parsing errors from
//! crafted encryption headers. The crypto header accepts parameters
//! (m_cost, t_cost, p_cost) from untrusted input, so validating that
//! resource-limit checks cannot be bypassed is critical.

#![no_main]

use libfuzzer_sys::fuzz_target;
use std::io::Cursor;

use aether_core::crypto::EncryptionHeader;
use aether_core::format::{MAX_ARGON2_M_COST, MAX_ARGON2_T_COST, MAX_ARGON2_P_COST};

fuzz_target!(|data: &[u8]| {
    let mut cursor = Cursor::new(data);
    if let Ok(header) = EncryptionHeader::read_from(&mut cursor) {
        // If parsing succeeded, verify that Argon2 safety bounds hold.
        // These prevent resource-exhaustion DoS from crafted archives.
        assert!(
            header.m_cost <= MAX_ARGON2_M_COST,
            "EncryptionHeader accepted m_cost {} above limit {}",
            header.m_cost, MAX_ARGON2_M_COST,
        );
        assert!(
            header.t_cost <= MAX_ARGON2_T_COST,
            "EncryptionHeader accepted t_cost {} above limit {}",
            header.t_cost, MAX_ARGON2_T_COST,
        );
        assert!(
            header.p_cost <= MAX_ARGON2_P_COST,
            "EncryptionHeader accepted p_cost {} above limit {}",
            header.p_cost, MAX_ARGON2_P_COST,
        );
    }
});
