# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.2.3 (latest) | Yes |
| < 0.2.3 | No |

Only the latest release receives security updates. Users should always
upgrade to the latest version.

## Reporting a Vulnerability

**Do NOT report security vulnerabilities through public GitHub Issues.**

Instead, please report vulnerabilities via one of:

1. **GitHub Security Advisories**: Use the "Report a vulnerability" button
   on the repository's Security tab (preferred).
2. **Email**: Send details to security@craton.com.ar.

### What to Include

- Description of the vulnerability
- Steps to reproduce
- Potential impact assessment
- Suggested fix (if any)

### Response Timeline

- **Acknowledgment**: Within 48 hours
- **Initial assessment**: Within 7 days
- **Fix or mitigation**: Within 30 days for critical issues

### Scope

The following are in scope for security reports:

- Memory safety issues (buffer overflows, use-after-free) in compression/decompression
- Crafted archives that cause denial of service (OOM, infinite loops, excessive CPU)
- Path traversal during extraction (e.g., `../../etc/passwd` in file paths)
- Integer overflows in size calculations
- Any issue in unsafe code or FFI boundaries

### Out of Scope

- Performance issues that don't lead to resource exhaustion
- Issues requiring physical access to the machine
- Social engineering

## Known Security Considerations

### Pure-Rust Dependencies
AetherArch uses the pure-Rust `divsufsort` crate for suffix array construction
in the BWT preprocessing step, eliminating the previous C FFI boundary
(`cdivsufsort`). The core library has no unsafe FFI dependencies.

### Archive Parsing
Archive headers contain size fields that control memory allocation. Bounds
checking is enforced with `MAX_DECOMPRESSED_BLOCK_SIZE` (64 MiB),
`MAX_FILE_COUNT`, and `MAX_BLOCK_COUNT` limits to prevent resource
exhaustion from crafted archives.

### Extraction Path Safety
File paths stored in archives are validated to prevent path traversal.
Paths containing `..` components or absolute paths are rejected during
extraction.

### Encryption (Enterprise Feature)
When the `enterprise` feature is enabled, AetherArch supports AES-256-GCM
and ChaCha20-Poly1305 encryption with Argon2id key derivation.

**Side-channel considerations**: The encryption implementation uses the
`aes-gcm` and `chacha20poly1305` crates, which provide constant-time
operations on most platforms. However, AetherArch does **not** currently
provide timing-attack resistance in the key comparison or KDF paths
beyond what Argon2id provides natively. Deployments where an attacker
can measure sub-microsecond timing differences on the decryption path
should evaluate whether additional hardening (e.g., hardware-backed key
storage) is appropriate.

Random nonces are generated via the OS CSPRNG (`getrandom` crate).
Per-block nonces are derived as `master_nonce XOR block_id`, ensuring
unique nonces for random-access decryption without requiring additional
random bytes per block.

### REST API Server (`aether-server`)
The server supports optional API key authentication via the
`AETHER_API_KEY` environment variable. When set, all mutating endpoints
require `Authorization: Bearer <key>`. The API key comparison uses
standard string equality (not constant-time), which is acceptable for
bearer tokens but noted here for completeness.

## Disclosure Policy

We follow coordinated disclosure:

1. Reporter notifies us privately
2. We confirm and assess the issue
3. We develop and test a fix
4. We release the fix and publish a security advisory
5. Reporter may publish details 90 days after initial report

We credit reporters in security advisories (unless they prefer anonymity).

---

Copyright 2024-2026 Craton Software Company Licensed under Apache-2.0.
