//! Amazon S3 storage backend.
//!
//! Requires the `aws-sdk-s3` crate (not included by default).
//! This module provides the `S3Backend` struct that implements `StorageBackend`.
//!
//! # Usage
//!
//! ```rust,ignore
//! use aether_core::cloud::s3::S3Backend;
//! use aether_core::cloud::CloudReader;
//!
//! let backend = S3Backend::new("us-east-1", None)?;
//! let reader = CloudReader::new(backend, "my-bucket/archive.aet".to_string())?;
//! ```

use super::{StorageBackend, ValidatedPath};
use crate::error::{AetherError, Result};

/// S3 storage backend configuration.
///
/// **Warning**: This backend is a stub — all operations return errors.
/// Real S3 SDK integration is not yet implemented.
#[deprecated(note = "S3Backend is a stub — all operations will fail at runtime. \
    Real S3 integration is not yet implemented.")]
#[allow(dead_code)]
pub struct S3Backend {
    region: String,
    endpoint: Option<String>,
    /// IP addresses resolved and validated at construction time.
    /// When integrating the real AWS SDK, configure the HTTP client to
    /// **only** connect to these pinned IPs (or re-validate via
    /// [`check_ip_is_public`] at connect time) to prevent DNS rebinding.
    pinned_ips: Vec<std::net::IpAddr>,
}

impl S3Backend {
    /// Create a new S3 backend.
    ///
    /// Uses default AWS credential chain (env vars, profile, EC2 instance role).
    /// Optionally specify a custom endpoint for S3-compatible services (MinIO, LocalStack).
    ///
    /// # Endpoint restrictions
    ///
    /// If provided, the endpoint must use HTTPS and must not point to a
    /// link-local or private IP address (blocks SSRF against cloud metadata
    /// services such as `169.254.169.254`).
    pub fn new(region: &str, endpoint: Option<String>) -> Result<Self> {
        let pinned_ips = if let Some(ref ep) = endpoint {
            Self::validate_endpoint(ep)?
        } else {
            Vec::new()
        };
        Ok(Self {
            region: region.to_string(),
            endpoint,
            pinned_ips,
        })
    }

    /// Returns the IP addresses that were resolved and validated at
    /// construction time.  When integrating the real HTTP client, use
    /// these to restrict connections and prevent DNS rebinding attacks.
    pub fn pinned_ips(&self) -> &[std::net::IpAddr] {
        &self.pinned_ips
    }

    /// Validate that the custom endpoint is safe to use.
    ///
    /// Resolves the hostname via DNS and checks that **all** resolved IP
    /// addresses are public.  DNS resolution failure is treated as an error
    /// (fail-closed) to prevent SSRF via transient DNS failures.
    ///
    /// Returns the validated IP addresses so they can be pinned on the
    /// `S3Backend` struct.  When integrating the real AWS SDK, the HTTP
    /// client must be configured to only connect to these pinned IPs (or
    /// re-validate via [`check_ip_is_public`] at connect time) to fully
    /// prevent DNS rebinding attacks.
    fn validate_endpoint(endpoint: &str) -> Result<Vec<std::net::IpAddr>> {
        // Must be HTTPS to prevent credential leakage over plaintext.
        if !endpoint.starts_with("https://") {
            return Err(AetherError::CloudStorage(
                "custom S3 endpoint must use HTTPS".into(),
            ));
        }

        // Extract the host portion (strip scheme and optional path/port).
        let host = endpoint
            .strip_prefix("https://")
            .unwrap_or(endpoint)
            .split('/')
            .next()
            .unwrap_or("")
            .split(':')
            .next()
            .unwrap_or("");

        // First, check if the host string itself is a known-bad literal.
        Self::check_host_is_public(host)?;

        // Resolve DNS and check all resolved IPs are public.
        // Fail-closed: if DNS resolution fails, reject the endpoint rather
        // than silently skipping validation.
        use std::net::ToSocketAddrs;
        let socket_addr = format!("{host}:443");
        let addrs = socket_addr.to_socket_addrs().map_err(|e| {
            AetherError::CloudStorage(format!("failed to resolve S3 endpoint hostname: {e}"))
        })?;

        let mut validated_ips = Vec::new();
        for addr in addrs {
            Self::check_ip_is_public(addr.ip())?;
            validated_ips.push(addr.ip());
        }

        if validated_ips.is_empty() {
            return Err(AetherError::CloudStorage(
                "S3 endpoint hostname resolved to zero addresses".into(),
            ));
        }

        Ok(validated_ips)
    }

    /// Check that a host string doesn't directly specify a private/loopback address.
    ///
    /// Handles plain IPv4 (`127.0.0.1`), bracketed IPv6 (`[::1]`), and
    /// common names (`localhost`).
    fn check_host_is_public(host: &str) -> Result<()> {
        // Block `localhost` and any subdomain thereof.
        let lower = host.to_ascii_lowercase();
        if lower == "localhost" || lower.ends_with(".localhost") {
            return Err(AetherError::CloudStorage(
                "custom S3 endpoint must not target localhost".into(),
            ));
        }

        // Try to parse the host as an IP address (stripping brackets for IPv6).
        // check_ip_is_public already handles unspecified (0.0.0.0) via is_unspecified().
        let bare = host.trim_start_matches('[').trim_end_matches(']');
        if let Ok(ip) = bare.parse::<std::net::IpAddr>() {
            Self::check_ip_is_public(ip)?;
        }

        Ok(())
    }

    /// Validate that an IP address is public (not loopback, private, or
    /// link-local).
    ///
    /// Handles both IPv4 and IPv6, including IPv4-mapped IPv6 addresses
    /// like `::ffff:127.0.0.1`.
    fn check_ip_is_public(ip: std::net::IpAddr) -> Result<()> {
        use std::net::IpAddr;

        // Canonicalize IPv4-mapped IPv6 (e.g. ::ffff:169.254.169.254) to IPv4
        // so that all private-range checks apply uniformly.
        let canonical = match ip {
            IpAddr::V6(v6) => match v6.to_ipv4_mapped() {
                Some(v4) => IpAddr::V4(v4),
                None => IpAddr::V6(v6),
            },
            other => other,
        };

        match canonical {
            IpAddr::V4(v4) => {
                let octets = v4.octets();
                let is_private = v4.is_loopback()        // 127.0.0.0/8
                    || v4.is_unspecified()                // 0.0.0.0
                    || v4.is_multicast()                  // 224.0.0.0/4
                    || v4.is_broadcast()                  // 255.255.255.255
                    || octets[0] == 10                    // 10.0.0.0/8
                    || (octets[0] == 172 && (16..=31).contains(&octets[1])) // 172.16.0.0/12
                    || (octets[0] == 192 && octets[1] == 168)              // 192.168.0.0/16
                    || (octets[0] == 169 && octets[1] == 254)              // 169.254.0.0/16 (link-local)
                    || (octets[0] == 100 && (64..=127).contains(&octets[1])) // 100.64.0.0/10 (CGN)
                    || (octets[0] == 198 && (18..=19).contains(&octets[1])); // 198.18.0.0/15 (benchmarking)

                if is_private {
                    return Err(AetherError::CloudStorage(
                        "custom S3 endpoint resolves to a non-public IPv4 address".into(),
                    ));
                }
            }
            IpAddr::V6(v6) => {
                let is_private = v6.is_loopback()         // ::1
                    || v6.is_unspecified()                 // ::
                    || v6.is_multicast()                   // ff00::/8
                    || is_ipv6_link_local(&v6)             // fe80::/10
                    || is_ipv6_unique_local(&v6); // fc00::/7

                if is_private {
                    return Err(AetherError::CloudStorage(
                        "custom S3 endpoint resolves to a non-public IPv6 address".into(),
                    ));
                }
            }
        }

        Ok(())
    }
}

/// Check if an IPv6 address is link-local (fe80::/10).
fn is_ipv6_link_local(addr: &std::net::Ipv6Addr) -> bool {
    let segments = addr.segments();
    (segments[0] & 0xffc0) == 0xfe80
}

/// Check if an IPv6 address is unique-local (fc00::/7).
fn is_ipv6_unique_local(addr: &std::net::Ipv6Addr) -> bool {
    let segments = addr.segments();
    (segments[0] & 0xfe00) == 0xfc00
}

impl StorageBackend for S3Backend {
    fn read_range(&self, _path: &ValidatedPath, _offset: u64, _length: u64) -> Result<Vec<u8>> {
        // TODO: Implement with aws-sdk-s3 GetObject with Range header
        Err(AetherError::CloudStorage(
            "S3 backend not yet implemented — requires 'cloud-s3' feature".into(),
        ))
    }

    fn write(&self, _path: &ValidatedPath, _data: &[u8]) -> Result<()> {
        Err(AetherError::CloudStorage(
            "S3 backend not yet implemented — requires 'cloud-s3' feature".into(),
        ))
    }

    fn delete(&self, _path: &ValidatedPath) -> Result<()> {
        Err(AetherError::CloudStorage(
            "S3 backend not yet implemented — requires 'cloud-s3' feature".into(),
        ))
    }

    fn size(&self, _path: &ValidatedPath) -> Result<u64> {
        Err(AetherError::CloudStorage(
            "S3 backend not yet implemented — requires 'cloud-s3' feature".into(),
        ))
    }

    fn exists(&self, _path: &ValidatedPath) -> Result<bool> {
        Err(AetherError::CloudStorage(
            "S3 backend not yet implemented — requires 'cloud-s3' feature".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_http_endpoint() {
        assert!(S3Backend::new("us-east-1", Some("http://example.com".into())).is_err());
    }

    #[test]
    fn rejects_link_local_endpoint() {
        assert!(
            S3Backend::new("us-east-1", Some("https://169.254.169.254/latest".into())).is_err()
        );
    }

    #[test]
    fn rejects_private_endpoints() {
        assert!(S3Backend::new("us-east-1", Some("https://10.0.0.1".into())).is_err());
        assert!(S3Backend::new("us-east-1", Some("https://192.168.1.1".into())).is_err());
        assert!(S3Backend::new("us-east-1", Some("https://172.16.0.1".into())).is_err());
        assert!(S3Backend::new("us-east-1", Some("https://127.0.0.1".into())).is_err());
        assert!(S3Backend::new("us-east-1", Some("https://localhost".into())).is_err());
    }

    #[test]
    fn rejects_full_loopback_range() {
        // All of 127.0.0.0/8 is loopback, not just 127.0.0.1
        assert!(S3Backend::new("us-east-1", Some("https://127.0.0.2".into())).is_err());
        assert!(S3Backend::new("us-east-1", Some("https://127.255.255.254".into())).is_err());
    }

    #[test]
    fn rejects_unspecified_address() {
        assert!(S3Backend::new("us-east-1", Some("https://0.0.0.0".into())).is_err());
    }

    #[test]
    fn rejects_ipv6_loopback() {
        assert!(S3Backend::new("us-east-1", Some("https://[::1]".into())).is_err());
    }

    #[test]
    fn rejects_cgn_range() {
        // 100.64.0.0/10 (Carrier-grade NAT)
        assert!(S3Backend::new("us-east-1", Some("https://100.64.0.1".into())).is_err());
        assert!(S3Backend::new("us-east-1", Some("https://100.127.255.254".into())).is_err());
    }

    #[test]
    fn rejects_localhost_subdomains() {
        assert!(S3Backend::new("us-east-1", Some("https://foo.localhost".into())).is_err());
    }

    #[test]
    fn check_ip_rejects_ipv4_mapped_ipv6() {
        use std::net::IpAddr;
        // ::ffff:127.0.0.1 should be caught as loopback
        let ip: IpAddr = "::ffff:127.0.0.1".parse().unwrap();
        assert!(S3Backend::check_ip_is_public(ip).is_err());
        // ::ffff:169.254.169.254 should be caught as link-local
        let ip: IpAddr = "::ffff:169.254.169.254".parse().unwrap();
        assert!(S3Backend::check_ip_is_public(ip).is_err());
        // ::ffff:10.0.0.1 should be caught as private
        let ip: IpAddr = "::ffff:10.0.0.1".parse().unwrap();
        assert!(S3Backend::check_ip_is_public(ip).is_err());
    }

    #[test]
    fn check_ip_rejects_ipv6_link_local() {
        use std::net::IpAddr;
        let ip: IpAddr = "fe80::1".parse().unwrap();
        assert!(S3Backend::check_ip_is_public(ip).is_err());
    }

    #[test]
    fn check_ip_rejects_ipv6_unique_local() {
        use std::net::IpAddr;
        let ip: IpAddr = "fd00::1".parse().unwrap();
        assert!(S3Backend::check_ip_is_public(ip).is_err());
    }

    #[test]
    fn check_ip_allows_public() {
        use std::net::IpAddr;
        let ip: IpAddr = "8.8.8.8".parse().unwrap();
        assert!(S3Backend::check_ip_is_public(ip).is_ok());
        let ip: IpAddr = "2001:4860:4860::8888".parse().unwrap();
        assert!(S3Backend::check_ip_is_public(ip).is_ok());
    }

    #[test]
    fn accepts_valid_https_endpoint() {
        assert!(S3Backend::new(
            "us-east-1",
            Some("https://s3.us-east-1.amazonaws.com".into())
        )
        .is_ok());
    }

    #[test]
    fn accepts_no_endpoint() {
        assert!(S3Backend::new("us-east-1", None).is_ok());
    }
}
