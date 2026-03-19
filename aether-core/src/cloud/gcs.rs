//! Google Cloud Storage backend.
//!
//! Requires the `google-cloud-storage` crate (not included by default).
//! This module provides the `GcsBackend` struct that implements `StorageBackend`.

use super::{StorageBackend, ValidatedPath};
use crate::error::{AetherError, Result};

/// GCS storage backend configuration.
///
/// **Warning**: This backend is a stub — all operations return errors.
/// Real GCS SDK integration is not yet implemented.
#[deprecated(note = "GcsBackend is a stub — all operations will fail at runtime. \
    Real GCS integration is not yet implemented.")]
#[allow(dead_code)]
pub struct GcsBackend {
    project_id: Option<String>,
}

impl GcsBackend {
    /// Create a new GCS backend.
    ///
    /// Uses default Google Cloud credential chain (GOOGLE_APPLICATION_CREDENTIALS, gcloud CLI).
    ///
    /// If provided, the `project_id` is validated: must be 6-30 characters,
    /// lowercase letters, digits, and hyphens only, starting with a letter
    /// (GCP project ID rules).  This prevents injection when the ID is
    /// interpolated into API requests.
    pub fn new(project_id: Option<String>) -> Result<Self> {
        if let Some(ref id) = project_id {
            Self::validate_project_id(id)?;
        }
        Ok(Self { project_id })
    }

    /// Validate a GCP project ID.
    ///
    /// GCP requires: 6-30 characters, lowercase letters, digits, and hyphens,
    /// must start with a letter, must not end with a hyphen.
    fn validate_project_id(id: &str) -> Result<()> {
        if id.len() < 6 || id.len() > 30 {
            return Err(AetherError::CloudStorage(
                "GCP project ID must be 6-30 characters".into(),
            ));
        }
        if !id.starts_with(|c: char| c.is_ascii_lowercase()) {
            return Err(AetherError::CloudStorage(
                "GCP project ID must start with a lowercase letter".into(),
            ));
        }
        if id.ends_with('-') {
            return Err(AetherError::CloudStorage(
                "GCP project ID must not end with a hyphen".into(),
            ));
        }
        if !id
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
        {
            return Err(AetherError::CloudStorage(
                "GCP project ID must contain only lowercase letters, digits, and hyphens".into(),
            ));
        }
        Ok(())
    }
}

impl StorageBackend for GcsBackend {
    fn read_range(&self, _path: &ValidatedPath, _offset: u64, _length: u64) -> Result<Vec<u8>> {
        Err(AetherError::CloudStorage(
            "GCS backend not yet implemented — requires 'cloud-gcs' feature".into(),
        ))
    }

    fn write(&self, _path: &ValidatedPath, _data: &[u8]) -> Result<()> {
        Err(AetherError::CloudStorage(
            "GCS backend not yet implemented — requires 'cloud-gcs' feature".into(),
        ))
    }

    fn delete(&self, _path: &ValidatedPath) -> Result<()> {
        Err(AetherError::CloudStorage(
            "GCS backend not yet implemented — requires 'cloud-gcs' feature".into(),
        ))
    }

    fn size(&self, _path: &ValidatedPath) -> Result<u64> {
        Err(AetherError::CloudStorage(
            "GCS backend not yet implemented — requires 'cloud-gcs' feature".into(),
        ))
    }

    fn exists(&self, _path: &ValidatedPath) -> Result<bool> {
        Err(AetherError::CloudStorage(
            "GCS backend not yet implemented — requires 'cloud-gcs' feature".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_valid_project_id() {
        assert!(GcsBackend::new(Some("my-project-123".into())).is_ok());
    }

    #[test]
    fn accepts_no_project_id() {
        assert!(GcsBackend::new(None).is_ok());
    }

    #[test]
    fn rejects_short_project_id() {
        assert!(GcsBackend::new(Some("short".into())).is_err());
    }

    #[test]
    fn rejects_long_project_id() {
        let long_id = "a".repeat(31);
        assert!(GcsBackend::new(Some(long_id)).is_err());
    }

    #[test]
    fn rejects_uppercase_project_id() {
        assert!(GcsBackend::new(Some("My-Project".into())).is_err());
    }

    #[test]
    fn rejects_project_id_not_starting_with_letter() {
        assert!(GcsBackend::new(Some("123-project".into())).is_err());
        assert!(GcsBackend::new(Some("-my-project".into())).is_err());
    }

    #[test]
    fn rejects_project_id_ending_with_hyphen() {
        assert!(GcsBackend::new(Some("my-project-".into())).is_err());
    }

    #[test]
    fn rejects_project_id_with_special_chars() {
        assert!(GcsBackend::new(Some("my_project!".into())).is_err());
        assert!(GcsBackend::new(Some("my project".into())).is_err());
    }
}
