//! Azure Blob Storage backend.
//!
//! Requires the `azure_storage_blobs` crate (not included by default).
//! This module provides the `AzureBackend` struct that implements `StorageBackend`.

use super::{StorageBackend, ValidatedPath};
use crate::error::{AetherError, Result};

/// Azure Blob Storage backend configuration.
///
/// **Warning**: This backend is a stub — all operations return errors.
/// Real Azure SDK integration is not yet implemented.
#[deprecated(
    note = "AzureBackend is a stub — all operations will fail at runtime. \
    Real Azure integration is not yet implemented."
)]
#[allow(dead_code)]
pub struct AzureBackend {
    account: String,
}

impl AzureBackend {
    /// Create a new Azure Blob Storage backend.
    ///
    /// Uses default Azure credential chain (AZURE_STORAGE_CONNECTION_STRING, managed identity).
    ///
    /// The `account` name is validated: must be 3-24 lowercase alphanumeric
    /// characters (Azure storage account naming rules).  This prevents
    /// injection when the account name is interpolated into URLs.
    pub fn new(account: &str) -> Result<Self> {
        Self::validate_account_name(account)?;
        Ok(Self {
            account: account.to_string(),
        })
    }

    /// Validate Azure storage account name.
    ///
    /// Azure requires: 3-24 characters, lowercase letters and digits only.
    fn validate_account_name(account: &str) -> Result<()> {
        if account.len() < 3 || account.len() > 24 {
            return Err(AetherError::CloudStorage(
                "Azure account name must be 3-24 characters".into(),
            ));
        }
        if !account
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit())
        {
            return Err(AetherError::CloudStorage(
                "Azure account name must contain only lowercase letters and digits".into(),
            ));
        }
        Ok(())
    }
}

impl StorageBackend for AzureBackend {
    fn read_range(&self, _path: &ValidatedPath, _offset: u64, _length: u64) -> Result<Vec<u8>> {
        Err(AetherError::CloudStorage(
            "Azure backend not yet implemented — requires 'cloud-azure' feature".into(),
        ))
    }

    fn write(&self, _path: &ValidatedPath, _data: &[u8]) -> Result<()> {
        Err(AetherError::CloudStorage(
            "Azure backend not yet implemented — requires 'cloud-azure' feature".into(),
        ))
    }

    fn delete(&self, _path: &ValidatedPath) -> Result<()> {
        Err(AetherError::CloudStorage(
            "Azure backend not yet implemented — requires 'cloud-azure' feature".into(),
        ))
    }

    fn size(&self, _path: &ValidatedPath) -> Result<u64> {
        Err(AetherError::CloudStorage(
            "Azure backend not yet implemented — requires 'cloud-azure' feature".into(),
        ))
    }

    fn exists(&self, _path: &ValidatedPath) -> Result<bool> {
        Err(AetherError::CloudStorage(
            "Azure backend not yet implemented — requires 'cloud-azure' feature".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_valid_account_name() {
        assert!(AzureBackend::new("mystorageaccount1").is_ok());
    }

    #[test]
    fn rejects_short_account_name() {
        assert!(AzureBackend::new("ab").is_err());
    }

    #[test]
    fn rejects_long_account_name() {
        let long_name = "a".repeat(25);
        assert!(AzureBackend::new(&long_name).is_err());
    }

    #[test]
    fn rejects_uppercase_account_name() {
        assert!(AzureBackend::new("MyStorage").is_err());
    }

    #[test]
    fn rejects_hyphens_in_account_name() {
        // Azure storage accounts do not allow hyphens
        assert!(AzureBackend::new("my-storage").is_err());
    }

    #[test]
    fn rejects_special_chars_in_account_name() {
        assert!(AzureBackend::new("my_storage!").is_err());
        assert!(AzureBackend::new("my storage").is_err());
        assert!(AzureBackend::new("my.storage").is_err());
    }

    #[test]
    fn accepts_boundary_lengths() {
        // Exactly 3 characters (minimum)
        assert!(AzureBackend::new("abc").is_ok());
        // Exactly 24 characters (maximum)
        let name_24 = "a".repeat(24);
        assert!(AzureBackend::new(&name_24).is_ok());
    }
}
