use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::json;

/// API error type that maps to JSON error responses.
#[derive(Debug)]
pub enum ApiError {
    BadRequest(String),
    Unauthorized(String),
    Internal(String),
}

/// Messages that are safe to return verbatim in BadRequest responses.
/// Any message not in this list is replaced with "Bad request" to prevent
/// information disclosure (hostnames, usernames, internal state, etc.).
const SAFE_BAD_REQUEST_MESSAGES: &[&str] = &[
    "Bad request",
    "No files uploaded",
    "No archive uploaded",
    "Invalid upload",
    "Archive contains a symlink, which is not permitted",
    "Extracted path escapes output directory",
    "Extracted file count exceeds server limit",
    "Extracted data exceeds server decompression size limit",
];

/// Check if a BadRequest message is safe to expose to clients.
/// Allows exact matches from the allowlist and parameterized messages that
/// follow known safe patterns.
fn is_safe_bad_request(msg: &str) -> bool {
    if SAFE_BAD_REQUEST_MESSAGES.contains(&msg) {
        return true;
    }
    // Allow parameterized messages with known-safe prefixes.
    msg.starts_with("Field exceeds maximum size of ")
        || msg.starts_with("Upload exceeds maximum size of ")
        || msg.starts_with("Upload exceeds maximum file count of ")
        || msg.starts_with("Archive contains ")
        || msg.starts_with("Archive declared total size ")
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            // S14 security fix: only expose allowlisted messages to clients.
            // Log the original message server-side for debugging, then replace
            // with a generic message if it's not on the safe list.
            ApiError::BadRequest(msg) => {
                if is_safe_bad_request(&msg) {
                    (StatusCode::BAD_REQUEST, msg)
                } else {
                    tracing::warn!("Sanitized BadRequest message: {msg}");
                    (StatusCode::BAD_REQUEST, "Bad request".to_string())
                }
            }
            // Hardcode the unauthorized message to prevent future call sites
            // from accidentally leaking information through this variant.
            ApiError::Unauthorized(msg) => {
                tracing::debug!("Unauthorized: {msg}");
                (
                    StatusCode::UNAUTHORIZED,
                    "Invalid or missing API key".to_string(),
                )
            }
            // M3 security fix: sanitize internal error messages before sending to
            // clients. Log the full detail server-side, but only return a generic
            // message to prevent leaking filesystem paths or internal state.
            ApiError::Internal(msg) => {
                tracing::error!("Internal error: {msg}");
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Internal server error".to_string(),
                )
            }
        };

        let body = json!({ "error": message });
        (status, axum::Json(body)).into_response()
    }
}

impl From<aether_core::error::AetherError> for ApiError {
    fn from(e: aether_core::error::AetherError) -> Self {
        ApiError::Internal(e.to_string())
    }
}

impl From<std::io::Error> for ApiError {
    fn from(e: std::io::Error) -> Self {
        ApiError::Internal(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::StatusCode;
    use axum::response::IntoResponse;
    use http_body_util::BodyExt;

    /// Extract the status code and JSON body from an ApiError response.
    async fn response_parts(err: ApiError) -> (StatusCode, serde_json::Value) {
        let resp = err.into_response();
        let status = resp.status();
        let body_bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
        (status, json)
    }

    #[tokio::test]
    async fn bad_request_allowlisted_message_preserved() {
        let (status, body) = response_parts(ApiError::BadRequest("No files uploaded".into())).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"], "No files uploaded");
    }

    #[tokio::test]
    async fn bad_request_parameterized_message_preserved() {
        let (status, body) = response_parts(ApiError::BadRequest(
            "Archive contains 50000 files, exceeding server limit of 10000".into(),
        ))
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(body["error"]
            .as_str()
            .unwrap()
            .starts_with("Archive contains"));
    }

    #[tokio::test]
    async fn bad_request_sanitizes_unknown_message() {
        let (status, body) =
            response_parts(ApiError::BadRequest("/etc/passwd leaked".into())).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"], "Bad request");
    }

    #[tokio::test]
    async fn bad_request_sanitizes_hostname_leak() {
        let (status, body) =
            response_parts(ApiError::BadRequest("failed on host db-prod-03".into())).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"], "Bad request");
    }

    #[tokio::test]
    async fn bad_request_sanitizes_backslash() {
        let (status, body) = response_parts(ApiError::BadRequest("C:\\Users\\secret".into())).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"], "Bad request");
    }

    #[tokio::test]
    async fn unauthorized_always_returns_hardcoded_message() {
        // Even with a custom message, Unauthorized always returns the hardcoded string
        // to prevent information leakage from future call sites.
        let (status, body) =
            response_parts(ApiError::Unauthorized("some internal detail".into())).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
        assert_eq!(body["error"], "Invalid or missing API key");
    }

    #[tokio::test]
    async fn internal_always_returns_generic_message() {
        let (status, body) = response_parts(ApiError::Internal(
            "db connection failed at /var/run/db.sock".into(),
        ))
        .await;
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        // The detailed message must never leak to the client.
        assert_eq!(body["error"], "Internal server error");
    }

    #[tokio::test]
    async fn internal_hides_filesystem_paths() {
        let (status, body) = response_parts(ApiError::Internal(
            "Failed to read /tmp/aether/archive.aet".into(),
        ))
        .await;
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        let msg = body["error"].as_str().unwrap();
        assert!(
            !msg.contains("/tmp"),
            "Internal error leaked filesystem path"
        );
        assert_eq!(msg, "Internal server error");
    }

    #[test]
    fn from_io_error_produces_internal() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let api_err: ApiError = io_err.into();
        // Verify it converts to Internal variant
        match api_err {
            ApiError::Internal(msg) => assert!(msg.contains("file not found")),
            _ => panic!("Expected Internal, got other variant"),
        }
    }
}
