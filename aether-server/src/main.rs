mod error;
mod routes;

use std::sync::Arc;

use axum::routing::{get, post};
use axum::Router;
use clap::Parser;
use tower_governor::governor::GovernorConfigBuilder;
use tower_governor::key_extractor::{PeerIpKeyExtractor, SmartIpKeyExtractor};
use tower_governor::GovernorLayer;
use tower_http::cors::{AllowMethods, AllowOrigin, CorsLayer};
use tower_http::limit::RequestBodyLimitLayer;
use tower_http::set_header::SetResponseHeaderLayer;
use tower_http::trace::TraceLayer;

/// Shared application state passed to handlers via axum's State extractor.
#[derive(Clone)]
pub struct AppState {
    /// The expected API key, read once at startup. `None` means auth is disabled.
    pub api_key: Option<String>,
}

#[derive(Parser)]
#[command(
    name = "aether-server",
    about = "AetherArch compression REST API server"
)]
struct Args {
    /// Port to listen on
    #[arg(short, long, default_value = "3000")]
    port: u16,

    /// Maximum upload size in MiB (max 4095 to prevent overflow)
    #[arg(long, default_value = "256")]
    max_upload_mib: usize,

    /// Bind address (default: 127.0.0.1 for safety; use 0.0.0.0 to expose)
    #[arg(long, default_value = "127.0.0.1")]
    bind: String,

    /// Trust X-Forwarded-For/X-Real-Ip/Forwarded headers from reverse proxy
    /// for per-client rate limiting. Only enable if behind a trusted reverse proxy.
    #[arg(long, default_value = "false")]
    trust_proxy: bool,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    // Security fix: checked arithmetic to prevent silent overflow on 32-bit targets.
    let max_body = args
        .max_upload_mib
        .checked_mul(1024)
        .and_then(|v| v.checked_mul(1024))
        .unwrap_or_else(|| {
            tracing::error!(
                "--max-upload-mib {} overflows usize; clamping to {} MiB",
                args.max_upload_mib,
                usize::MAX / (1024 * 1024)
            );
            usize::MAX
        });

    // Read API key once at startup — avoids per-request env var access (thread-unsafe)
    // and syscall overhead. Restart server to rotate keys.
    let api_key = std::env::var("AETHER_API_KEY").ok();
    if api_key.is_none() {
        tracing::warn!(
            "AETHER_API_KEY not set — all endpoints are unauthenticated! \
             Set AETHER_API_KEY to require Bearer token authentication."
        );
    }
    let state = Arc::new(AppState { api_key });

    // TLS warning: this server does not terminate TLS. If exposed to a network,
    // API keys travel in plaintext.
    if args.bind == "0.0.0.0" || args.bind == "::" {
        tracing::warn!(
            "Server is binding to all interfaces ({}). Ensure TLS is terminated \
             by a reverse proxy (nginx, Caddy, etc.) before exposing to a network. \
             API keys are transmitted in plaintext over HTTP.",
            args.bind,
        );
    }

    if args.trust_proxy {
        tracing::info!("Trusting X-Forwarded-For for rate limiting (--trust-proxy)");
    }

    // M2: Limit concurrent blocking operations to prevent thread pool starvation.
    let concurrency_limit = tower::limit::ConcurrencyLimitLayer::new(
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4),
    );

    let cors = build_cors_layer();

    let app = Router::new()
        .route("/compress", post(routes::compress))
        .route("/extract", post(routes::extract))
        .route("/verify", post(routes::verify))
        .route("/list", post(routes::list))
        .route("/health", get(routes::health))
        .route("/version", get(routes::version))
        .with_state(state)
        .layer(cors);

    // S17 security fix: per-IP rate limiting to prevent abuse.
    // When behind a reverse proxy, use --trust-proxy so rate limiting uses
    // X-Forwarded-For/X-Real-Ip instead of the proxy's IP (which would
    // bucket all clients together).
    let app = if args.trust_proxy {
        let conf = GovernorConfigBuilder::default()
            .key_extractor(SmartIpKeyExtractor)
            .per_second(10)
            .burst_size(30)
            .finish()
            .expect("governor config with SmartIpKeyExtractor");
        app.layer(GovernorLayer::new(conf))
    } else {
        let conf = GovernorConfigBuilder::default()
            .key_extractor(PeerIpKeyExtractor)
            .per_second(10)
            .burst_size(30)
            .finish()
            .expect("governor config with PeerIpKeyExtractor");
        app.layer(GovernorLayer::new(conf))
    };

    let app = app
        .layer(concurrency_limit)
        .layer(RequestBodyLimitLayer::new(max_body))
        // Request logging for audit trail.
        .layer(TraceLayer::new_for_http())
        // Security headers
        .layer(SetResponseHeaderLayer::overriding(
            axum::http::header::X_CONTENT_TYPE_OPTIONS,
            axum::http::HeaderValue::from_static("nosniff"),
        ))
        .layer(SetResponseHeaderLayer::overriding(
            axum::http::header::X_FRAME_OPTIONS,
            axum::http::HeaderValue::from_static("DENY"),
        ))
        .layer(SetResponseHeaderLayer::overriding(
            axum::http::header::CACHE_CONTROL,
            axum::http::HeaderValue::from_static("no-store"),
        ))
        .layer(SetResponseHeaderLayer::overriding(
            axum::http::header::CONTENT_SECURITY_POLICY,
            axum::http::HeaderValue::from_static("default-src 'none'"),
        ));

    let addr = format!("{}:{}", args.bind, args.port);
    tracing::info!("Starting aether-server on {addr}");
    tracing::info!("Max upload: {} MiB", args.max_upload_mib);

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .expect("Failed to bind");

    axum::serve(listener, app).await.expect("Server error");
}

/// Build CORS layer with restrictive defaults.
///
/// M1: Restrictive CORS — no cross-origin requests by default.
/// Set AETHER_CORS_ORIGIN to allow specific origins.
/// V13 fix: reject wildcard "*" and "null" to prevent accidental open CORS.
fn build_cors_layer() -> CorsLayer {
    let localhost_origin = AllowOrigin::exact("http://localhost".parse().unwrap());

    let allow_origin = match std::env::var("AETHER_CORS_ORIGIN") {
        Ok(origin) if origin == "*" => {
            tracing::error!(
                "AETHER_CORS_ORIGIN='*' is not allowed — this would expose the API to \
                 all origins. Set a specific origin (e.g. 'https://app.example.com') or \
                 leave unset for localhost-only. Falling back to localhost."
            );
            localhost_origin
        }
        Ok(origin) if origin.eq_ignore_ascii_case("null") => {
            tracing::error!(
                "AETHER_CORS_ORIGIN='null' is not allowed — the 'null' origin matches \
                 sandboxed iframes and can be spoofed. Falling back to localhost."
            );
            localhost_origin
        }
        Ok(ref origin) if !origin.starts_with("http://") && !origin.starts_with("https://") => {
            tracing::error!(
                "AETHER_CORS_ORIGIN='{}' must start with http:// or https://. \
                 Falling back to localhost.",
                origin,
            );
            localhost_origin
        }
        Ok(origin) => match origin.parse::<axum::http::HeaderValue>() {
            Ok(val) => AllowOrigin::exact(val),
            Err(e) => {
                tracing::warn!(
                    "Invalid AETHER_CORS_ORIGIN value '{}': {e}. Falling back to localhost.",
                    origin,
                );
                localhost_origin
            }
        },
        Err(_) => localhost_origin,
    };

    CorsLayer::new()
        .allow_origin(allow_origin)
        .allow_methods(AllowMethods::list([
            axum::http::Method::GET,
            axum::http::Method::POST,
        ]))
        .allow_headers([
            axum::http::header::CONTENT_TYPE,
            axum::http::header::AUTHORIZATION,
        ])
        .max_age(std::time::Duration::from_secs(300))
}
