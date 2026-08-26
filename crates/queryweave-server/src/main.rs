#![forbid(unsafe_code)]

use axum::{
    extract::{DefaultBodyLimit, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{delete, get, post},
    Json, Router,
};
use queryweave_core::{
    BuiltinBm25Index, Document, EngineStats, ExactVectorIndex, HashEmbedder, HashSparseEncoder,
    LexicalRetriever, QueryWeaveEngine, SearchRequest, SearchResponse, VectorIndex,
};
use queryweave_tantivy::TantivyLexicalIndex;
use queryweave_usearch::USearchHnswIndex;
use serde::{Deserialize, Serialize};
use std::{env, sync::Arc};
use tokio::sync::Semaphore;

const DEFAULT_MAX_BODY_BYTES: usize = 32 * 1024 * 1024;
const DEFAULT_MAX_BATCH_DOCUMENTS: usize = 10_000;
const DEFAULT_MAX_QUERY_CHARS: usize = 16_384;

#[derive(Debug, Clone)]
struct ServerConfig {
    address: String,
    max_concurrent_work: usize,
    max_body_bytes: usize,
    max_batch_documents: usize,
    max_query_chars: usize,
}

impl ServerConfig {
    fn from_env() -> Result<Self, String> {
        let default_workers = std::thread::available_parallelism()
            .map(|value| value.get().saturating_mul(2))
            .unwrap_or(8)
            .max(4);

        Ok(Self {
            address: env::var("QUERYWEAVE_ADDR").unwrap_or_else(|_| "0.0.0.0:7777".into()),
            max_concurrent_work: env_usize("QUERYWEAVE_MAX_CONCURRENT_WORK", default_workers)?,
            max_body_bytes: env_usize("QUERYWEAVE_MAX_BODY_BYTES", DEFAULT_MAX_BODY_BYTES)?,
            max_batch_documents: env_usize(
                "QUERYWEAVE_MAX_BATCH_DOCUMENTS",
                DEFAULT_MAX_BATCH_DOCUMENTS,
            )?,
            max_query_chars: env_usize("QUERYWEAVE_MAX_QUERY_CHARS", DEFAULT_MAX_QUERY_CHARS)?,
        })
    }
}

fn env_usize(name: &str, default: usize) -> Result<usize, String> {
    match env::var(name) {
        Ok(raw) => raw
            .parse::<usize>()
            .ok()
            .filter(|value| *value > 0)
            .ok_or_else(|| format!("{name} must be a positive integer, got {raw:?}")),
        Err(env::VarError::NotPresent) => Ok(default),
        Err(error) => Err(format!("failed to read {name}: {error}")),
    }
}

#[derive(Clone)]
struct AppState {
    engine: Arc<QueryWeaveEngine>,
    work_slots: Arc<Semaphore>,
    max_concurrent_work: usize,
    max_batch_documents: usize,
    max_query_chars: usize,
}

#[derive(Debug, Deserialize)]
struct UpsertRequest {
    documents: Vec<Document>,
}

#[derive(Debug, Serialize)]
struct UpsertResponse {
    accepted: usize,
    stats: EngineStats,
}

#[derive(Debug, Serialize)]
struct HealthResponse {
    status: &'static str,
    engine: &'static str,
    version: &'static str,
    lexical_backend: String,
    vector_backend: String,
    max_concurrent_work: usize,
    available_work_slots: usize,
}

#[derive(Debug, Serialize)]
struct ErrorEnvelope {
    error: ErrorDetail,
}

#[derive(Debug, Serialize)]
struct ErrorDetail {
    code: &'static str,
    message: String,
}

#[derive(Debug)]
struct ApiError {
    status: StatusCode,
    code: &'static str,
    message: String,
}

impl ApiError {
    fn new(status: StatusCode, code: &'static str, message: impl Into<String>) -> Self {
        Self {
            status,
            code,
            message: message.into(),
        }
    }

    fn busy() -> Self {
        Self::new(
            StatusCode::TOO_MANY_REQUESTS,
            "server_busy",
            "QueryWeave is at its configured work limit; retry with backoff",
        )
    }

    fn worker_failed() -> Self {
        Self::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "worker_failed",
            "QueryWeave worker task failed",
        )
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(ErrorEnvelope {
                error: ErrorDetail {
                    code: self.code,
                    message: self.message,
                },
            }),
        )
            .into_response()
    }
}

fn build_engine() -> Result<QueryWeaveEngine, String> {
    let lexical_name = env::var("QUERYWEAVE_LEXICAL_BACKEND").unwrap_or_else(|_| "tantivy".into());
    let vector_name = env::var("QUERYWEAVE_VECTOR_BACKEND").unwrap_or_else(|_| "hnsw-f32".into());

    let lexical: Box<dyn LexicalRetriever> = match lexical_name.as_str() {
        "builtin" | "bm25" => Box::new(BuiltinBm25Index::default()),
        "tantivy" => Box::new(TantivyLexicalIndex::default()),
        unsupported => {
            return Err(format!(
                "unsupported QUERYWEAVE_LEXICAL_BACKEND={unsupported:?}; expected tantivy, builtin, or bm25"
            ))
        }
    };
    let vector: Box<dyn VectorIndex> = match vector_name.as_str() {
        "exact" => Box::new(ExactVectorIndex::default()),
        "hnsw-i8" | "i8" => Box::new(USearchHnswIndex::i8()),
        "hnsw-f32" | "f32" => Box::new(USearchHnswIndex::f32()),
        unsupported => {
            return Err(format!(
                "unsupported QUERYWEAVE_VECTOR_BACKEND={unsupported:?}; expected hnsw-f32, f32, hnsw-i8, i8, or exact"
            ))
        }
    };

    Ok(QueryWeaveEngine::with_backends(
        lexical,
        vector,
        Box::new(HashEmbedder::default()),
        Box::new(HashSparseEncoder),
    ))
}

#[tokio::main]
async fn main() {
    let config = ServerConfig::from_env()
        .unwrap_or_else(|error| panic!("invalid QueryWeave server configuration: {error}"));
    let engine = build_engine()
        .unwrap_or_else(|error| panic!("invalid QueryWeave backend configuration: {error}"));
    let state = AppState {
        engine: Arc::new(engine),
        work_slots: Arc::new(Semaphore::new(config.max_concurrent_work)),
        max_concurrent_work: config.max_concurrent_work,
        max_batch_documents: config.max_batch_documents,
        max_query_chars: config.max_query_chars,
    };
    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/stats", get(stats))
        .route("/v1/documents:upsert", post(upsert))
        .route("/v1/search", post(search))
        .route("/v1/index", delete(reset))
        .layer(DefaultBodyLimit::max(config.max_body_bytes))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(&config.address)
        .await
        .expect("failed to bind QueryWeave server");
    println!(
        "QueryWeave listening on http://{} (work_limit={}, body_limit={} bytes)",
        config.address, config.max_concurrent_work, config.max_body_bytes
    );
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .expect("QueryWeave server failed");
}

async fn health(State(state): State<AppState>) -> Result<Json<HealthResponse>, ApiError> {
    let stats = engine_stats(Arc::clone(&state.engine)).await?;
    Ok(Json(HealthResponse {
        status: "ok",
        engine: "QueryWeave",
        version: env!("CARGO_PKG_VERSION"),
        lexical_backend: stats.lexical_backend,
        vector_backend: stats.vector_backend,
        max_concurrent_work: state.max_concurrent_work,
        available_work_slots: state.work_slots.available_permits(),
    }))
}

async fn stats(State(state): State<AppState>) -> Result<Json<EngineStats>, ApiError> {
    Ok(Json(engine_stats(Arc::clone(&state.engine)).await?))
}

async fn upsert(
    State(state): State<AppState>,
    Json(payload): Json<UpsertRequest>,
) -> Result<Json<UpsertResponse>, ApiError> {
    if payload.documents.is_empty() {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "empty_batch",
            "documents must not be empty",
        ));
    }
    if payload.documents.len() > state.max_batch_documents {
        return Err(ApiError::new(
            StatusCode::PAYLOAD_TOO_LARGE,
            "batch_too_large",
            format!(
                "document batch contains {} items; configured maximum is {}",
                payload.documents.len(),
                state.max_batch_documents
            ),
        ));
    }
    if payload
        .documents
        .iter()
        .any(|document| document.id.trim().is_empty())
    {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_document_id",
            "document ids must not be empty",
        ));
    }

    let accepted = run_engine_work(&state, move |engine| engine.upsert(payload.documents)).await?;
    let stats = engine_stats(Arc::clone(&state.engine)).await?;
    Ok(Json(UpsertResponse { accepted, stats }))
}

async fn search(
    State(state): State<AppState>,
    Json(payload): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, ApiError> {
    if payload.query.trim().is_empty() {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "empty_query",
            "query must not be empty",
        ));
    }
    let query_chars = payload.query.chars().count();
    if query_chars > state.max_query_chars {
        return Err(ApiError::new(
            StatusCode::PAYLOAD_TOO_LARGE,
            "query_too_large",
            format!(
                "query contains {query_chars} characters; configured maximum is {}",
                state.max_query_chars
            ),
        ));
    }

    Ok(Json(
        run_engine_work(&state, move |engine| engine.search(payload)).await?,
    ))
}

async fn reset(State(state): State<AppState>) -> Result<StatusCode, ApiError> {
    run_engine_work(&state, |engine| engine.reset()).await?;
    Ok(StatusCode::NO_CONTENT)
}

async fn run_engine_work<T, F>(state: &AppState, work: F) -> Result<T, ApiError>
where
    T: Send + 'static,
    F: FnOnce(Arc<QueryWeaveEngine>) -> T + Send + 'static,
{
    let permit = state
        .work_slots
        .clone()
        .try_acquire_owned()
        .map_err(|_| ApiError::busy())?;
    let engine = Arc::clone(&state.engine);

    tokio::task::spawn_blocking(move || {
        let _permit = permit;
        work(engine)
    })
    .await
    .map_err(|_| ApiError::worker_failed())
}

async fn engine_stats(engine: Arc<QueryWeaveEngine>) -> Result<EngineStats, ApiError> {
    tokio::task::spawn_blocking(move || engine.stats())
        .await
        .map_err(|_| ApiError::worker_failed())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_work_limit_scales_with_available_parallelism() {
        let workers = std::thread::available_parallelism()
            .map(|value| value.get().saturating_mul(2))
            .unwrap_or(8)
            .max(4);
        assert!(workers >= 4);
    }
}
