#![forbid(unsafe_code)]

use axum::{extract::State, http::StatusCode, routing::{delete, get, post}, Json, Router};
use queryweave_core::{Document, EngineStats, QueryWeaveEngine, SearchRequest, SearchResponse};
use serde::{Deserialize, Serialize};
use std::{env, sync::Arc};

#[derive(Clone)]
struct AppState { engine: Arc<QueryWeaveEngine> }

#[derive(Debug, Deserialize)]
struct UpsertRequest { documents: Vec<Document> }

#[derive(Debug, Serialize)]
struct UpsertResponse { accepted: usize, stats: EngineStats }

#[derive(Debug, Serialize)]
struct HealthResponse { status: &'static str, engine: &'static str, version: &'static str }

#[tokio::main]
async fn main() {
    let state = AppState { engine: Arc::new(QueryWeaveEngine::new()) };
    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/stats", get(stats))
        .route("/v1/documents:upsert", post(upsert))
        .route("/v1/search", post(search))
        .route("/v1/index", delete(reset))
        .with_state(state);

    let address = env::var("QUERYWEAVE_ADDR").unwrap_or_else(|_| "0.0.0.0:7777".into());
    let listener = tokio::net::TcpListener::bind(&address).await.expect("failed to bind QueryWeave server");
    println!("QueryWeave listening on http://{address}");
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .expect("QueryWeave server failed");
}

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse { status: "ok", engine: "QueryWeave", version: env!("CARGO_PKG_VERSION") })
}

async fn stats(State(state): State<AppState>) -> Json<EngineStats> { Json(state.engine.stats()) }

async fn upsert(State(state): State<AppState>, Json(payload): Json<UpsertRequest>) -> Result<Json<UpsertResponse>, (StatusCode, String)> {
    if payload.documents.is_empty() { return Err((StatusCode::BAD_REQUEST, "documents must not be empty".into())); }
    let accepted = state.engine.upsert(payload.documents);
    Ok(Json(UpsertResponse { accepted, stats: state.engine.stats() }))
}

async fn search(State(state): State<AppState>, Json(payload): Json<SearchRequest>) -> Result<Json<SearchResponse>, (StatusCode, String)> {
    if payload.query.trim().is_empty() { return Err((StatusCode::BAD_REQUEST, "query must not be empty".into())); }
    Ok(Json(state.engine.search(payload)))
}

async fn reset(State(state): State<AppState>) -> StatusCode {
    state.engine.reset();
    StatusCode::NO_CONTENT
}

async fn shutdown_signal() {
    let ctrl_c = async { tokio::signal::ctrl_c().await.expect("failed to install Ctrl+C handler"); };
    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler").recv().await;
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! { _ = ctrl_c => {}, _ = terminate => {} }
}
