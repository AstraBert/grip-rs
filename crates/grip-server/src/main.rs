use std::{
    error::Error,
    fmt::Display,
    io,
    path::PathBuf,
    str::FromStr,
    sync::{
        Arc, Mutex, RwLock,
        mpsc::{self, RecvTimeoutError},
    },
    thread,
    time::{Duration, Instant},
};

use actix_web::{
    App, HttpResponse, HttpServer, Responder, ResponseError,
    body::BoxBody,
    http::header::ContentType,
    web::{self, Json},
};
use clap::Parser;
use grip_rs::{
    CleanupParams, MutableIndex, SearchParams, SparseVector, Visited, cleanup,
    delta::{DeltaError, MAX_DELTA, persist_delta_insert},
    search_all,
    tombstone::delete as tombstone_deletion,
};
use serde::{Deserialize, Serialize};

const MAX_EF: usize = 500;
const MAX_TOP_K: usize = 100;
const CLEANUP_THRESHOLD: f64 = 0.5;

#[derive(Debug, Clone, Copy)]
#[derive(Default)]
enum LogLevel {
    Error,
    Warn,
    #[default]
    Info,
    Debug,
}


impl FromStr for LogLevel {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "info" => Ok(Self::Info),
            "warn" => Ok(Self::Warn),
            "debug" => Ok(Self::Debug),
            "error" => Ok(Self::Error),
            _ => Err(format!("Unrecognized log level: {s}")),
        }
    }
}

impl Display for LogLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Debug => write!(f, "debug"),
            Self::Error => write!(f, "error"),
            Self::Info => write!(f, "info"),
            Self::Warn => write!(f, "warn"),
        }
    }
}

#[derive(Debug)]
struct ServerState {
    /// The index in use. A cleanup swaps it in place.
    index: Arc<RwLock<MutableIndex>>,
    /// Base directory: the versions `v0`, `v1`, ... live here
    base_dir: PathBuf,
    vocab_size: usize,
    scratch: Mutex<Vec<Scratch>>,
}

#[derive(Debug, Clone)]
struct Scratch {
    /// Version of the index that `visited` was sized for
    version: usize,
    dense: Vec<u8>,
    dense_rescoring: Vec<f32>,
    delta_acc: Vec<f32>,
    visited: Visited,
}

impl Scratch {
    fn new(vocab_size: usize, n_docs: usize, version: usize) -> Self {
        Self {
            version,
            dense: vec![0; vocab_size],
            dense_rescoring: vec![0.0; vocab_size],
            visited: Visited {
                counter: 0,
                visited: vec![0; n_docs],
            },
            delta_acc: vec![0.0; MAX_DELTA],
        }
    }
}

#[derive(Debug, Parser)]
struct Args {
    /// Host to bind the server to.
    /// Defaults to localhost.
    #[arg(long, default_value = None)]
    host: Option<String>,
    /// Port to run the server on.
    /// Defaults to 7433.
    #[arg(short, long, default_value_t = 7433)]
    port: u16,
    /// Size of the vocabulary for the model
    /// that generated the embeddings
    #[arg(short, long)]
    vocab_size: usize,
    /// Base directory of the index: it holds the
    /// versions `v0`, `v1`, ... and the latest one is served
    #[arg(short, long)]
    index_dir: PathBuf,
    /// Number of server workers
    #[arg(short, long, default_value_t = 4)]
    num_workers: usize,
    /// Log level for the server
    #[arg(short, long, default_value = None)]
    log_level: Option<LogLevel>,
    #[command(flatten)]
    cleanup: CleanupArgs,
}

/// Settings of the background cleanup (heal + compaction).
/// The vocabulary size is the one of the server.
#[derive(Debug, Clone, clap::Args)]
struct CleanupArgs {
    /// Alpha value of RobustPrune
    #[arg(long, default_value_t = 1.0)]
    cleanup_alpha: f32,
    /// Maximum number of nodes visited by the healer around a deleted node
    #[arg(long, default_value_t = 100)]
    cleanup_max_dfs: usize,
    /// Maximum size of the border the healer collects around a deleted node
    #[arg(long, default_value_t = 200)]
    cleanup_max_border_length: usize,
    /// Number of components kept in the routing sketch of a delta document
    #[arg(long, default_value_t = 80)]
    cleanup_sketch_size: usize,
    /// Size of the results set when a delta document is inserted in the graph
    #[arg(long, default_value_t = 100)]
    cleanup_ef: usize,
    /// Stop that search after this many expansions without a change
    #[arg(long, default_value_t = 20)]
    cleanup_patience: usize,
    /// Number of terms of a document used to pick the entry points of that search
    #[arg(long, default_value_t = 2)]
    cleanup_max_query_dims: usize,
    /// Number of seed documents stored for each term
    #[arg(long, default_value_t = 64)]
    cleanup_entrypoint_seeds: usize,
}

impl CleanupArgs {
    fn to_params(&self, vocab_size: usize) -> CleanupParams {
        CleanupParams {
            robust_prune_alpha: self.cleanup_alpha,
            max_dfs: self.cleanup_max_dfs,
            max_border_length: self.cleanup_max_border_length,
            vocab_size,
            sketch_size: self.cleanup_sketch_size,
            ef: self.cleanup_ef,
            patience: self.cleanup_patience,
            max_query_dims: self.cleanup_max_query_dims,
            entrypoint_seeds: self.cleanup_entrypoint_seeds,
        }
    }
}

#[derive(Serialize, Clone, Copy)]
struct GripInfo {
    vocab_size: usize,
    num_documents: usize,
    scale: f32,
    deleted_ratio: f64,
}

#[derive(Debug, Deserialize, Clone)]
struct SearchRequest {
    vector: SparseVector,
    #[serde(default)]
    max_query_dims: Option<usize>,
    #[serde(default)]
    patience: Option<usize>,
    #[serde(default)]
    top_k: Option<usize>,
    #[serde(default)]
    ef: Option<usize>,
    #[serde(default)]
    rescore_factor: Option<usize>,
}

impl SearchRequest {
    fn resolve(&mut self) {
        if self.max_query_dims.is_none() {
            self.max_query_dims = Some(2);
        }
        if self.patience.is_none() {
            self.patience = Some(20);
        }
        if self.top_k.is_none() {
            self.top_k = Some(10);
        }
        if self.ef.is_none() {
            self.ef = Some(100);
        }
        if self.rescore_factor.is_none() {
            self.rescore_factor = Some(5);
        }
    }

    // always call after resolve()
    fn validate(&self) -> Result<(), String> {
        if [self.top_k,
            self.ef,
            self.max_query_dims,
            self.rescore_factor]
        .iter()
        .any(|v| v.is_none())
        {
            return Err("validate() can only be called after resolve()".to_string());
        }

        let max_query_dims = self.max_query_dims.unwrap();
        let top_k = self.top_k.unwrap();
        let ef = self.ef.unwrap();
        let rescore_factor = self.rescore_factor.unwrap();

        if max_query_dims == 0 {
            return Err("max_query_dims needs to be at least 1".to_string());
        }
        if top_k == 0 {
            return Err("top_k needs to be at least 1".to_string());
        }
        if ef == 0 {
            return Err("ef needs to be at least 1".to_string());
        }
        if ef < top_k {
            return Err("ef cannot be less than top_k".to_string());
        }
        if rescore_factor == 0 {
            return Err("rescord_factor cannot be 0".to_string());
        }
        if ef > MAX_EF {
            return Err(format!("ef is capped at {MAX_EF}"));
        }
        if top_k > MAX_TOP_K {
            return Err(format!("top_k is capped at {MAX_TOP_K}"));
        }

        Ok(())
    }
}

impl Responder for GripInfo {
    type Body = BoxBody;

    fn respond_to(self, _req: &actix_web::HttpRequest) -> HttpResponse<Self::Body> {
        let body = serde_json::to_string(&self).unwrap();

        // Create response and set content type
        HttpResponse::Ok()
            .content_type(ContentType::json())
            .body(body)
    }
}

#[derive(Debug, Serialize)]
struct SearchResponse {
    ids: Vec<u32>,
    scores: Vec<f32>,
    search_time: f32,
    total_time: f32,
}

impl Responder for SearchResponse {
    type Body = BoxBody;

    fn respond_to(self, _req: &actix_web::HttpRequest) -> HttpResponse<Self::Body> {
        let body = serde_json::to_string(&self).unwrap();

        // Create response and set content type
        HttpResponse::Ok()
            .content_type(ContentType::json())
            .body(body)
    }
}

type InfoError = UpdateError;

async fn get_info(data: web::Data<ServerState>) -> Result<GripInfo, InfoError> {
    let mi = data.index.read().map_err(|_| InfoError::Mutex)?;
    let delta_m = mi.delta.read().map_err(|_| InfoError::Mutex)?;
    let n_docs = mi.ids.len() + delta_m.docs.len();
    drop(delta_m);

    Ok(GripInfo {
        vocab_size: data.vocab_size,
        num_documents: n_docs,
        scale: mi.vectors_and_sketches.scale,
        deleted_ratio: mi.tombstones.ratio(n_docs),
    })
}

#[derive(Debug)]
enum SearchError {
    MutexError,
    ValidationError(String),
    CoreSearchError(String),
}

impl Display for SearchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MutexError => write!(f, "Mutex lock is poisoned"),
            Self::ValidationError(s) => write!(f, "Validation error: {}", s),
            Self::CoreSearchError(s) => write!(f, "Error while searching: {}", s),
        }
    }
}

impl Error for SearchError {}

impl ResponseError for SearchError {
    fn error_response(&self) -> HttpResponse<BoxBody> {
        HttpResponse::build(self.status_code())
            .insert_header(ContentType::plaintext())
            .body(self.to_string())
    }

    fn status_code(&self) -> actix_web::http::StatusCode {
        match self {
            Self::MutexError => actix_web::http::StatusCode::INTERNAL_SERVER_ERROR,
            Self::CoreSearchError(_) => actix_web::http::StatusCode::INTERNAL_SERVER_ERROR,
            Self::ValidationError(_) => actix_web::http::StatusCode::BAD_REQUEST,
        }
    }
}

async fn post_search(
    request: web::Json<SearchRequest>,
    data: web::Data<ServerState>,
) -> Result<SearchResponse, SearchError> {
    let total_start = Instant::now();

    let mut request = request.into_inner();

    request.resolve();
    request
        .validate()
        .map_err(SearchError::ValidationError)?;

    if request
        .vector
        .indices
        .iter()
        .any(|t| *t as usize >= data.vocab_size)
    {
        return Err(SearchError::ValidationError(
            "The query vector contains terms outside of the vocabulary of the model".to_string(),
        ));
    }

    if !request.vector.is_optimal() {
        request
            .vector
            .make_optimal()
            .map_err(SearchError::ValidationError)?;
    }

    // the read lock is held for the whole search: a cleanup waits for it
    // before swapping the index
    let mi = data.index.read().map_err(|_| SearchError::MutexError)?;
    let n_main = mi.ids.len();

    let Ok(mut scratch) = data.scratch.lock() else {
        return Err(SearchError::MutexError);
    };

    let mut pad = match scratch.pop() {
        // a scratch made for another version has a wrong size
        Some(s) if s.version == mi.version => s,
        _ => Scratch::new(data.vocab_size, n_main, mi.version),
    };

    drop(scratch);

    let delta_m = mi.delta.read().map_err(|_| SearchError::MutexError)?;

    let start = Instant::now();
    let results = search_all(
        &request.vector,
        &mi.vectors_and_sketches,
        &mi.index_output,
        &SearchParams {
            patience: request.patience.unwrap(),
            top_k: request.top_k.unwrap(),
            rescore_factor: request.rescore_factor.unwrap(),
            ef: request.ef.unwrap(),
            max_query_dims: request.max_query_dims.unwrap(),
        },
        &mut pad.visited,
        &mut pad.dense,
        &mut pad.dense_rescoring,
        &mi.vectors_mmap,
        &mi.tombstones,
        n_main as u32,
        &delta_m,
        &mut pad.delta_acc,
    )
    .map_err(SearchError::CoreSearchError)?;

    drop(delta_m);

    let duration = start.elapsed().as_secs_f32() * 1000_f32;

    let Ok(mut scratch_push) = data.scratch.lock() else {
        return Err(SearchError::MutexError);
    };

    scratch_push.push(pad);

    drop(scratch_push);

    // the search gives internal ids: the user sees external ids
    Ok(SearchResponse {
        ids: results.iter().map(|i| mi.external_id(i.0)).collect(),
        scores: results.iter().map(|i| i.1).collect(),
        search_time: duration,
        total_time: total_start.elapsed().as_secs_f32() * 1000_f32,
    })
}

#[derive(Debug)]
pub enum DeletionError {
    NotFound,
    IOError(String),
    MutexError,
}

impl Display for DeletionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound => write!(f, "Point not found in the collection"),
            Self::IOError(s) => write!(f, "IO Error when deleting: {}", s),
            Self::MutexError => write!(f, "Mutext holding delta is poisoned"),
        }
    }
}

impl Error for DeletionError {}

impl ResponseError for DeletionError {
    fn error_response(&self) -> HttpResponse<BoxBody> {
        HttpResponse::build(self.status_code())
            .insert_header(ContentType::plaintext())
            .body(self.to_string())
    }

    fn status_code(&self) -> actix_web::http::StatusCode {
        match self {
            Self::NotFound => actix_web::http::StatusCode::NOT_FOUND,
            Self::IOError(_) => actix_web::http::StatusCode::INTERNAL_SERVER_ERROR,
            Self::MutexError => actix_web::http::StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}

async fn delete_point(
    path: web::Path<u32>,
    data: web::Data<ServerState>,
) -> Result<HttpResponse, DeletionError> {
    let mi = data.index.read().map_err(|_| DeletionError::MutexError)?;
    let delta_m = mi.delta.read().map_err(|_| DeletionError::MutexError)?;
    let n_delta = delta_m.docs.len();
    drop(delta_m);
    // the user gives an external id
    let internal = mi
        .internal_id(path.into_inner(), n_delta)
        .ok_or(DeletionError::NotFound)?;
    let res = tombstone_deletion(
        internal,
        &mi.tombstones,
        &mi.dir(&data.base_dir),
        mi.ids.len() + n_delta,
    );
    match res {
        Err(e) => match e.kind() {
            io::ErrorKind::NotFound => return Err(DeletionError::NotFound),
            _ => return Err(DeletionError::IOError(e.to_string())),
        },
        Ok(d) => {
            if !d {
                return Ok(HttpResponse::NoContent().finish());
            }
        }
    }

    Ok(HttpResponse::Accepted().finish())
}

#[derive(Debug, Clone, Deserialize)]
struct UpdateRequest {
    indices: Vec<u32>,
    values: Vec<f32>,
}

#[derive(Debug)]
enum UpdateError {
    Validation(String),
    IsFull,
    IO(String),
    Mutex,
}

impl Display for UpdateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Validation(e) => write!(f, "Validation error: {}", e),
            Self::IsFull => write!(f, "Delta is full"),
            Self::IO(e) => write!(f, "IO error: {}", e),
            Self::Mutex => write!(f, "Mutex holding the delta is poisoned"),
        }
    }
}

impl Error for UpdateError {}

#[derive(Debug, Serialize)]
struct UpdateResponse {
    external_id: u32,
}

impl Responder for UpdateResponse {
    type Body = BoxBody;

    fn respond_to(self, _req: &actix_web::HttpRequest) -> HttpResponse<Self::Body> {
        let body = serde_json::to_string(&self).unwrap();

        // Create response and set content type
        HttpResponse::Created()
            .content_type(ContentType::json())
            .body(body)
    }
}

impl ResponseError for UpdateError {
    fn error_response(&self) -> HttpResponse<BoxBody> {
        HttpResponse::build(self.status_code())
            .insert_header(ContentType::plaintext())
            .body(self.to_string())
    }

    fn status_code(&self) -> actix_web::http::StatusCode {
        match self {
            Self::IsFull => actix_web::http::StatusCode::INSUFFICIENT_STORAGE,
            Self::IO(_) => actix_web::http::StatusCode::INTERNAL_SERVER_ERROR,
            Self::Mutex => actix_web::http::StatusCode::INTERNAL_SERVER_ERROR,
            Self::Validation(_) => actix_web::http::StatusCode::BAD_REQUEST,
        }
    }
}

impl From<DeltaError> for UpdateError {
    fn from(value: DeltaError) -> Self {
        match value {
            DeltaError::IO(e) => Self::IO(e),
            DeltaError::IsFull => Self::IsFull,
            DeltaError::Mutex => Self::Mutex,
            DeltaError::Validation(e) => Self::Validation(e),
        }
    }
}

async fn post_point(
    request: Json<UpdateRequest>,
    data: web::Data<ServerState>,
) -> Result<UpdateResponse, UpdateError> {
    if request
        .indices
        .iter()
        .any(|t| *t as usize >= data.vocab_size)
    {
        return Err(UpdateError::Validation(
            "The vector to insert contains terms outside of the vocabulary of the model"
                .to_string(),
        ));
    }

    let req = request.into_inner();
    let mi = data.index.read().map_err(|_| UpdateError::Mutex)?;
    let ext_id = persist_delta_insert(
        req.indices,
        req.values,
        &mi.delta,
        &mi.dir(&data.base_dir),
        mi.ext_base,
    )?;

    Ok(UpdateResponse {
        external_id: ext_id,
    })
}

#[actix_web::main]
async fn main() -> Result<(), String> {
    let args = Args::parse();

    // the latest version in the base directory. Leftover `.tmp` folders
    // of a cleanup that never finished are deleted
    let index = MutableIndex::from_env(&args.index_dir, args.vocab_size, true)?;

    let mut scratches: Vec<Scratch> = Vec::with_capacity(args.num_workers);
    for _ in 0..args.num_workers {
        scratches.push(Scratch::new(
            args.vocab_size,
            index.ids.len(),
            index.version,
        ));
    }

    let mutable_index_arc = Arc::new(RwLock::new(index));

    let state = web::Data::new(ServerState {
        index: mutable_index_arc.clone(),
        base_dir: args.index_dir.clone(),
        vocab_size: args.vocab_size,
        scratch: Mutex::new(scratches),
    });

    env_logger::init_from_env(
        env_logger::Env::default()
            .default_filter_or(args.log_level.unwrap_or_default().to_string()),
    );

    let cleanup_params = args.cleanup.to_params(args.vocab_size);
    let cleanup_dir = args.index_dir.clone();
    let (tx, rx) = mpsc::channel::<()>();

    let handle = thread::spawn(move || {
        loop {
            let mi = Arc::clone(&mutable_index_arc);
            match rx.recv_timeout(Duration::from_secs(10)) {
                Ok(_) | Err(RecvTimeoutError::Disconnected) => break,
                Err(RecvTimeoutError::Timeout) => {
                    let mi_read = match mi.read() {
                        Ok(r) => r,
                        Err(e) => {
                            log::error!(
                                "[CLEANUP THREAD] Could not acquire read lock for mutable index: {e}"
                            );
                            continue;
                        }
                    };
                    let delta_m = match mi_read.delta.read() {
                        Ok(r) => r,
                        Err(e) => {
                            log::error!(
                                "[CLEANUP THREAD] Could not acquire read lock for delta: {e}"
                            );
                            continue;
                        }
                    };
                    let delta_len = delta_m.docs.len();
                    let n_docs = mi_read.ids.len() + delta_len;
                    drop(delta_m);
                    if mi_read.tombstones.ratio(n_docs) < CLEANUP_THRESHOLD
                        || delta_len >= MAX_DELTA
                    {
                        drop(mi_read);
                        continue;
                    }
                    drop(mi_read);
                    match cleanup(mi, &cleanup_dir, &cleanup_params) {
                        Ok(()) => continue,
                        Err(e) => {
                            log::error!("[CLEANUP THREAD] Error while executing cleanup: {e}");
                            continue;
                        }
                    }
                }
            }
        }
    });

    HttpServer::new(move || {
        App::new()
            .app_data(state.clone())
            .route("/health", web::get().to(HttpResponse::Ok))
            .route("/info", web::get().to(get_info))
            .route("/search", web::post().to(post_search))
            .route("/points/{id}", web::delete().to(delete_point))
            .route("/points", web::post().to(post_point))
    })
    .workers(args.num_workers)
    .bind((args.host.as_deref().unwrap_or("localhost"), args.port))
    .map_err(|e| e.to_string())?
    .run()
    .await
    .map_err(|e| e.to_string())?;

    drop(tx);

    handle.join().map_err(|e| match e.downcast::<String>() {
        Ok(s) => s.to_string(),
        Err(_) => "An unknown error has occurred while joining the background thread".to_string(),
    })
}
