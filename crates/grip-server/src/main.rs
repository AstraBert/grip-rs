use std::{
    error::Error,
    fmt::Display,
    fs::File,
    path::PathBuf,
    str::FromStr,
    sync::{Arc, Mutex},
    time::Instant,
};

use actix_web::{
    App, HttpResponse, HttpServer, Responder, ResponseError, body::BoxBody,
    http::header::ContentType, web,
};
use clap::Parser;
use grip_rs::{
    IndexOutput, SearchParams, SparseVector, VectorsAndSketches, Visited, read_index,
    read_sketches, search,
};
use memmap2::Mmap;
use serde::{Deserialize, Serialize};

const INDEX_FILE: &str = "index.bin";
const VECTORS_FILE: &str = "vectors.bin";
const SKETCHES_FILE: &str = "sketches.bin";
const MAX_EF: usize = 500;
const MAX_TOP_K: usize = 100;

#[derive(Debug, Clone, Copy)]
enum LogLevel {
    Error,
    Warn,
    Info,
    Debug,
}

impl Default for LogLevel {
    fn default() -> Self {
        Self::Info
    }
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
    mmap: Arc<Mmap>,
    index_output: IndexOutput,
    sketches: VectorsAndSketches,
    vocab_size: usize,
    scratch: Mutex<Vec<Scratch>>,
}

#[derive(Debug, Clone)]
struct Scratch {
    dense: Vec<u8>,
    dense_rescoring: Vec<f32>,
    visited: Visited,
}

impl Scratch {
    fn new(vocab_size: usize, n_docs: usize) -> Self {
        Self {
            dense: vec![0; vocab_size],
            dense_rescoring: vec![0.0; vocab_size],
            visited: Visited {
                counter: 0,
                visited: vec![0; n_docs],
            },
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
    /// Directory where the index build outputs
    /// have been stored
    #[arg(short, long)]
    index_dir: PathBuf,
    /// Number of server workers
    #[arg(short, long, default_value_t = 4)]
    num_workers: usize,
    /// Log level for the server
    #[arg(short, long, default_value = None)]
    log_level: Option<LogLevel>,
}

#[derive(Serialize, Clone, Copy)]
struct GripInfo {
    vocab_size: usize,
    num_documents: usize,
    scale: f32,
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
        if vec![
            self.top_k,
            self.ef,
            self.max_query_dims,
            self.rescore_factor,
        ]
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

async fn get_info(data: web::Data<ServerState>) -> impl Responder {
    GripInfo {
        vocab_size: data.vocab_size,
        num_documents: data.sketches.offsets.len() - 1,
        scale: data.sketches.scale,
    }
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
        .map_err(|e| SearchError::ValidationError(e))?;

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
            .map_err(|e| SearchError::ValidationError(e))?;
    }

    let Ok(mut scratch) = data.scratch.lock() else {
        return Err(SearchError::MutexError);
    };

    let mut pad = if let Some(s) = scratch.pop() {
        s
    } else {
        Scratch::new(data.vocab_size, data.sketches.offsets.len())
    };

    drop(scratch);

    let start = Instant::now();
    let results = search(
        &request.vector,
        &data.sketches,
        &data.index_output,
        SearchParams {
            patience: request.patience.unwrap(),
            top_k: request.top_k.unwrap(),
            rescore_factor: request.rescore_factor.unwrap(),
            ef: request.ef.unwrap(),
            max_query_dims: request.max_query_dims.unwrap(),
        },
        &mut pad.visited,
        &mut pad.dense,
        &mut pad.dense_rescoring,
        &data.mmap,
    )
    .map_err(|e| SearchError::CoreSearchError(e))?;

    let duration = start.elapsed().as_secs_f32() * 1000_f32;

    let Ok(mut scratch_push) = data.scratch.lock() else {
        return Err(SearchError::MutexError);
    };

    scratch_push.push(pad);

    drop(scratch_push);

    Ok(SearchResponse {
        ids: results.iter().map(|i| i.0).collect(),
        scores: results.iter().map(|i| i.1).collect(),
        search_time: duration,
        total_time: total_start.elapsed().as_secs_f32() * 1000_f32,
    })
}

#[actix_web::main]
async fn main() -> Result<(), String> {
    let args = Args::parse();

    let index = read_index(args.index_dir.join(INDEX_FILE)).map_err(|e| e.to_string())?;
    let vectors_and_sketches = read_sketches(args.index_dir.join(SKETCHES_FILE))?;

    let vectors_file = File::open(args.index_dir.join(VECTORS_FILE)).map_err(|e| e.to_string())?;
    // Safety: the vectors file must not Stringchange while iStringt is mapped
    let mmap = unsafe { Mmap::map(&vectors_file).map_err(|e| e.to_string())? };

    let mut scratches: Vec<Scratch> = Vec::with_capacity(args.num_workers);
    for _ in 0..args.num_workers {
        scratches.push(Scratch::new(
            args.vocab_size,
            vectors_and_sketches.offsets.len(),
        ));
    }

    let state = web::Data::new(ServerState {
        mmap: Arc::new(mmap),
        index_output: index,
        sketches: vectors_and_sketches,
        vocab_size: args.vocab_size,
        scratch: Mutex::new(scratches),
    });

    env_logger::init_from_env(
        env_logger::Env::default()
            .default_filter_or(&args.log_level.unwrap_or_default().to_string()),
    );

    HttpServer::new(move || {
        App::new()
            .app_data(state.clone())
            .route("/health", web::get().to(HttpResponse::Ok))
            .route("/info", web::get().to(get_info))
            .route("/search", web::post().to(post_search))
    })
    .workers(args.num_workers)
    .bind((args.host.as_deref().unwrap_or("localhost"), args.port))
    .map_err(|e| e.to_string())?
    .run()
    .await
    .map_err(|e| e.to_string())
}
