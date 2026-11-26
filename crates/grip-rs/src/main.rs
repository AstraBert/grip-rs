use std::{
    collections::HashSet,
    error::Error,
    fs::{self, File},
    io::{BufReader, Read, Write},
    path::{Path, PathBuf},
    time::Instant,
};

use clap::{Args, Parser, Subcommand};
use grip_rs::{
    BuildParams, MutableIndex, SearchParams, SparseVector, Visited, build_index,
    delta::{MAX_DELTA, persist_delta_insert},
    disk::decode_sparse_vectors_raw,
    save_index, save_vectors_and_sketches, search_all,
    tombstone::delete,
};

const INDEX_FILE: &str = "index.bin";
const VECTORS_FILE: &str = "vectors.bin";
const SKETCHES_FILE: &str = "sketches.bin";

#[derive(Parser)]
#[command(
    name = "grip-rs",
    about = "Graph-based approximate search for sparse vectors"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Build an index from a collection in the Seismic binary format
    Build(BuildArgs),
    /// Search an index with a set of queries
    Search(SearchArgs),
    /// Mark documents of an index as deleted
    Delete(DeleteArgs),
    /// Add new documents to an index, without rebuilding it
    Update(UpdateArgs),
}

#[derive(Args)]
struct BuildArgs {
    /// Path to the documents file (Seismic binary format)
    #[arg(long)]
    documents: PathBuf,
    /// Base directory of the index. The files are written to its `v0` folder
    #[arg(long)]
    output_dir: PathBuf,
    /// Index only the first N documents of the file
    #[arg(long)]
    n_documents: Option<usize>,
    /// Vocabulary size. If not set, it is the largest term id plus one
    #[arg(long)]
    vocab_size: Option<usize>,
    /// Number of hash bits of HashPrune (at most 16)
    #[arg(long, default_value_t = 12)]
    m: usize,
    /// Seed of the random generators
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// Number of neighbors that each point keeps inside a leaf
    #[arg(long, default_value_t = 8)]
    leaf_k: usize,
    /// Alpha value of RobustPrune
    #[arg(long, default_value_t = 1.0)]
    alpha: f32,
    /// Number of seed documents stored for each term
    #[arg(long, default_value_t = 64)]
    entrypoint_seeds: usize,
    /// Number of components kept in the routing sketch of each document
    #[arg(long, default_value_t = 80)]
    sketch_size: usize,
    /// Fanout of the partition at depth 0: each point goes to this many groups
    #[arg(long, default_value_t = 10)]
    fanout_depth0: usize,
    /// Fanout of the partition at depth 1 (deeper levels always use 1)
    #[arg(long, default_value_t = 3)]
    fanout_depth1: usize,
    /// Run a refinement pass: search each point in the built graph and prune the results
    #[arg(long)]
    refine: bool,
    /// Number of terms of a document used to pick the entry points of the refinement search (default 2)
    #[arg(long, requires = "refine")]
    refinement_query_dims: Option<usize>,
    /// Size of the results set of the refinement search (default 100)
    #[arg(long, requires = "refine")]
    refinement_ef: Option<usize>,
    /// Stop the refinement search after this many expansions without a change (default 20)
    #[arg(long, requires = "refine")]
    refinement_patience: Option<usize>,
}

#[derive(Args)]
struct SearchArgs {
    /// Base directory of the index (the one given to `build`)
    #[arg(long)]
    index_dir: PathBuf,
    /// Path to the queries file (Seismic binary format)
    #[arg(long)]
    queries: PathBuf,
    /// Number of results for each query
    #[arg(long, default_value_t = 10)]
    top_k: usize,
    /// Size of the results set during the graph walk (at least top_k)
    #[arg(long, default_value_t = 100)]
    ef: usize,
    /// Stop after this many expansions without a change in the results
    #[arg(long, default_value_t = 20)]
    patience: usize,
    /// Number of query terms used to pick the entry points
    #[arg(long, default_value_t = 2)]
    max_query_dims: usize,
    /// Use only the first N queries
    #[arg(long)]
    n_queries: Option<usize>,
    /// Write the results to this file as `query<TAB>document<TAB>score`
    #[arg(long)]
    output: Option<PathBuf>,
    /// Compute recall against an exact search (slow, for small collections)
    #[arg(long)]
    recall: bool,
    /// Vocabulary size. If not set, it is the largest term id plus one
    #[arg(long)]
    vocab_size: usize,
    /// For rescoring against full-precision vectors, consider top_k x rescore_factor candidates.
    #[arg(long, default_value_t = 5)]
    rescore_factor: usize,
}

#[derive(Args)]
struct DeleteArgs {
    /// Base directory of the index (the one given to `build`)
    #[arg(long)]
    index_dir: PathBuf,
    /// Ids of the documents to delete, separated by commas
    #[arg(long, required = true, value_delimiter = ',')]
    ids: Vec<u32>,
    /// Vocabulary size of the index
    #[arg(long)]
    vocab_size: usize,
}

#[derive(Args)]
struct UpdateArgs {
    /// Base directory of the index (the one given to `build`)
    #[arg(long)]
    index_dir: PathBuf,
    /// Path to the new documents (Seismic binary format)
    #[arg(long)]
    documents: PathBuf,
    /// Vocabulary size of the index. Terms outside of it are rejected
    #[arg(long)]
    vocab_size: usize,
    /// Add only the first N documents of the file
    #[arg(long)]
    n_documents: Option<usize>,
}

fn read_u32(reader: &mut impl Read) -> Result<u32, String> {
    let mut b = [0_u8; 4];
    reader
        .read_exact(&mut b)
        .map_err(|e| format!("cannot read the file: {e}"))?;
    Ok(u32::from_le_bytes(b))
}

/// Reads the Seismic binary format: a `u32` count, then for each vector a `u32`
/// length, that many `u32` term ids, and that many `f32` values (little endian).
/// The id of a vector is its position in the file.
/// The file is streamed, and reading stops after `limit` vectors.
fn read_seismic_bin(path: &Path, limit: Option<usize>) -> Result<Vec<SparseVector>, String> {
    let file = File::open(path).map_err(|e| format!("cannot open {}: {e}", path.display()))?;
    let mut reader = BufReader::with_capacity(1 << 20, file);
    let n = read_u32(&mut reader)? as usize;
    let n = limit.map_or(n, |l| l.min(n));
    let mut vectors = Vec::with_capacity(n);
    // ids and values are stored one after the other, so one read gets both
    let mut buf: Vec<u8> = Vec::new();
    for _ in 0..n {
        let len = read_u32(&mut reader)? as usize;
        buf.resize(len * 8, 0);
        reader
            .read_exact(&mut buf)
            .map_err(|e| format!("cannot read the file: {e}"))?;
        let indices: Vec<u32> = buf[..len * 4]
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let values: Vec<f32> = buf[len * 4..]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        vectors.push(SparseVector::new(indices, values)?);
    }
    Ok(vectors)
}

fn run_build(args: BuildArgs) -> Result<(), Box<dyn Error>> {
    let t = Instant::now();
    let vectors = read_seismic_bin(&args.documents, args.n_documents)?;
    eprintln!("read {} documents in {:.1?}", vectors.len(), t.elapsed());

    let vocab_size = args.vocab_size.unwrap_or_else(|| {
        vectors
            .iter()
            .filter_map(|v| v.indices.last())
            .max()
            .map(|m| *m as usize + 1)
            .unwrap_or(0)
    });

    let params = BuildParams::new(
        args.m,
        args.seed,
        vocab_size,
        args.leaf_k,
        args.refine,
        args.alpha,
        args.entrypoint_seeds,
        args.sketch_size,
        args.fanout_depth0,
        args.fanout_depth1,
        args.refinement_query_dims,
        args.refinement_ef,
        args.refinement_patience,
    )?;

    let version_dir = args.output_dir.join("v0");
    fs::create_dir_all(&version_dir)?;

    let t = Instant::now();
    let (graph, seeds, term_max, sketches, scale) = build_index(&vectors, params)?;
    eprintln!("built the index in {:.1?}", t.elapsed());

    let t = Instant::now();
    save_index(graph, seeds, term_max, version_dir.join(INDEX_FILE))?;
    save_vectors_and_sketches(
        vectors,
        sketches,
        scale,
        version_dir.join(VECTORS_FILE),
        version_dir.join(SKETCHES_FILE),
    )?;
    eprintln!("saved the files in {:.1?}", t.elapsed());

    Ok(())
}

fn run_search(args: SearchArgs) -> Result<(), Box<dyn Error>> {
    let mi = MutableIndex::from_env(&args.index_dir, args.vocab_size, false)?;
    // search results are internal ids: only the file output shows external ids
    let n_docs = mi.ids.len();
    let delta = mi.delta.read().map_err(|e| e.to_string())?;

    let queries = read_seismic_bin(&args.queries, args.n_queries)?;
    if queries.is_empty() {
        return Err("there are no queries to run".into());
    }

    let mut visited = Visited {
        counter: 0,
        visited: vec![0; n_docs],
    };

    // `search` takes the parameters by value, so we build them for each query
    let make_params = || {
        SearchParams::new(
            args.max_query_dims,
            args.patience,
            args.top_k,
            args.ef,
            args.rescore_factor,
        )
    };
    make_params()?; // stop early if the values are not valid

    let mut results = Vec::with_capacity(queries.len());
    let t = Instant::now();
    let mut dense = vec![0; args.vocab_size];
    let mut dense_rescoring = vec![0_f32; args.vocab_size];
    let mut delta_acc = vec![0_f32; MAX_DELTA];
    for q in &queries {
        results.push(search_all(
            q,
            &mi.vectors_and_sketches,
            &mi.index_output,
            &make_params()?,
            &mut visited,
            &mut dense,
            &mut dense_rescoring,
            &mi.vectors_mmap,
            &mi.tombstones,
            n_docs as u32,
            &delta,
            &mut delta_acc,
        )?);
    }
    let elapsed = t.elapsed();
    eprintln!(
        "{} queries in {:.1?} ({:.3} ms per query)",
        queries.len(),
        elapsed,
        elapsed.as_secs_f64() * 1000.0 / queries.len() as f64
    );

    if let Some(path) = &args.output {
        let mut out = std::io::BufWriter::new(File::create(path)?);
        for (qi, found) in results.iter().enumerate() {
            for (doc, score) in found {
                writeln!(out, "{qi}\t{}\t{score}", mi.external_id(*doc))?;
            }
        }
    }

    if args.recall {
        let raw = decode_sparse_vectors_raw(&mi.vectors_mmap, None, false)?;
        let docs = raw
            .iter()
            .map(|v| {
                let indices = v.iter().map(|i| i.0).collect::<Vec<u32>>();
                let values = v.iter().map(|i| i.1).collect::<Vec<f32>>();
                SparseVector::new(indices, values)
            })
            .collect::<Result<Vec<SparseVector>, String>>()?;
        let k = args.top_k;
        let mut total = 0.0_f32;
        // time only the exact search, not the loading of the documents above
        let t = Instant::now();
        for (q, found) in queries.iter().zip(&results) {
            let mut exact: Vec<(f32, u32)> = docs
                .iter()
                .enumerate()
                .filter(|(i, _)| !mi.tombstones.is_deleted(*i as u32))
                .map(|(i, d)| (q.dot(d), i as u32))
                .collect();
            if exact.len() > k {
                exact.select_nth_unstable_by(k - 1, |a, b| b.0.total_cmp(&a.0));
                exact.truncate(k);
            }
            let truth: HashSet<u32> = exact.iter().map(|e| e.1).collect();
            let hits = found.iter().filter(|(id, _)| truth.contains(id)).count();
            total += hits as f32 / k as f32;
        }
        let brute_ms = t.elapsed().as_secs_f64() * 1000.0 / queries.len() as f64;
        let index_ms = elapsed.as_secs_f64() * 1000.0 / queries.len() as f64;
        eprintln!("recall@{k}: {:.4}", total / queries.len() as f32);
        eprintln!(
            "brute force: {brute_ms:.3} ms per query, index: {index_ms:.3} ms per query ({:.0}x faster)",
            brute_ms / index_ms
        );
    }

    Ok(())
}

fn run_delete(args: DeleteArgs) -> Result<(), Box<dyn Error>> {
    let mi = MutableIndex::from_env(&args.index_dir, args.vocab_size, false)?;
    let dir = mi.dir(&args.index_dir);
    let n_delta = mi.delta.read().map_err(|e| e.to_string())?.docs.len();
    let n_docs = mi.ids.len() + n_delta;

    // the user gives external ids: turn them all into internal ids first,
    // so a bad id does not leave a half-applied deletion
    let mut internal_ids = Vec::with_capacity(args.ids.len());
    for id in &args.ids {
        match mi.internal_id(*id, n_delta) {
            Some(internal) => internal_ids.push(internal),
            None => return Err(format!("document id {id} does not exist in the index").into()),
        }
    }

    let (mut deleted, mut already_deleted) = (0, 0);
    for internal in internal_ids {
        if delete(internal, &mi.tombstones, &dir, n_docs)? {
            deleted += 1
        } else {
            already_deleted += 1
        }
    }
    eprintln!("deleted {deleted} documents ({already_deleted} were already deleted)");

    Ok(())
}

fn run_update(args: UpdateArgs) -> Result<(), Box<dyn Error>> {
    let mi = MutableIndex::from_env(&args.index_dir, args.vocab_size, false)?;
    let dir = mi.dir(&args.index_dir);
    let new_docs = read_seismic_bin(&args.documents, args.n_documents)?;

    // check everything first, so a bad document does not leave a half-applied update
    if let Some(term) = new_docs
        .iter()
        .flat_map(|v| v.indices.iter())
        .find(|&&t| t as usize >= args.vocab_size)
    {
        return Err(format!(
            "term id {term} is out of range: the vocabulary size is {}",
            args.vocab_size
        )
        .into());
    }

    let existing = mi.delta.read().map_err(|e| e.to_string())?.docs.len();
    if existing + new_docs.len() > MAX_DELTA {
        return Err(format!(
            "the delta holds {existing} documents and the limit is {MAX_DELTA}: cannot add {} more",
            new_docs.len()
        )
        .into());
    }

    let t = Instant::now();
    let mut ids: Vec<u32> = Vec::with_capacity(new_docs.len());
    for v in new_docs {
        ids.push(persist_delta_insert(
            v.indices,
            v.values,
            &mi.delta,
            &dir,
            mi.ext_base,
        )?);
    }
    match (ids.first(), ids.last()) {
        (Some(first), Some(last)) => eprintln!(
            "added {} documents with ids {first}..={last} in {:.1?}",
            ids.len(),
            t.elapsed()
        ),
        _ => eprintln!("there are no documents to add"),
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    match Cli::parse().command {
        Command::Build(args) => run_build(args),
        Command::Search(args) => run_search(args),
        Command::Delete(args) => run_delete(args),
        Command::Update(args) => run_update(args),
    }
}
