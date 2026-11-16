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
    BuildParams, SearchParams, SparseVector, Visited, build_index, disk::decode_sparse_vectors_raw,
    read_index, read_sketches, robustprune::MAX_GRAPH_DEGREE, save_index,
    save_vectors_and_sketches, search,
};
use memmap2::Mmap;

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
}

#[derive(Args)]
struct BuildArgs {
    /// Path to the documents file (Seismic binary format)
    #[arg(long)]
    documents: PathBuf,
    /// Directory where the index files are written
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
}

#[derive(Args)]
struct SearchArgs {
    /// Directory that holds the index files written by `build`
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

    // `refine` is not implemented yet, so it is always false
    let params = BuildParams::new(
        args.m,
        args.seed,
        vocab_size,
        args.leaf_k,
        false,
        args.alpha,
        args.entrypoint_seeds,
        args.sketch_size,
    )?;

    fs::create_dir_all(&args.output_dir)?;

    let t = Instant::now();
    // the vectors are cloned because `build_index` takes ownership of them
    let (graph, seeds, term_max, sketches, scale) = build_index(vectors.clone(), params)?;
    eprintln!("built the index in {:.1?}", t.elapsed());

    let t = Instant::now();
    save_index(graph, seeds, term_max, args.output_dir.join(INDEX_FILE))?;
    save_vectors_and_sketches(
        vectors,
        sketches,
        scale,
        args.output_dir.join(VECTORS_FILE),
        args.output_dir.join(SKETCHES_FILE),
    )?;
    eprintln!("saved the files in {:.1?}", t.elapsed());

    Ok(())
}

fn run_search(args: SearchArgs) -> Result<(), Box<dyn Error>> {
    let index = read_index(args.index_dir.join(INDEX_FILE))?;
    let vectors_and_sketches = read_sketches(args.index_dir.join(SKETCHES_FILE))?;

    let vectors_file = File::open(args.index_dir.join(VECTORS_FILE))?;
    // Safety: the vectors file must not change while it is mapped
    let mmap = unsafe { Mmap::map(&vectors_file)? };

    let queries = read_seismic_bin(&args.queries, args.n_queries)?;
    if queries.is_empty() {
        return Err("there are no queries to run".into());
    }

    let n_docs = index.graph.len() / MAX_GRAPH_DEGREE;
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
    for q in &queries {
        results.push(search(
            q,
            &vectors_and_sketches,
            &index,
            make_params()?,
            &mut visited,
            &mut dense,
            &mut dense_rescoring,
            &mmap,
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
                writeln!(out, "{qi}\t{doc}\t{score}")?;
            }
        }
    }

    if args.recall {
        let raw = decode_sparse_vectors_raw(&mmap, None)?;
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

fn main() -> Result<(), Box<dyn Error>> {
    match Cli::parse().command {
        Command::Build(args) => run_build(args),
        Command::Search(args) => run_search(args),
    }
}
