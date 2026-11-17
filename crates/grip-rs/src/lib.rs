use std::{collections::BTreeMap, fs, io, path::Path, sync::Mutex};

use crate::{
    disk::{decode_sparse_sketches, encode_vectors},
    hashprune::{Reservoir, insert_in_reservoir, make_projection, make_sketch},
    leaf::build_leaf_edges,
    partition::partition,
    robustprune::{MAX_GRAPH_DEGREE, robust_prune},
    seeding::{SeedVecs, build_seed_list, encode_seed_vecs, read_seed_vec_from_bytes},
    timing::{CpuClock, Phases},
};
use half::bf16;
use rayon::prelude::*;

pub mod disk;
pub mod hashprune;
pub mod leaf;
pub mod partition;
pub mod robustprune;
mod search;
pub mod seeding;
mod timing;
mod types;

pub use search::*;
pub use types::*;

pub type IndexBuildOutput = Result<
    (
        Vec<u32>,
        SeedVecs,
        BTreeMap<u32, f32>,
        Vec<QuantizedSparseVector>,
        f32,
    ),
    String,
>;

struct Scratch {
    visited: Visited,
    dense: Vec<u8>,
    dense_prune: Vec<f32>,
}

impl Scratch {
    fn new(vocab_size: usize, n_docs: usize) -> Self {
        Self {
            visited: Visited {
                counter: 0,
                visited: vec![0_u32; n_docs],
            },
            dense: vec![0_u8; vocab_size],
            dense_prune: vec![0_f32; vocab_size],
        }
    }
}

fn refine(
    graph: &[u32],
    vectors: &[SparseVector],
    sketches: &[QuantizedSparseVector],
    seeds: &SeedVecs,
    term_max: &BTreeMap<u32, f32>,
    params: &BuildParams,
    scale: f32,
) -> Result<Vec<u32>, String> {
    let capacity: usize = sketches.iter().map(|v| v.len()).sum();

    let mut offsets: Vec<usize> = sketches
        .iter()
        .scan(0, |running, v| {
            let start = *running;
            *running += v.len();
            Some(start)
        })
        .collect();

    offsets.push(capacity);

    let mut indices = Vec::with_capacity(capacity);
    let mut values = Vec::with_capacity(capacity);

    for v in sketches {
        indices.extend_from_slice(&v.indices);
        values.extend_from_slice(&v.values);
    }

    let vectors_and_sketches = &VectorsAndSketches {
        offsets,
        indices,
        values,
        scale,
    };

    // built once, not for every point
    let borrowed_vectors = vectors.iter().collect::<Vec<&SparseVector>>();

    let mut graph_new = vec![u32::MAX; vectors.len() * MAX_GRAPH_DEGREE];
    graph_new
        .par_chunks_mut(MAX_GRAPH_DEGREE)
        .enumerate()
        .try_for_each_init(
            || Scratch::new(params.vocab_size, sketches.len()),
            |scratch, (p, row)| -> Result<(), String> {
                let me = p as u32;

                // the beam search reads the query from `dense`,
                // and resets it before returning
                for (t, w) in sketches[p].indices.iter().zip(sketches[p].values.iter()) {
                    scratch.dense[*t as usize] = *w;
                }

                // skip the padding of the row
                let neighbours: Vec<u32> = graph[p * MAX_GRAPH_DEGREE..(p + 1) * MAX_GRAPH_DEGREE]
                    .iter()
                    .copied()
                    .filter(|&nb| nb != u32::MAX)
                    .collect();

                let mut by_impact = sketches[p].clone();
                let mut starts = compute_entry_points(
                    &mut by_impact,
                    seeds,
                    term_max,
                    params.refinement_query_dims.unwrap(),
                );
                starts.extend_from_slice(&neighbours);
                // the point itself is not a start
                // (the beam search sorts and dedups the starts)
                starts.retain(|&s| s != me);

                let found = run_beam_search(
                    &sketches[p],
                    &mut scratch.visited,
                    &mut starts,
                    vectors_and_sketches,
                    &mut scratch.dense,
                    graph,
                    params.refinement_ef.unwrap(),
                    params.refinement_patience.unwrap(),
                )?;

                let mut pool = found.iter().map(|v| v.1).collect::<Vec<u32>>();
                pool.extend_from_slice(&neighbours);
                // dedup only removes equal neighbours, so sort first
                pool.sort_unstable();
                pool.dedup();
                // a point is not its own neighbour
                pool.retain(|&id| id != me);

                let exact_scores = pool
                    .iter()
                    .map(|id| {
                        let score = vectors[p].dot(&vectors[*id as usize]);
                        (*id, score)
                    })
                    .collect::<Vec<(u32, f32)>>();
                let pruned = robust_prune::<f32>(
                    exact_scores,
                    params.robust_prune_alpha,
                    &borrowed_vectors,
                    &mut scratch.dense_prune,
                );
                row.copy_from_slice(&pruned);
                Ok(())
            },
        )?;

    Ok(graph_new)
}

pub fn build_index(vectors: &[SparseVector], params: BuildParams) -> IndexBuildOutput {
    let mut phases = Phases::new("build_index");

    let w = make_projection(params.vocab_size, params.m, params.seed);
    let sketches = vectors
        .par_iter()
        .map(|v| make_sketch(&v, &w, params.m))
        .collect::<Vec<Vec<f32>>>();

    let borrowed_sketches = sketches
        .iter()
        .map(|v| v.as_slice())
        .collect::<Vec<&[f32]>>();

    let reservoirs = (0..vectors.len())
        .map(|_| Mutex::new(Reservoir::default()))
        .collect::<Vec<Mutex<Reservoir>>>();

    let borrowed_vectors = vectors.iter().collect::<Vec<&SparseVector>>();
    phases.mark("projection, sketches and reservoirs");

    let leaves = partition(
        &borrowed_vectors,
        &(0..vectors.len()).map(|i| i as u32).collect::<Vec<u32>>(),
        0,
        params.seed,
        params.fanouts,
    );
    phases.mark("partition");
    phases.note(|| {
        let largest = leaves.iter().map(|l| l.len()).max().unwrap_or(0);
        let memberships: usize = leaves.iter().map(|l| l.len()).sum();
        format!(
            "{} leaves, {memberships} memberships, largest leaf {largest} (its score matrix is {} MiB)",
            leaves.len(),
            largest * largest * 4 / (1 << 20),
        )
    });

    // these add up the time of all threads, so they can exceed the wall time
    let scoring_clock = CpuClock::default();
    let insert_clock = CpuClock::default();

    leaves
        .par_iter()
        .map(|leaf| {
            let edges = scoring_clock.time(|| {
                let mut scores = vec![0.0_f32; leaf.len() * leaf.len()];
                let mut triples = Vec::with_capacity(leaf.len());
                build_leaf_edges(
                    leaf,
                    params.leaf_k,
                    &mut scores,
                    &borrowed_vectors,
                    &mut triples,
                )
            });
            insert_clock.time(|| -> Result<(), String> {
                for (src, dst, score) in edges {
                    let mut res = reservoirs[src as usize].lock().map_err(|e| e.to_string())?;
                    insert_in_reservoir(
                        &mut res,
                        src,
                        dst,
                        bf16::from_f32(score),
                        &borrowed_sketches,
                        params.m,
                    )
                    .map_err(|e| e.to_string())?;
                    drop(res);
                }
                Ok(())
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    phases.mark("leaf scoring and reservoir inserts");
    phases.note(|| {
        format!(
            "thread time, summed over all threads: leaf scoring {:.1}s, reservoir inserts {:.1}s",
            scoring_clock.seconds(),
            insert_clock.seconds()
        )
    });

    let mut graph = vec![0u32; vectors.len() * MAX_GRAPH_DEGREE];

    graph
        .par_chunks_mut(MAX_GRAPH_DEGREE)
        .enumerate()
        .try_for_each_init(
            || vec![0_f32; params.vocab_size],
            |dense, (p, row)| -> Result<(), String> {
                let candidates = reservoirs[p].lock().map_err(|e| e.to_string())?;
                let kept = robust_prune::<half::bf16>(
                    candidates.slots.iter().map(|s| (s.id, s.score)).collect(),
                    params.robust_prune_alpha,
                    &borrowed_vectors,
                    dense,
                );
                row.copy_from_slice(&kept); // panics if kept.len() != MAX_GRAPH_DEGREE
                Ok(())
            },
        )?;
    phases.mark("robust prune");

    let (seeds, term_max) = build_seed_list(&borrowed_vectors, params.entrypoint_seeds);
    phases.mark("seed lists");

    let mut max_w = f32::MIN;
    borrowed_vectors
        .iter()
        .map(|v| {
            for w in v.values.iter() {
                max_w = max_w.max(*w);
            }
        })
        .for_each(drop);
    let scale = max_w / 255_f32;

    let routing_sketches = borrowed_vectors
        .iter()
        .map(|v| make_routing_sketch(v, params.sketch_size, scale))
        .collect::<Result<Vec<QuantizedSparseVector>, String>>()?;
    phases.mark("routing sketches");

    if params.refine {
        graph = refine(
            &graph,
            vectors,
            &routing_sketches,
            &seeds,
            &term_max,
            &params,
            scale,
        )?;

        phases.mark("refine");
    }

    Ok((graph, seeds, term_max, routing_sketches, scale))
}

pub fn save_vectors_and_sketches(
    vectors: Vec<SparseVector>,
    sketches: Vec<QuantizedSparseVector>,
    scale: f32,
    dest_vectors: impl AsRef<Path>,
    dest_sketches: impl AsRef<Path>,
) -> io::Result<()> {
    let mut phases = Phases::new("save_vectors_and_sketches");
    let vectors_encoded = encode_vectors(&vectors.iter().collect::<Vec<&SparseVector>>(), 4, None);
    phases.mark("encode vectors");
    let sketches_encoded = encode_vectors(
        &sketches.iter().collect::<Vec<&QuantizedSparseVector>>(),
        1,
        Some(scale),
    );
    phases.mark("encode sketches");
    fs::write(dest_vectors, vectors_encoded)?;
    phases.mark("write vectors");
    let written = fs::write(dest_sketches, sketches_encoded);
    phases.mark("write sketches");
    written
}

pub fn read_sketches(src: impl AsRef<Path>) -> Result<VectorsAndSketches, String> {
    let content = fs::read(src).map_err(|e| e.to_string())?;
    let (offsets, indices, values, scale) = decode_sparse_sketches(&content, None)?;
    Ok(VectorsAndSketches {
        offsets,
        indices,
        values,
        scale,
    })
}

pub fn save_index(
    graph: Vec<u32>,
    seeds: SeedVecs,
    term_max: BTreeMap<u32, f32>,
    dest: impl AsRef<Path>,
) -> io::Result<()> {
    let mut phases = Phases::new("save_index");
    let encoded_seeds = encode_seed_vecs(&seeds);
    let lengths = &[graph.len() * 4, term_max.len() * 8, seeds.len()];
    let mut encoded: Vec<u8> = vec![];
    for l in lengths {
        encoded.extend_from_slice(
            &u64::try_from(*l)
                .expect("Should convert to u32")
                .to_le_bytes(),
        );
    }
    for g in graph {
        encoded.extend_from_slice(&g.to_le_bytes());
    }
    for (k, v) in term_max {
        encoded.extend_from_slice(&k.to_le_bytes());
        encoded.extend_from_slice(&v.to_le_bytes());
    }
    encoded.extend_from_slice(&encoded_seeds);
    phases.mark("encode graph, term max and seeds");

    let written = fs::write(dest, encoded);
    phases.mark("write index");
    written
}

pub fn read_index(src: impl AsRef<Path>) -> io::Result<IndexOutput> {
    let content = fs::read(src)?;

    let lengths = &[
        u64::from_le_bytes(content[0..8].try_into().expect("Should convert to [u8; 8]")),
        u64::from_le_bytes(
            content[8..16]
                .try_into()
                .expect("Should convert to [u8; 8]"),
        ),
        u64::from_le_bytes(
            content[16..24]
                .try_into()
                .expect("Should convert to [u8; 4]"),
        ),
    ];

    let graph_bytes = &content[24..24 + lengths[0] as usize];
    let mut graph: Vec<u32> = vec![];

    for b in graph_bytes.chunks_exact(4) {
        graph.push(u32::from_le_bytes(
            b.try_into().expect("Should convert to [u8; 4]"),
        ));
    }

    let term_max_bytes =
        &content[24 + lengths[0] as usize..24 + lengths[0] as usize + lengths[1] as usize];
    let mut term_max: BTreeMap<u32, f32> = BTreeMap::new();

    for be in term_max_bytes.chunks_exact(8) {
        let k = &be[0..4];
        let v = &be[4..8];
        let ku = u32::from_le_bytes(k.try_into().expect("Should convert to [u8; 4]"));
        let vf = f32::from_le_bytes(v.try_into().expect("Should convert to [u8; 4]"));
        term_max.insert(ku, vf);
    }

    let seeds_bytes = &content[24 + lengths[0] as usize + lengths[1] as usize..];
    let mut seeds: SeedVecs = BTreeMap::new();

    for i in 0..lengths[2] {
        let (k, v) = read_seed_vec_from_bytes(seeds_bytes, i as usize);
        seeds.insert(k, v);
    }

    Ok(IndexOutput::new(graph, seeds, term_max))
}

// function build_index(vectors, params):
//     W = make_projection(vocab_size, params.m, seed)
//     sketches = parallel map make_sketch over vectors
//     reservoirs = one empty reservoir for each point
//
//     leaves = partition(all point ids, depth = 0)
//
//     parallel for leaf in leaves:
//         for (src, dst, score) in build_leaf_edges(leaf, params.k):
//             lock(reservoirs[src])
//             insert(reservoirs[src], src, dst, sketches, params.cap)
//             unlock(reservoirs[src])
//
//     graph = new fixed-degree graph with MAX_DEGREE = 64
//     parallel for p in all points:
//         candidates = reservoirs[p].slots
//         graph[p] = robust_prune(p, candidates, params.alpha, MAX_DEGREE)
//
//     seeds = build_seed_lists(vectors, SEEDS_PER_TERM)
//
//     if params.refine:
//         parallel for p in all points:
//             found = beam_search_graph(graph, vectors[p], entry_points(vectors[p], seeds))
//             graph[p] = robust_prune(p, found + graph[p], params.alpha, MAX_DEGREE)
//
//     return graph, seeds
