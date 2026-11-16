use std::{collections::BTreeMap, fs, io, path::Path, sync::Mutex};

use crate::{
    disk::{decode_sparse_sketches, encode_vectors},
    hashprune::{Reservoir, insert_in_reservoir, make_projection, make_sketch},
    leaf::build_leaf_edges,
    partition::partition,
    robustprune::{MAX_GRAPH_DEGREE, robust_prune},
    seeding::{SeedVecs, build_seed_list, encode_seed_vecs, read_seed_vec_from_bytes},
};
use arrayvec::ArrayVec;
use half::bf16;
use rayon::prelude::*;

pub mod disk;
pub mod hashprune;
pub mod leaf;
pub mod partition;
pub mod robustprune;
mod search;
pub mod seeding;
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

pub fn build_index(vectors: Vec<SparseVector>, params: BuildParams) -> IndexBuildOutput {
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

    let leaves = partition(
        &borrowed_vectors,
        &(0..vectors.len()).map(|i| i as u32).collect::<Vec<u32>>(),
        0,
        params.seed,
    );

    leaves
        .par_iter()
        .map(|leaf| {
            let mut scores = vec![0.0_f32; leaf.len() * leaf.len()];
            let mut triples = Vec::with_capacity(leaf.len());
            let edges = build_leaf_edges(
                leaf,
                params.leaf_k,
                &mut scores,
                &borrowed_vectors,
                &mut triples,
            );
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
        .collect::<Result<Vec<_>, String>>()?;

    let graph: Vec<u32> = (0..vectors.len())
        .into_par_iter()
        .map(|p| {
            let candidates = reservoirs[p].lock().map_err(|e| e.to_string())?;
            let kept = robust_prune(
                &candidates.slots,
                params.robust_prune_alpha,
                &borrowed_vectors,
            );

            Ok(kept)
        })
        .collect::<Result<Vec<ArrayVec<u32, MAX_GRAPH_DEGREE>>, String>>()?
        .iter()
        .flatten()
        .map(|v| *v)
        .collect();

    let (seeds, term_max) = build_seed_list(&borrowed_vectors, params.entrypoint_seeds);

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

    Ok((graph, seeds, term_max, routing_sketches, scale))
}

pub fn save_vectors_and_sketches(
    vectors: Vec<SparseVector>,
    sketches: Vec<QuantizedSparseVector>,
    scale: f32,
    dest_vectors: impl AsRef<Path>,
    dest_sketches: impl AsRef<Path>,
) -> io::Result<()> {
    let vectors_encoded = encode_vectors(&vectors.iter().collect::<Vec<&SparseVector>>(), 4, None);
    let sketches_encoded = encode_vectors(
        &sketches.iter().collect::<Vec<&QuantizedSparseVector>>(),
        1,
        Some(scale),
    );
    fs::write(dest_vectors, vectors_encoded)?;
    fs::write(dest_sketches, sketches_encoded)
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

    fs::write(dest, encoded)
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
