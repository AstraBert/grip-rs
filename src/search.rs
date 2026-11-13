use std::{
    cmp::Reverse,
    collections::{BTreeMap, BinaryHeap, HashMap},
};

use serde::{Deserialize, Serialize};

use crate::{
    seeding::SeedVecs,
    types::{QuantizedSparseVector, SearchParams, SparseVector, SparseVectorSortKey},
};

#[derive(Debug, Serialize, Deserialize)]
pub struct VectorsAndSketches {
    pub vectors: Vec<SparseVector>,
    pub sketches: Vec<QuantizedSparseVector>,
    pub scale: f32,
}

#[derive(Debug)]
pub struct IndexOutput {
    pub graph: Vec<u32>,
    pub seeds: SeedVecs,
    pub term_max: BTreeMap<u32, f32>,
}

impl IndexOutput {
    pub fn new(graph: Vec<u32>, seeds: SeedVecs, term_max: BTreeMap<u32, f32>) -> Self {
        Self {
            graph,
            seeds,
            term_max,
        }
    }
}

pub struct Visited {
    pub counter: u32,
    pub visited: Vec<u32>,
}

pub fn make_routing_sketch(
    vector: &SparseVector,
    sketch_size: usize,
    scale: f32,
) -> Result<QuantizedSparseVector, String> {
    let mut kept = vector.clone();
    kept.sort_by(SparseVectorSortKey::ValuesDesc);
    kept.truncate(sketch_size);
    let quant = kept.quantize(scale)?;

    Ok(quant)
}

fn maybe_add_result(
    results: &mut BinaryHeap<(Reverse<u32>, u32)>,
    size: usize,
    score: u32,
    node: u32,
) -> bool {
    if results.len() < size {
        results.push((Reverse(score), node));
        true
    } else if results.peek().unwrap().0.0 < score {
        results.pop();
        results.push((Reverse(score), node));
        true
    } else {
        false
    }
}

pub fn search(
    query: &SparseVector,
    vectors_and_sketches: &VectorsAndSketches,
    index_output: &IndexOutput,
    params: SearchParams,
    visited: &mut Visited,
) -> Result<Vec<(u32, f32)>, String> {
    let q8 = query.quantize(vectors_and_sketches.scale)?;
    let mut by_impact = q8.clone();
    by_impact.sort_with_fn(|el_a: (u32, u8), el_b: (u32, u8)| {
        let max_weigth_a = index_output.term_max.get(&el_a.0).unwrap_or(&0_f32);
        let max_weigth_b = index_output.term_max.get(&el_b.0).unwrap_or(&0_f32);
        (el_b.1 as f32 * *max_weigth_b).total_cmp(&(*max_weigth_a * el_a.1 as f32))
    });
    by_impact.truncate(params.max_query_dims);
    let mut entry_points = vec![];
    for t in by_impact.indices.iter() {
        let seeds = index_output.seeds.get(t);
        if let Some(sds) = seeds {
            for s in sds {
                entry_points.push(s.1);
            }
        }
    }
    if entry_points.is_empty() {
        // use the first node
        // as fallback for empty
        // seeds
        entry_points = vec![0];
    }
    let mut candidates: BinaryHeap<(u32, u32)> = BinaryHeap::new();
    let mut results: BinaryHeap<(Reverse<u32>, u32)> = BinaryHeap::new();
    visited.counter += 1; // move this line above the seeding
    entry_points.sort_unstable();
    entry_points.dedup();
    for &e in &entry_points {
        visited.visited[e as usize] = visited.counter;
        let score = q8.dot_unscaled(&vectors_and_sketches.sketches[e as usize]);
        candidates.push((score, e));
        maybe_add_result(&mut results, params.ef, score, e);
    }
    let mut no_change = 0;

    while !candidates.is_empty() && no_change < params.patience {
        let (_, node) = candidates.pop().unwrap();
        let mut changed = false;
        for nb in &index_output.graph[(node as usize) * 64..(node as usize + 1) * 64] {
            if *nb == u32::MAX {
                break;
            }
            if visited.visited[*nb as usize] != visited.counter {
                visited.visited[*nb as usize] = visited.counter;
                let score = q8.dot_unscaled(&vectors_and_sketches.sketches[*nb as usize]);
                let added = maybe_add_result(&mut results, params.ef, score, *nb);
                if added {
                    candidates.push((score, *nb));
                    changed = true;
                }
            }
        }
        if !changed {
            no_change += 1;
        }
    }

    let mut exact_scores: HashMap<u32, f32> = HashMap::new();
    for node in results {
        let exact_score = query.dot(&vectors_and_sketches.vectors[node.1 as usize]);
        exact_scores.insert(node.1, exact_score);
    }

    let mut entries = exact_scores.iter().collect::<Vec<(&u32, &f32)>>();
    entries.sort_by(|(_, va), (_, vb)| (**vb).total_cmp(*va));
    entries.truncate(params.top_k);

    Ok(entries.iter().map(|(i, v)| (**i, **v)).collect())
}
