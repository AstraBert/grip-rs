use std::{
    cmp::Reverse,
    collections::{BTreeMap, BinaryHeap},
};

use memmap2::Mmap;
use serde::{Deserialize, Serialize};

use crate::{
    disk::decode_sparse_vectors_raw,
    seeding::SeedVecs,
    types::{QuantizedSparseVector, SearchParams, SparseVector, SparseVectorSortKey},
};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct VectorsAndSketches {
    pub offsets: Vec<usize>,
    pub indices: Vec<u32>,
    pub values: Vec<u8>,
    pub scale: f32,
}

impl VectorsAndSketches {
    pub fn get<'a>(&'a self, index: usize) -> Option<(&'a [u32], &'a [u8])> {
        let offset_start = self.offsets.get(index);
        let offset_end = self.offsets.get(index + 1);
        if offset_end.is_none() || offset_start.is_none() {
            return None;
        }
        let (start, end) = (offset_start.unwrap(), offset_end.unwrap());
        let indices = &self.indices[*start..*end];
        let values = &self.values[*start..*end];

        Some((indices, values))
    }
}

#[derive(Debug, Clone)]
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

#[derive(Debug, Clone)]
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

pub fn compute_entry_points(
    by_impact: &mut QuantizedSparseVector,
    seeds: &SeedVecs,
    term_max: &BTreeMap<u32, f32>,
    max_query_dims: usize,
) -> Vec<u32> {
    by_impact.sort_with_fn(|el_a: (u32, u8), el_b: (u32, u8)| {
        let max_weigth_a = term_max.get(&el_a.0).unwrap_or(&0_f32);
        let max_weigth_b = term_max.get(&el_b.0).unwrap_or(&0_f32);
        (el_b.1 as f32 * *max_weigth_b).total_cmp(&(*max_weigth_a * el_a.1 as f32))
    });
    by_impact.truncate(max_query_dims);
    let mut entry_points = vec![];
    for t in by_impact.indices.iter() {
        let seeds = seeds.get(t);
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
    entry_points
}

pub fn run_beam_search(
    q8: &QuantizedSparseVector,
    visited: &mut Visited,
    entry_points: &mut Vec<u32>,
    vectors_and_sketches: &VectorsAndSketches,
    dense: &mut Vec<u8>,
    graph: &[u32],
    ef: usize,
    patience: usize,
) -> Result<Vec<(Reverse<u32>, u32)>, String> {
    let mut candidates: BinaryHeap<(u32, u32)> = BinaryHeap::new();
    let mut results: BinaryHeap<(Reverse<u32>, u32)> = BinaryHeap::new();
    visited.counter += 1; // move this line above the seeding
    entry_points.sort_unstable();
    entry_points.dedup();
    for &e in entry_points.iter() {
        visited.visited[e as usize] = visited.counter;
        let mut score = 0u32;
        let sketch = vectors_and_sketches.get(e as usize);
        if sketch.is_none() {
            return Err("Tried to read an out-of-bounds vector".to_string());
        }
        let sketch = sketch.unwrap();
        for (i, w) in sketch.0.iter().zip(sketch.1.iter()) {
            score += *w as u32 * dense[*i as usize] as u32;
        }
        candidates.push((score, e));
        maybe_add_result(&mut results, ef, score, e);
    }
    let mut no_change = 0;

    while !candidates.is_empty() && no_change < patience {
        let (_, node) = candidates.pop().unwrap();
        let mut changed = false;
        for nb in &graph[(node as usize) * 64..(node as usize + 1) * 64] {
            if *nb == u32::MAX {
                break;
            }
            if visited.visited[*nb as usize] != visited.counter {
                visited.visited[*nb as usize] = visited.counter;
                let mut score = 0u32;
                let sketch = vectors_and_sketches.get(*nb as usize);
                if sketch.is_none() {
                    return Err("Tried to read an out-of-bounds vector".to_string());
                }
                let sketch = sketch.unwrap();
                for (i, w) in sketch.0.iter().zip(sketch.1.iter()) {
                    score += *w as u32 * dense[*i as usize] as u32;
                }
                let added = maybe_add_result(&mut results, ef, score, *nb);
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

    let res = results.into_sorted_vec();

    // reset dense for reuse
    for t in q8.indices.iter() {
        dense[*t as usize] = 0;
    }

    Ok(res)
}

pub fn rescore(
    ids: &mut Vec<u32>,
    query: &SparseVector,
    sparse_vectors_mmap: &Mmap,
    dense_rescoring: &mut Vec<f32>,
    params: &SearchParams,
) -> Result<Vec<(u32, f32)>, String> {
    let mut exact_scores: Vec<(u32, f32)> = Vec::new();
    if ids.len() > params.top_k * params.rescore_factor {
        ids.truncate(params.top_k * params.rescore_factor);
    }
    let vectors = decode_sparse_vectors_raw(
        sparse_vectors_mmap,
        Some(ids.iter().map(|&i| i as usize).collect()),
    )?;

    // reset dense for reuse
    for (t, w) in query.indices.iter().zip(query.values.iter()) {
        dense_rescoring[*t as usize] = *w;
    }

    for (id, v) in ids.iter().zip(vectors.iter()) {
        let mut score = 0_f32;
        for (i, w) in v.iter() {
            score += dense_rescoring[*i as usize] * *w;
        }
        exact_scores.push((*id, score));
    }

    // clean before returning
    for t in query.indices.iter() {
        dense_rescoring[*t as usize] = 0_f32;
    }

    exact_scores.sort_by(|(_, va), (_, vb)| (*vb).total_cmp(va));
    exact_scores.truncate(params.top_k);

    Ok(exact_scores.iter().map(|(i, v)| (*i, *v)).collect())
}

pub fn search(
    query: &SparseVector,
    vectors_and_sketches: &VectorsAndSketches,
    index_output: &IndexOutput,
    params: SearchParams,
    visited: &mut Visited,
    dense: &mut Vec<u8>,
    dense_rescoring: &mut Vec<f32>,
    sparse_vectors_mmap: &Mmap,
) -> Result<Vec<(u32, f32)>, String> {
    let q8 = query.quantize(vectors_and_sketches.scale)?;
    for (t, w) in q8.indices.iter().zip(q8.values.iter()) {
        dense[*t as usize] = *w;
    }
    let mut by_impact = q8.clone();
    let mut entry_points = compute_entry_points(
        &mut by_impact,
        &index_output.seeds,
        &index_output.term_max,
        params.max_query_dims,
    );
    let res = run_beam_search(
        &q8,
        visited,
        &mut entry_points,
        vectors_and_sketches,
        dense,
        &index_output.graph,
        params.ef,
        params.patience,
    )?;
    let mut ids = res.iter().map(|v| v.1).collect();
    rescore(
        &mut ids,
        query,
        sparse_vectors_mmap,
        dense_rescoring,
        &params,
    )
}
