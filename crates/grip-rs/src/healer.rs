use memmap2::Mmap;
use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::ParallelSliceMut,
};

use crate::{
    SparseVector, Visited,
    disk::decode_sparse_vectors_raw,
    robustprune::{MAX_GRAPH_DEGREE, robust_prune},
    tombstone::Tombstones,
};

struct Scratch {
    visited: Visited,
    dense_scatter: Vec<f32>,
}

impl Scratch {
    fn new(n_docs: usize, vocab_size: usize) -> Self {
        Self {
            visited: Visited {
                counter: 0,
                visited: vec![0; n_docs],
            },
            dense_scatter: vec![0.0; vocab_size],
        }
    }
}

pub fn heal(
    graph: &[u32],
    tombstones: &Tombstones,
    alpha: f32,
    max_dfs: usize,
    max_border_length: usize,
    vocab_size: usize,
    vectors_mmap: &Mmap,
) -> Result<Vec<u32>, String> {
    let mut new_graph = graph.to_vec();
    new_graph
        .par_chunks_mut(MAX_GRAPH_DEGREE)
        .enumerate()
        .try_for_each_init(
            || Scratch::new(graph.len() / MAX_GRAPH_DEGREE, vocab_size),
            |s, (p, row)| {
                if tombstones.is_deleted(p as u32) {
                    for r in row.iter_mut() {
                        *r = u32::MAX;
                    }
                    return Ok::<(), String>(());
                }
                if row.iter().all(|id| !tombstones.is_deleted(*id)) {
                    return Ok(());
                }
                let mut live_nb = row
                    .iter()
                    .filter(|x| **x != u32::MAX && !tombstones.is_deleted(**x))
                    .map(|x| *x)
                    .collect::<Vec<u32>>();
                let mut border = vec![];
                s.visited.counter += 1;
                s.visited.visited[p] = s.visited.counter;
                for r in row.iter() {
                    if *r != u32::MAX {
                        s.visited.visited[*r as usize] = s.visited.counter;
                    }
                }

                let mut stack = row
                    .iter()
                    .filter(|x| **x != u32::MAX && tombstones.is_deleted(**x))
                    .map(|x| *x)
                    .collect::<Vec<u32>>();
                let mut steps = 0;

                while !stack.is_empty() && steps < max_dfs && border.len() < max_border_length {
                    let d = stack.pop().unwrap();
                    steps += 1;
                    let d_row =
                        &graph[d as usize * MAX_GRAPH_DEGREE..(d as usize + 1) * MAX_GRAPH_DEGREE];
                    for x in d_row {
                        if *x == u32::MAX || s.visited.visited[*x as usize] == s.visited.counter {
                            continue;
                        }
                        s.visited.visited[*x as usize] = s.visited.counter;
                        if tombstones.is_deleted(*x) {
                            stack.push(*x);
                        } else {
                            border.push(*x);
                        }
                    }
                }

                live_nb.append(&mut border);

                // early return if no neighbor is alive
                // and search finds nothing
                if live_nb.is_empty() {
                    for r in row.iter_mut() {
                        *r = u32::MAX;
                    }
                    return Ok(());
                }

                let mut ids = live_nb.iter().map(|v| *v as usize).collect::<Vec<usize>>();
                ids.insert(0, p);
                let vectors = decode_sparse_vectors_raw(&vectors_mmap, Some(ids))?;
                for (t, w) in vectors[0].iter() {
                    s.dense_scatter[*t as usize] = *w;
                }

                let mut scored = vec![];
                for k in 1..vectors.len() {
                    let mut dot = 0.0;
                    for (tk, wk) in vectors[k].iter() {
                        dot += s.dense_scatter[*tk as usize] * *wk;
                    }
                    scored.push((k as u32, dot));
                }

                for (t, _) in vectors[0].iter() {
                    s.dense_scatter[*t as usize] = 0.0;
                }

                let actual_vectors: Vec<SparseVector> = vectors
                    .iter()
                    .map(|v| SparseVector::from_id_value_pairs_unchecked(v))
                    .collect();
                let borrowed_vectors: Vec<&SparseVector> = actual_vectors.iter().collect();

                let kept =
                    robust_prune::<f32>(scored, alpha, &borrowed_vectors, &mut s.dense_scatter);

                for j in 0..MAX_GRAPH_DEGREE {
                    let local = kept[j];
                    if local == u32::MAX {
                        row[j] = u32::MAX;
                    } else {
                        row[j] = live_nb[local as usize - 1];
                    }
                }

                Ok(())
            },
        )?;

    Ok(new_graph)
}
