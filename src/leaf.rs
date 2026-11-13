use std::collections::HashMap;

use crate::types::SparseVector;

pub fn build_leaf_edges(
    leaf: &[u32],
    k: usize,
    scores: &mut [f32],
    vectors: &[&SparseVector],
) -> Vec<(u32, u32, f32)> {
    let n = leaf.len();

    scores.fill(0.0);

    let mut postings: HashMap<u32, Vec<(u32, f32)>> = HashMap::new();
    for local_id in 0..leaf.len() {
        for (term, value) in vectors[leaf[local_id] as usize]
            .indices
            .iter()
            .zip(vectors[leaf[local_id] as usize].values.iter())
        {
            postings
                .entry(*term)
                .and_modify(|e| e.push((local_id as u32, *value)))
                .or_insert(vec![(local_id as u32, *value)]);
        }
    }

    for ls in postings.values() {
        for a in 0..ls.len() {
            for b in a + 1..ls.len() {
                let (ia, va) = ls[a];
                let (ib, vb) = ls[b];
                scores[(ia as usize * n) + ib as usize] += va * vb;
                scores[(ib as usize * n) + ia as usize] += va * vb;
            }
        }
    }

    let mut edges = vec![];
    for i in 0..n {
        let row = &scores[i * n..(i + 1) * n];
        let mut cand: Vec<(usize, f32)> = row
            .iter()
            .copied()
            .enumerate() // pairs of (j, score)
            .filter(|&(j, s)| leaf[j] != leaf[i] && s > 0.0) // skip itself and pairs with no shared term
            .collect();

        let k = k.min(cand.len());
        if k > 0 {
            // after this call, the first k entries are the k largest scores, in no particular order
            cand.select_nth_unstable_by(k - 1, |a, b| b.1.total_cmp(&a.1));
        }
        let top = &cand[..k]; // each entry is (j, score)
        for c in top {
            edges.push((leaf[i], leaf[c.0], c.1));
            edges.push((leaf[c.0], leaf[i], c.1));
        }
    }

    edges
}
