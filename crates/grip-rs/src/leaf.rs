use crate::types::SparseVector;

pub fn build_leaf_edges(
    leaf: &[u32],
    k: usize,
    scores: &mut [f32],
    vectors: &[&SparseVector],
    triples: &mut Vec<(u32, u32, f32)>,
) -> Vec<(u32, u32, f32)> {
    let n = leaf.len();

    scores.fill(0.0);

    triples.clear();
    for local_id in 0..leaf.len() {
        for (term, value) in vectors[leaf[local_id] as usize]
            .indices
            .iter()
            .zip(vectors[leaf[local_id] as usize].values.iter())
        {
            triples.push((*term, local_id as u32, *value));
        }
    }
    triples.sort_unstable_by_key(|x| (x.0, x.1));

    // write only upper triangle of scores
    for run in triples.chunk_by(|a, b| a.0 == b.0) {
        for (a, &(_, ia, va)) in run.iter().enumerate() {
            let row = &mut scores[ia as usize * n..(ia as usize + 1) * n];
            for &(_, ib, vb) in &run[a + 1..] {
                row[ib as usize] += va * vb;
            }
        }
    }

    // in one go, copy upper -> lower triangle
    for i in 0..n {
        for j in i + 1..n {
            scores[j * n + i] = scores[i * n + j];
        }
    }

    let mut edges = vec![];
    let mut cand: Vec<(usize, f32)> = vec![];
    for i in 0..n {
        cand.clear();
        let row = &scores[i * n..(i + 1) * n];
        cand = row
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
