use std::cmp::Ordering;

use arrayvec::ArrayVec;

use crate::types::SparseVector;

pub const MAX_GRAPH_DEGREE: usize = 64;

pub trait AsF32 {
    fn as_f32(&self) -> f32;
    fn safe_compare(&self, other: &Self) -> Ordering;
}

impl AsF32 for f32 {
    fn as_f32(&self) -> f32 {
        *self
    }

    fn safe_compare(&self, other: &Self) -> Ordering {
        self.total_cmp(other)
    }
}

impl AsF32 for half::bf16 {
    fn as_f32(&self) -> f32 {
        self.to_f32()
    }

    fn safe_compare(&self, other: &Self) -> Ordering {
        self.total_cmp(other)
    }
}

pub fn robust_prune<T: AsF32 + Clone>(
    candidates: Vec<(u32, T)>,
    alpha: f32,
    vectors: &[&SparseVector],
    dense: &mut [f32],
) -> ArrayVec<u32, MAX_GRAPH_DEGREE> {
    let mut cand = candidates;
    // sort in reverse
    cand.sort_by(|a, b| a.1.safe_compare(&b.1));
    let mut kept = ArrayVec::new();
    while !cand.is_empty() && kept.len() < MAX_GRAPH_DEGREE {
        // since it is sorted in reverse,
        // popping here means we get
        // the best of the remaining candidates
        let q = cand.pop().unwrap();
        kept.push(q.0);
        if cand.is_empty() {
            break;
        }
        for (t, w) in vectors[q.0 as usize]
            .indices
            .iter()
            .zip(vectors[q.0 as usize].values.iter())
        {
            dense[*t as usize] = *w;
        }
        cand.retain(|c| {
            let vector = vectors[c.0 as usize];
            let mut dot = 0_f32;
            for (t, w) in vector.indices.iter().zip(vector.values.iter()) {
                dot += dense[*t as usize] * *w;
            }
            dot <= alpha * c.1.as_f32()
        });

        for t in vectors[q.0 as usize].indices.iter() {
            dense[*t as usize] = 0.0;
        }
    }

    while kept.len() < MAX_GRAPH_DEGREE {
        kept.push(u32::MAX);
    }

    kept
}
