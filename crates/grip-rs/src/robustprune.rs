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
    candidates: &Vec<(u32, T)>,
    alpha: f32,
    vectors: &[&SparseVector],
) -> ArrayVec<u32, MAX_GRAPH_DEGREE> {
    let mut cand = candidates.clone();
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
        let mut i = (cand.len() - 1) as isize;
        let mut to_remove = vec![];
        // we iter backwards so that
        // we go from best -> worst
        // candidate
        while i >= 0 {
            let dot = vectors[q.0 as usize].dot(vectors[cand[i as usize].0 as usize]);
            if dot > alpha * cand[i as usize].1.as_f32() {
                to_remove.push(i as usize);
            }
            i -= 1;
        }
        to_remove
            .iter()
            .map(|u| {
                let _ = cand.remove(*u);
            })
            .for_each(drop);
    }

    while kept.len() < MAX_GRAPH_DEGREE {
        kept.push(u32::MAX);
    }

    kept
}
