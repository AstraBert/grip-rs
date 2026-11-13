use arrayvec::ArrayVec;

use crate::{
    hashprune::{MAX_RESERVOIR_SLOTS, Slot},
    types::SparseVector,
};

pub const MAX_GRAPH_DEGREE: usize = 64;

pub fn robust_prune(
    candidates: &ArrayVec<Slot, MAX_RESERVOIR_SLOTS>,
    alpha: f32,
    vectors: &[&SparseVector],
) -> ArrayVec<u32, MAX_GRAPH_DEGREE> {
    let mut cand = candidates.clone();
    // sort in reverse
    cand.sort_by(|a, b| a.score.total_cmp(&b.score));
    let mut kept = ArrayVec::new();
    while !cand.is_empty() && kept.len() < MAX_GRAPH_DEGREE {
        // since it is sorted in reverse,
        // popping here means we get
        // the best of the remaining candidates
        let q = cand.pop().unwrap();
        kept.push(q.id);
        if cand.is_empty() {
            break;
        }
        let mut i = (cand.len() - 1) as isize;
        let mut to_remove = vec![];
        // we iter backwards so that
        // we go from best -> worst
        // candidate
        while i >= 0 {
            let dot = vectors[q.id as usize].dot(vectors[cand[i as usize].id as usize]);
            if dot > alpha * cand[i as usize].score.to_f32() {
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
