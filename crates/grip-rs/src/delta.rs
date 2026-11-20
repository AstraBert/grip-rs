use std::collections::HashMap;

use crate::{SparseVector, tombstone::Tombstones};

pub struct Delta {
    pub docs: Vec<SparseVector>,
    pub ext_ids: Vec<u32>,
    pub postings: HashMap<u32, Vec<(u32, f32)>>,
}

pub fn delta_insert(delta: &mut Delta, vector: SparseVector, ext_id: u32) {
    let local_base = delta.docs.len();
    for (t, w) in vector.indices.iter().zip(vector.values.iter()) {
        delta
            .postings
            .entry(*t)
            .and_modify(|v| {
                v.push((
                    u32::try_from(local_base).expect("Should convert to u32"),
                    *w,
                ))
            })
            .or_insert(vec![(
                u32::try_from(local_base).expect("Should convert to u32"),
                *w,
            )]);
    }
    delta.docs.push(vector);
    delta.ext_ids.push(ext_id);
}

pub fn delta_search(
    delta: &Delta,
    query: &SparseVector,
    k: usize,
    tombstones: &Tombstones,
    base: u32,
    acc: &mut Vec<f32>,
) -> Vec<(u32, f32)> {
    let mut touched: Vec<u32> = vec![];
    for (t, qw) in query.indices.iter().zip(query.values.iter()) {
        if let Some(postings) = delta.postings.get(t) {
            for (local, w) in postings.iter() {
                if acc[*local as usize] == 0.0 {
                    touched.push(*local);
                }
                acc[*local as usize] += *qw * *w;
            }
        }
    }
    touched.retain(|v| !tombstones.is_deleted(*v));
    touched.sort_by(|a, b| acc[*b as usize].total_cmp(&acc[*a as usize]));
    let results = touched
        .iter()
        .take(k)
        .map(|v| (*v + base, acc[*v as usize]))
        .collect::<Vec<(u32, f32)>>();
    for t in &touched {
        acc[*t as usize] = 0.0;
    }
    results
}
