use std::{
    collections::HashMap,
    hash::{DefaultHasher, Hash, Hasher},
};

use crate::types::SparseVector;
use rand::prelude::*;
use rayon::prelude::*;

pub const MIN_LEAF_LENGTH: usize = 100;
pub const MAX_LEAF_LENGTH: usize = 1024;
pub const MAX_DEPTH: usize = 5;
pub const LEADER_FRACTION: f64 = 0.001;

fn random_sample<T: Clone, R: Rng + ?Sized>(pool: &[T], size: usize, mut rng: &mut R) -> Vec<T> {
    let sampled = pool
        .choose_iter(&mut rng)
        .unwrap()
        .take(size)
        .map(|t| t.clone())
        .collect::<Vec<T>>();
    sampled
}

fn build_inverted_index(
    vectors: &[&SparseVector],
    points: &[u32],
) -> HashMap<u32, Vec<(u32, f32)>> {
    let mut postings: HashMap<u32, Vec<(u32, f32)>> = HashMap::new();

    for (i, v) in points.iter().zip(vectors) {
        for (term, value) in v.indices.iter().zip(v.values.iter()) {
            postings
                .entry(*term)
                .and_modify(|e| e.push((*i, *value)))
                .or_insert(vec![(*i, *value)]);
        }
    }

    postings
}

pub fn compute_hash<T: Hash>(value: &T) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

/// `fanouts` holds the fanout of depth 0 and of depth 1. Deeper levels always use 1.
fn fanout_for_dept(depth: usize, fanouts: [usize; 2]) -> usize {
    fanouts.get(depth).copied().unwrap_or(1)
}

fn pop_random<T, R: Rng + ?Sized>(pool: &mut Vec<T>, rng: &mut R) -> T {
    let idx = rng.random_range::<usize, _>(..pool.len());
    pool.remove(idx)
}

fn merge_small_groups_randomly<T, R: Rng + ?Sized>(
    groups: &mut Vec<Vec<T>>,
    min_len: usize,
    mut rng: &mut R,
) {
    let mut small_groups = groups
        .extract_if(.., |v| v.len() < min_len)
        .collect::<Vec<Vec<T>>>();

    if small_groups.is_empty() {
        return;
    }

    loop {
        let mut sg = small_groups.pop().unwrap();
        while !small_groups.is_empty() && sg.len() < min_len {
            let other = pop_random(&mut small_groups, &mut rng);
            sg.extend(other);
        }
        groups.push(sg);
        if small_groups.is_empty() {
            break;
        }
    }
}

pub fn partition(
    vectors: &[&SparseVector],
    ids: &[u32],
    depth: usize,
    seed: u64,
    fanouts: [usize; 2],
) -> Vec<Vec<u32>> {
    if depth > MAX_DEPTH {
        return vec![ids.to_vec()];
    }

    let mut rng = StdRng::seed_from_u64(seed + depth as u64);
    let points: Vec<u32> = vectors.iter().enumerate().map(|(i, _)| i as u32).collect();

    if points.len() <= MAX_LEAF_LENGTH {
        return vec![ids.to_vec()];
    }

    let n = points.len();
    let fanout = fanout_for_dept(depth, fanouts);
    let leader_count = ((LEADER_FRACTION * n as f64) as usize)
        .clamp(4 * fanout, 1000)
        .min(n);
    let leaders = random_sample(&points, leader_count, &mut rng);
    let leader_vectors: Vec<&SparseVector> = leaders.iter().map(|&l| vectors[l as usize]).collect();
    let leader_positions: Vec<u32> = (0..leaders.len() as u32).collect();
    let index = build_inverted_index(&leader_vectors, &leader_positions);

    let mut groups: Vec<Vec<u32>> = vec![vec![]; leader_count];

    let point_groups = points
        .into_par_iter()
        .map(|i| {
            let mut scores: HashMap<u32, f32> = HashMap::new();
            for (term, value) in vectors[i as usize]
                .indices
                .iter()
                .zip(vectors[i as usize].values.iter())
            {
                if let Some(ls) = index.get(term) {
                    for (leader, leader_value) in ls {
                        scores
                            .entry(*leader)
                            .and_modify(|e| *e += value * leader_value)
                            .or_insert(*value * leader_value);
                    }
                }
            }
            let best = if scores.is_empty() {
                vec![(compute_hash(&i) as usize % leader_count) as u32]
            } else {
                let mut entries: Vec<(&u32, &f32)> = scores.iter().collect();
                let k = fanout.min(entries.len());
                entries.select_nth_unstable_by(k - 1, |a, b| b.1.total_cmp(a.1));
                entries.truncate(k);
                entries.iter().map(|(i, _)| **i).collect::<Vec<u32>>()
            };

            (best, i)
        })
        .collect::<Vec<(Vec<u32>, u32)>>();

    for (best, i) in point_groups {
        for leader in best {
            groups[leader as usize].push(i);
        }
    }

    merge_small_groups_randomly(&mut groups, MIN_LEAF_LENGTH, &mut rng);

    let results: Vec<Vec<u32>> = groups
        .par_iter()
        .enumerate()
        .map(|(gid, g)| {
            let sub_vectors: Vec<&SparseVector> = g.iter().map(|&p| vectors[p as usize]).collect();
            let sub_ids: Vec<u32> = g.iter().map(|&p| ids[p as usize]).collect();
            partition(
                &sub_vectors,
                &sub_ids,
                depth + 1,
                seed + gid as u64,
                fanouts,
            )
        })
        .collect::<Vec<Vec<Vec<u32>>>>()
        .into_iter()
        .flatten()
        .collect();

    results
}
