use std::cmp::Ordering;

use arrayvec::{ArrayVec, CapacityError};
use half::bf16;
use rand::prelude::*;
use rand_distr::StandardNormal;

use crate::types::SparseVector;

pub const MAX_RESERVOIR_SLOTS: usize = 64;

struct NormalRng {
    inner: StdRng,
}

impl NormalRng {
    fn new(seed: u64) -> Self {
        Self {
            inner: StdRng::seed_from_u64(seed),
        }
    }

    fn normal(&mut self) -> f32 {
        let val: f32 = self.inner.sample(StandardNormal);
        val
    }
}

/// Creates `vocab_size` Gaussian hyperplanes of `m` size sampling their components
/// from a standard normal distribution
pub fn make_projection(vocab_size: usize, m: usize, seed: u64) -> Vec<f32> {
    let mut rng = NormalRng::new(seed);
    let mut w = vec![0_f32; m * vocab_size];
    for t in 0..vocab_size {
        for i in 0..m {
            w[t * m + i] = rng.normal();
        }
    }

    w
}

/// The sketch of a document is its `m` dot products with the `m` random vectors generated with [make_projection]
pub fn make_sketch(doc: &SparseVector, w: &[f32], m: usize) -> Vec<f32> {
    let mut sketch = vec![0_f32; m];
    for (idx, val) in doc.indices.iter().zip(doc.values.iter()) {
        for i in 0..m {
            sketch[i] += *val * w[*idx as usize * m + i];
        }
    }
    sketch
}

/// Hash of the arrow from p to c. One bit for each of the m random vectors.
pub fn residual_hash(sketch_p: &[f32], sketch_c: &[f32], m: usize) -> u16 {
    let mut h: u16 = 0;
    for i in 0..m {
        if sketch_c[i] - sketch_p[i] >= 0_f32 {
            h = h | (1 << i);
        }
    }
    h
}

#[derive(Debug, Clone, Copy)]
pub struct Slot {
    pub id: u32,
    pub hash: u16,
    pub score: bf16,
}

pub struct Reservoir {
    pub slots: ArrayVec<Slot, MAX_RESERVOIR_SLOTS>,
    pub worst_slot_hash: u16,
}

impl Default for Reservoir {
    fn default() -> Self {
        Self {
            slots: ArrayVec::new(),
            worst_slot_hash: 0,
        }
    }
}

impl Reservoir {
    pub fn get(&self, hash: u16) -> Option<usize> {
        let mut lo = 0;
        let mut hi = self.slots.len();
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            match self.slots[mid].hash.cmp(&hash) {
                Ordering::Equal => return Some(mid),
                Ordering::Greater => hi = mid,
                Ordering::Less => lo = mid + 1,
            }
        }
        None
    }

    pub fn remove(&mut self, hash: u16) {
        let mut lo = 0;
        let mut hi = self.slots.len();
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            match self.slots[mid].hash.cmp(&hash) {
                Ordering::Equal => {
                    self.slots.remove(mid);
                    break;
                }
                Ordering::Greater => hi = mid,
                Ordering::Less => lo = mid + 1,
            }
        }
    }

    pub fn try_put(&mut self, slot: Slot) -> Result<(), CapacityError<Slot>> {
        let pos = self.slots.partition_point(|s| s.hash < slot.hash);
        self.slots.try_insert(pos, slot)
    }

    fn refresh_worst(&mut self) {
        let min_score_slot = self
            .slots
            .iter()
            .min_by(|a, b| a.score.total_cmp(&b.score))
            .expect("Slots should not be empty at this time");
        self.worst_slot_hash = min_score_slot.hash;
    }
}

pub fn insert_in_reservoir(
    reservoir: &mut Reservoir,
    src: u32,
    dst: u32,
    score: bf16,
    sketches: &[&[f32]],
    m: usize,
) -> Result<(), CapacityError<Slot>> {
    let h = residual_hash(sketches[src as usize], sketches[dst as usize], m);
    let pos = reservoir.get(h);

    // another edge has the same cone (hash)
    if let Some(p) = pos {
        // this edge is closer,
        // gets picked as cone
        // representative
        if score > reservoir.slots[p].score {
            reservoir.slots[p] = Slot {
                id: dst,
                hash: h,
                score,
            };
            reservoir.refresh_worst();
        }
        return Ok(());
    }

    // free space, just insert
    if reservoir.slots.len() < MAX_RESERVOIR_SLOTS {
        reservoir.try_put(Slot {
            id: dst,
            hash: h,
            score,
        })?;
        reservoir.refresh_worst();
        return Ok(());
    }

    // full, so compare with the weakest edge
    let worst = reservoir.slots[reservoir.get(reservoir.worst_slot_hash).unwrap()];
    if score > worst.score {
        reservoir.remove(reservoir.worst_slot_hash);
        reservoir.try_put(Slot {
            id: dst,
            hash: h,
            score,
        })?;
        reservoir.refresh_worst();
    }

    Ok(())
}
