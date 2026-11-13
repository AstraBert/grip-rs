use std::{
    cmp::Reverse,
    collections::{BTreeMap, BinaryHeap, HashMap, HashSet},
};

use crate::types::SparseVector;
use ordered_float::OrderedFloat;

pub type SeedHeaps = HashMap<u32, BinaryHeap<(Reverse<OrderedFloat<f32>>, u32)>>;
pub type SeedVecs = BTreeMap<u32, Vec<(f32, u32)>>;

pub fn build_seed_list(vectors: &[&SparseVector], size: usize) -> (SeedVecs, BTreeMap<u32, f32>) {
    let mut seed_heaps: SeedHeaps = HashMap::new();
    let mut term_max: BTreeMap<u32, f32> = BTreeMap::new();

    let mut all_terms: HashSet<u32> = HashSet::new();

    for v in vectors.iter() {
        for t in v.indices.iter() {
            all_terms.insert(*t);
        }
    }

    for t in all_terms {
        seed_heaps.insert(t, BinaryHeap::new());
        term_max.insert(t, 0_f32);
    }

    for d in 0..vectors.len() {
        for (t, w) in vectors[d].indices.iter().zip(vectors[d].values.iter()) {
            term_max
                .entry(*t)
                .and_modify(|e| *e = e.max(*w))
                .or_insert(*w);
            if let Some(heap) = seed_heaps.get_mut(t) {
                if heap.len() < size {
                    heap.push((Reverse(OrderedFloat(*w)), d as u32));
                } else if OrderedFloat(*w) > heap.peek().unwrap().0.0 {
                    heap.pop();
                    heap.push((Reverse(OrderedFloat(*w)), d as u32));
                }
            }
        }
    }

    let seed_vecs: SeedVecs = seed_heaps
        .iter()
        .map(|(k, v)| {
            (
                *k,
                v.iter()
                    .map(|f| (f.0.0.into_inner(), f.1 as u32))
                    .collect::<Vec<(f32, u32)>>(),
            )
        })
        .collect();

    (seed_vecs, term_max)
}

pub fn encode_seed_vecs(entries: &SeedVecs) -> Vec<u8> {
    let header_len = entries.len() * 4;
    let body_len: usize = entries.iter().map(|(_, v)| 4 + 2 + v.len() * 8).sum();
    let mut buf = Vec::with_capacity(header_len + body_len);

    // 1) offset table
    let mut offset = header_len;
    for (_, v) in entries {
        buf.extend_from_slice(
            &u32::try_from(offset)
                .expect("offset > u32::MAX")
                .to_le_bytes(),
        );
        offset += 4 + 2 + v.len() * 8;
    }

    // 2) entries
    for (key, v) in entries {
        buf.extend_from_slice(&key.to_le_bytes());
        buf.extend_from_slice(
            &u16::try_from(v.len())
                .expect("vec longer than u16::MAX")
                .to_le_bytes(),
        );
        for (score, id) in v {
            buf.extend_from_slice(&score.to_le_bytes());
            buf.extend_from_slice(&id.to_le_bytes());
        }
    }
    buf
}

fn read_u32(buf: &[u8], pos: usize) -> u32 {
    u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap())
}

pub fn read_seed_vec_from_bytes(buf: &[u8], i: usize) -> (u32, Vec<(f32, u32)>) {
    let off = read_u32(buf, i * 4) as usize;
    let key = read_u32(buf, off);
    let len = u16::from_le_bytes(buf[off + 4..off + 6].try_into().unwrap()) as usize;

    let data = &buf[off + 6..off + 6 + len * 8];
    let items = data
        .chunks_exact(8)
        .map(|c| {
            (
                f32::from_le_bytes(c[0..4].try_into().unwrap()),
                u32::from_le_bytes(c[4..8].try_into().unwrap()),
            )
        })
        .collect();
    (key, items)
}
