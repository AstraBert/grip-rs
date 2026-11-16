use std::collections::HashSet;

use num_traits::{FromBytes, ToBytes};
use serde::{Deserialize, Serialize};

pub trait Vector<'a, T: ToBytes + FromBytes> {
    fn indices(&'a self) -> &'a [u32];
    fn values(&'a self) -> &'a [T];
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseVector {
    pub indices: Vec<u32>,
    pub values: Vec<f32>,
}

impl<'a> Vector<'a, f32> for SparseVector {
    fn indices(&'a self) -> &'a [u32] {
        &self.indices
    }

    fn values(&'a self) -> &'a [f32] {
        &self.values
    }
}

impl<'a> Vector<'a, u8> for QuantizedSparseVector {
    fn indices(&'a self) -> &'a [u32] {
        &self.indices
    }

    fn values(&'a self) -> &'a [u8] {
        &self.values
    }
}

pub enum SparseVectorSortKey {
    IndicesAsc,
    IndicesDesc,
    ValuesAsc,
    ValuesDesc,
}

impl SparseVector {
    pub fn new(indices: Vec<u32>, values: Vec<f32>) -> Result<Self, String> {
        if indices.len() != values.len() {
            return Err(format!(
                "Expected indices and values to contain the same number of elements, got ({}, {})",
                indices.len(),
                values.len()
            ));
        }
        if values.iter().any(|v| v.is_nan()) {
            return Err("values must not contain NaN".to_string());
        }

        let mut pairs: Vec<(u32, f32)> = indices.into_iter().zip(values).collect();
        pairs.sort_by_key(|&(id, _)| id);

        let mut out_indices: Vec<u32> = Vec::with_capacity(pairs.len());
        let mut out_values: Vec<f32> = Vec::with_capacity(pairs.len());
        for (id, value) in pairs {
            if out_indices.last() == Some(&id) {
                *out_values.last_mut().unwrap() += value; // duplicate id: add the values
            } else {
                out_indices.push(id);
                out_values.push(value);
            }
        }

        // A merge can produce a zero, so drop zeros after merging, not before.
        let (indices, values): (Vec<u32>, Vec<f32>) = out_indices
            .into_iter()
            .zip(out_values)
            .filter(|&(_, v)| v != 0.0)
            .unzip();

        Ok(SparseVector { indices, values })
    }

    pub fn is_optimal(&self) -> bool {
        self.indices.len() == self.values.len()
            && !self.values.iter().any(|v| *v == 0_f32 || v.is_nan())
            && self.indices.iter().is_sorted()
            && self.indices.iter().collect::<HashSet<&u32>>().len() == self.indices.len()
    }

    pub fn make_optimal(&mut self) -> Result<(), String> {
        if self.indices.len() != self.values.len() {
            return Err(format!(
                "Expected indices and values to contain the same number of elements, got ({}, {})",
                self.indices.len(),
                self.values.len()
            ));
        }
        if self.values.iter().any(|v| v.is_nan()) {
            return Err("values must not contain NaN".to_string());
        }

        let mut pairs: Vec<(u32, f32)> = self
            .indices
            .iter()
            .zip(self.values.iter())
            .map(|(i, v)| (*i, *v))
            .collect();
        pairs.sort_by_key(|&(id, _)| id);

        let mut out_indices: Vec<u32> = Vec::with_capacity(pairs.len());
        let mut out_values: Vec<f32> = Vec::with_capacity(pairs.len());
        for (id, value) in pairs {
            if out_indices.last() == Some(&id) {
                *out_values.last_mut().unwrap() += value; // duplicate id: add the values
            } else {
                out_indices.push(id);
                out_values.push(value);
            }
        }

        // A merge can produce a zero, so drop zeros after merging, not before.
        let (indices, values): (Vec<u32>, Vec<f32>) = out_indices
            .into_iter()
            .zip(out_values)
            .filter(|&(_, v)| v != 0.0)
            .unzip();

        self.indices = indices;
        self.values = values;

        Ok(())
    }

    pub fn dot(&self, other: &SparseVector) -> f32 {
        let (mut i, mut j) = (0, 0);
        let mut sum = 0.0_f32;
        while i < self.indices.len() && j < other.indices.len() {
            match self.indices[i].cmp(&other.indices[j]) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    sum += self.values[i] * other.values[j];
                    i += 1;
                    j += 1;
                }
            }
        }
        sum
    }

    pub fn sort_by(&mut self, key: SparseVectorSortKey) {
        match key {
            SparseVectorSortKey::IndicesAsc => {
                let mut positions = (0..self.indices.len()).collect::<Vec<usize>>();
                positions.sort_by(|pa, pb| self.indices[*pa].cmp(&self.indices[*pb]));
                self.indices = positions.iter().map(|p| self.indices[*p]).collect();
                self.values = positions.iter().map(|p| self.values[*p]).collect();
            }
            SparseVectorSortKey::IndicesDesc => {
                let mut positions = (0..self.indices.len()).collect::<Vec<usize>>();
                positions.sort_by(|pa, pb| self.indices[*pb].cmp(&self.indices[*pa]));
                self.indices = positions.iter().map(|p| self.indices[*p]).collect();
                self.values = positions.iter().map(|p| self.values[*p]).collect();
            }
            SparseVectorSortKey::ValuesAsc => {
                let mut positions = (0..self.values.len()).collect::<Vec<usize>>();
                positions.sort_by(|pa, pb| self.values[*pa].total_cmp(&self.values[*pb]));
                self.indices = positions.iter().map(|p| self.indices[*p]).collect();
                self.values = positions.iter().map(|p| self.values[*p]).collect();
            }
            SparseVectorSortKey::ValuesDesc => {
                let mut positions = (0..self.values.len()).collect::<Vec<usize>>();
                positions.sort_by(|pa, pb| self.values[*pb].total_cmp(&self.values[*pa]));
                self.indices = positions.iter().map(|p| self.indices[*p]).collect();
                self.values = positions.iter().map(|p| self.values[*p]).collect();
            }
        }
    }

    pub fn truncate(&mut self, dim: usize) {
        self.indices.truncate(dim);
        self.values.truncate(dim);
    }

    pub fn quantize(&self, scale: f32) -> Result<QuantizedSparseVector, String> {
        let indices = self.indices.clone();
        let mut values: Vec<u8> = vec![];
        for v in self.values.iter() {
            let q = (*v / scale).round().min(255_f32);
            values.push(q as u8);
        }

        let mut pairs: Vec<(u32, u8)> = indices.into_iter().zip(values).collect();
        pairs.sort_by_key(|&(id, _)| id);

        let mut out_indices: Vec<u32> = Vec::with_capacity(pairs.len());
        let mut out_values: Vec<u8> = Vec::with_capacity(pairs.len());
        for (id, value) in pairs {
            if out_indices.last() == Some(&id) {
                *out_values.last_mut().unwrap() += value; // duplicate id: add the values
            } else {
                out_indices.push(id);
                out_values.push(value);
            }
        }

        let (indices, values): (Vec<u32>, Vec<u8>) = out_indices
            .into_iter()
            .zip(out_values)
            .filter(|&(_, v)| v != 0)
            .unzip();

        QuantizedSparseVector::new(indices, values)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizedSparseVector {
    pub indices: Vec<u32>,
    pub values: Vec<u8>,
}

impl QuantizedSparseVector {
    pub fn new(indices: Vec<u32>, values: Vec<u8>) -> Result<Self, String> {
        if indices.len() != values.len() {
            return Err(format!(
                "Expected indices and values to contain the same number of elements, got ({}, {})",
                indices.len(),
                values.len()
            ));
        }

        Ok(Self { indices, values })
    }

    pub fn dot_unscaled(&self, other: &QuantizedSparseVector) -> u32 {
        let (mut i, mut j) = (0, 0);
        let mut sum: u32 = 0;
        while i < self.indices.len() && j < other.indices.len() {
            match self.indices[i].cmp(&other.indices[j]) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    sum += self.values[i] as u32 * other.values[j] as u32;
                    i += 1;
                    j += 1;
                }
            }
        }
        sum
    }

    pub fn dot(&self, other: &QuantizedSparseVector, our_scale: f32, their_scale: f32) -> f32 {
        let sum = self.dot_unscaled(other);
        sum as f32 * our_scale * their_scale
    }

    pub fn sort_with_fn<F: Fn((u32, u8), (u32, u8)) -> std::cmp::Ordering>(&mut self, func: F) {
        let mut positions = (0..self.indices.len()).collect::<Vec<usize>>();
        positions.sort_by(|a, b| {
            func(
                (self.indices[*a], self.values[*a]),
                (self.indices[*b], self.values[*b]),
            )
        });
        self.indices = positions.iter().map(|p| self.indices[*p]).collect();
        self.values = positions.iter().map(|p| self.values[*p]).collect();
    }

    pub fn truncate(&mut self, dim: usize) {
        self.indices.truncate(dim);
        self.values.truncate(dim);
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BuildParams {
    pub vocab_size: usize,
    pub m: usize,
    pub seed: u64,
    pub leaf_k: usize,
    pub refine: bool,
    pub robust_prune_alpha: f32,
    pub entrypoint_seeds: usize,
    pub sketch_size: usize,
}

impl BuildParams {
    pub fn new(
        m: usize,
        seed: u64,
        vocab_size: usize,
        leaf_k: usize,
        refine: bool,
        robust_prune_alpha: f32,
        entrypoint_seeds: usize,
        sketch_size: usize,
    ) -> Result<Self, String> {
        let s = Self {
            m,
            seed,
            vocab_size,
            leaf_k,
            refine,
            robust_prune_alpha,
            entrypoint_seeds,
            sketch_size,
        };
        s.validate()?;
        Ok(s)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.m > 16 {
            return Err("m cannot be larger than 16".to_string());
        }
        Ok(())
    }
}

pub struct SearchParams {
    pub max_query_dims: usize,
    pub patience: usize,
    pub top_k: usize,
    pub ef: usize,
    pub rescore_factor: usize,
}

impl SearchParams {
    pub fn new(
        max_query_dims: usize,
        patience: usize,
        top_k: usize,
        ef: usize,
        rescore_factor: usize,
    ) -> Result<Self, String> {
        let s = Self {
            max_query_dims,
            patience,
            top_k,
            ef,
            rescore_factor,
        };
        s.validate()?;
        Ok(s)
    }

    fn validate(&self) -> Result<(), String> {
        if self.max_query_dims == 0 {
            return Err("max_query_dims needs to be at least 1".to_string());
        }
        if self.top_k == 0 {
            return Err("top_k needs to be at least 1".to_string());
        }
        if self.ef == 0 {
            return Err("ef needs to be at least 1".to_string());
        }
        if self.ef < self.top_k {
            return Err("ef cannot be less than top_k".to_string());
        }
        if self.rescore_factor == 0 {
            return Err("rescord_factor cannot be 0".to_string());
        }

        Ok(())
    }
}
