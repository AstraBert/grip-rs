use std::io::{self, Write};

use memmap2::Mmap;
use num_traits::{FromBytes, ToBytes};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::types::Vector;

pub const RAYON_THRESHOLD: usize = 100;

/// Writes the vectors in the on-disk format, one piece at a time.
/// Pass a `BufWriter` so the data does not need to be held in memory first.
pub fn write_vectors<'a, T: ToBytes + FromBytes + 'a, V: Vector<'a, T>>(
    out: &mut impl Write,
    entries: &'a [&V],
    value_size: usize,
    scale: Option<f32>,
) -> io::Result<()> {
    let header_len = 4 + 8 + entries.len() * 8;

    out.write_all(&(entries.len() as u64).to_le_bytes())?;
    out.write_all(&scale.unwrap_or(0.0).to_le_bytes())?;

    // 1) offset table
    let mut offset = header_len;
    for v in entries {
        out.write_all(&(offset as u64).to_le_bytes())?;
        offset += 2 + v.indices().len() * 4 + v.values().len() * value_size;
    }

    // 2) entries
    for v in entries {
        let len = u16::try_from(v.indices().len())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "vec longer than u16::MAX"))?;
        out.write_all(&len.to_le_bytes())?;
        for i in v.indices().iter() {
            out.write_all(&i.to_le_bytes())?;
        }
        for w in v.values().iter() {
            out.write_all(w.to_le_bytes().as_ref())?;
        }
    }
    Ok(())
}

pub fn decode_sparse_vectors_raw(
    blob: &Mmap,
    indices: Option<Vec<usize>>,
) -> Result<Vec<Vec<(u32, f32)>>, String> {
    let l = u64::from_le_bytes(blob[0..8].try_into().expect("Should convert to [u8; 8]"));
    let to_read = if let Some(idx) = indices {
        if idx.iter().any(|i| *i >= l as usize) {
            return Err(format!(
                "You are requesting vector IDs which are outside the number of available vectors ({l})"
            ));
        }
        idx
    } else {
        (0..l as usize).collect()
    };

    let vectors = if to_read.len() > RAYON_THRESHOLD {
        to_read
            .par_iter()
            .map(|i| {
                let offset_start = 12 + *i * 8;
                let offset = u64::from_le_bytes(
                    blob[offset_start..offset_start + 8]
                        .try_into()
                        .expect("Should convert to [u8; 8]"),
                );
                let vector_len = u16::from_le_bytes(
                    blob[offset as usize..offset as usize + 2]
                        .try_into()
                        .expect("Should convert to [u8; 2]"),
                );
                let indices_start = offset as usize + 2;
                let indices_end = offset as usize + 2 + vector_len as usize * 4;
                let indices_bytes = &blob[indices_start..indices_end];
                let values_bytes = &blob[indices_end..indices_end + vector_len as usize * 4];
                let mut indices: Vec<u32> = vec![];
                let mut values: Vec<f32> = vec![];
                for b in indices_bytes.chunks_exact(4) {
                    let idx = u32::from_le_bytes(b.try_into().expect("Should convert to [u8; 4] "));
                    indices.push(idx);
                }
                for v in values_bytes.chunks_exact(4) {
                    let val = f32::from_le_bytes(v.try_into().expect("Should convert to [u8; 4]"));
                    values.push(val);
                }
                indices
                    .iter()
                    .zip(values)
                    .map(|(i, v)| (*i, v))
                    .collect::<Vec<(u32, f32)>>()
            })
            .collect::<Vec<Vec<(u32, f32)>>>()
    } else {
        to_read
            .iter()
            .map(|i| {
                let offset_start = 12 + *i * 8;
                let offset = u64::from_le_bytes(
                    blob[offset_start..offset_start + 8]
                        .try_into()
                        .expect("Should convert to [u8; 8]"),
                );
                let vector_len = u16::from_le_bytes(
                    blob[offset as usize..offset as usize + 2]
                        .try_into()
                        .expect("Should convert to [u8; 2]"),
                );
                let indices_start = offset as usize + 2;
                let indices_end = offset as usize + 2 + vector_len as usize * 4;
                let indices_bytes = &blob[indices_start..indices_end];
                let values_bytes = &blob[indices_end..indices_end + vector_len as usize * 4];
                let mut indices: Vec<u32> = vec![];
                let mut values: Vec<f32> = vec![];
                for b in indices_bytes.chunks_exact(4) {
                    let idx = u32::from_le_bytes(b.try_into().expect("Should convert to [u8; 4] "));
                    indices.push(idx);
                }
                for v in values_bytes.chunks_exact(4) {
                    let val = f32::from_le_bytes(v.try_into().expect("Should convert to [u8; 4]"));
                    values.push(val);
                }
                indices
                    .iter()
                    .zip(values)
                    .map(|(i, v)| (*i, v))
                    .collect::<Vec<(u32, f32)>>()
            })
            .collect::<Vec<Vec<(u32, f32)>>>()
    };

    Ok(vectors)
}

pub fn decode_sparse_sketches(
    blob: &[u8],
    indices: Option<Vec<usize>>,
) -> Result<(Vec<usize>, Vec<u32>, Vec<u8>, f32), String> {
    let l = u64::from_le_bytes(blob[0..8].try_into().expect("Should convert to [u8; 8]"));
    let to_read = if let Some(idx) = indices {
        idx.iter()
            // indices should be less than
            // the number of vectors
            .filter(|i| **i < l as usize)
            .map(|i| *i)
            .collect::<Vec<usize>>()
    } else {
        (0..l as usize).collect()
    };

    let scale = f32::from_le_bytes(blob[8..12].try_into().expect("Should convert to [u8; 4]"));

    let vectors = if to_read.len() > RAYON_THRESHOLD {
        to_read
            .par_iter()
            .map(|i| {
                let offset_start = 12 + *i * 8;
                let offset = u64::from_le_bytes(
                    blob[offset_start..offset_start + 8]
                        .try_into()
                        .expect("Should convert to [u8; 8]"),
                );
                let vector_len = u16::from_le_bytes(
                    blob[offset as usize..offset as usize + 2]
                        .try_into()
                        .expect("Should convert to [u8; 2]"),
                );
                let indices_start = offset as usize + 2;
                let indices_end = offset as usize + 2 + vector_len as usize * 4;
                let indices_bytes = &blob[indices_start..indices_end];
                let values_bytes = &blob[indices_end..indices_end + vector_len as usize * 1];
                let mut indices: Vec<u32> = vec![];
                let mut values: Vec<u8> = vec![];
                for b in indices_bytes.chunks_exact(4) {
                    let idx = u32::from_le_bytes(b.try_into().expect("Should convert to [u8; 4] "));
                    indices.push(idx);
                }
                for v in values_bytes.chunks_exact(1) {
                    let val = u8::from_be_bytes(v.try_into().expect("Should convert to [u8; 4]"));
                    values.push(val);
                }
                (indices, values)
            })
            .collect::<Vec<(Vec<u32>, Vec<u8>)>>()
    } else {
        to_read
            .iter()
            .map(|i| {
                let offset_start = 12 + *i * 8;
                let offset = u64::from_le_bytes(
                    blob[offset_start..offset_start + 8]
                        .try_into()
                        .expect("Should convert to [u8; 8]"),
                );
                let vector_len = u16::from_le_bytes(
                    blob[offset as usize..offset as usize + 2]
                        .try_into()
                        .expect("Should convert to [u8; 2]"),
                );
                let indices_start = offset as usize + 2;
                let indices_end = offset as usize + 2 + vector_len as usize * 4;
                let indices_bytes = &blob[indices_start..indices_end];
                let values_bytes = &blob[indices_end..indices_end + vector_len as usize * 1];
                let mut indices: Vec<u32> = vec![];
                let mut values: Vec<u8> = vec![];
                for b in indices_bytes.chunks_exact(4) {
                    let idx = u32::from_le_bytes(b.try_into().expect("Should convert to [u8; 4] "));
                    indices.push(idx);
                }
                for v in values_bytes.chunks_exact(1) {
                    let val = u8::from_be_bytes(v.try_into().expect("Should convert to [u8; 4]"));
                    values.push(val);
                }
                (indices, values)
            })
            .collect::<Vec<(Vec<u32>, Vec<u8>)>>()
    };

    let capacity = vectors
        .iter()
        .map(|v| v.0.len())
        .collect::<Vec<usize>>()
        .iter()
        .sum();

    let mut offsets: Vec<usize> = vectors
        .iter()
        .scan(0, |running, v| {
            let start = *running;
            *running += v.0.len();
            Some(start)
        })
        .collect();

    offsets.push(capacity);

    let mut indices = Vec::with_capacity(capacity);
    let mut values = Vec::with_capacity(capacity);

    for v in vectors {
        for i in 0..v.0.len() {
            indices.push(v.0[i]);
            values.push(v.1[i]);
        }
    }

    Ok((offsets, indices, values, scale))
}
