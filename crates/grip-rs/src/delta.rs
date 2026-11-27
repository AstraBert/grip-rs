use std::{
    collections::HashMap,
    error::Error,
    fmt::Display,
    fs,
    io::{self, BufWriter, Write},
    path::Path,
    sync::{Arc, RwLock},
    time::{Duration, Instant},
};

use crate::{SparseVector, partition::build_inverted_index, tombstone::Tombstones};

pub const MAX_DELTA: usize = 50_000;
pub const DELTA_LOG_FILE: &'static str = "delta.log";
/// An insert slower than this is logged, with the time of each stage.
const SLOW_INSERT: Duration = Duration::from_millis(100);

fn ms(from: Instant, to: Instant) -> f32 {
    to.duration_since(from).as_secs_f32() * 1000.0
}

#[derive(Debug)]
pub enum DeltaError {
    IO(String),
    IsFull,
    Validation(String),
    Mutex,
}

impl Display for DeltaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IO(s) => write!(f, "IO error: {}", s),
            Self::IsFull => write!(f, "Delta is full"),
            Self::Validation(s) => write!(f, "Validation error: {}", s),
            Self::Mutex => write!(f, "The RwLock holding the delta is poisoned"),
        }
    }
}

impl Error for DeltaError {}

#[derive(Debug, Clone)]
pub struct Delta {
    pub docs: Vec<Arc<SparseVector>>,
    pub postings: HashMap<u32, Vec<(u32, f32)>>,
}

impl Default for Delta {
    fn default() -> Self {
        Self {
            docs: Vec::with_capacity(MAX_DELTA),
            postings: HashMap::new(),
        }
    }
}

pub fn delta_insert(delta: &mut Delta, vector: Arc<SparseVector>) {
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
}

pub fn delta_search(
    delta: &Delta,
    query: &SparseVector,
    k: usize,
    tombstones: &Tombstones,
    internal_base: u32,
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
    touched.retain(|v| !tombstones.is_deleted(*v + internal_base));
    touched.sort_by(|a, b| acc[*b as usize].total_cmp(&acc[*a as usize]));
    let results = touched
        .iter()
        .take(k)
        .map(|v| (*v + internal_base, acc[*v as usize]))
        .collect::<Vec<(u32, f32)>>();
    for t in &touched {
        acc[*t as usize] = 0.0;
    }
    results
}

pub fn persist_delta_insert(
    indices: Vec<u32>,
    values: Vec<f32>,
    delta: &RwLock<Delta>,
    log_dir: &Path,
    ext_base: u32,
) -> Result<u32, DeltaError> {
    let t0 = Instant::now();
    let validated = SparseVector::new(indices, values).map_err(|e| DeltaError::Validation(e))?;
    let t_validated = Instant::now();
    let mut delta_m = delta.write().map_err(|_| DeltaError::Mutex)?;
    let t_locked = Instant::now();
    if delta_m.docs.len() >= MAX_DELTA {
        drop(delta_m);
        return Err(DeltaError::IsFull);
    }
    let ext_id = ext_base + delta_m.docs.len() as u32;
    let mut content: Vec<u8> = Vec::with_capacity(validated.indices.len() * 8 + 8);
    content.extend_from_slice(
        &u32::try_from(validated.indices.len())
            .expect("Should convert to u32")
            .to_le_bytes(),
    );
    for i in validated.indices.iter() {
        content.extend_from_slice(&(i.to_le_bytes()));
    }
    for v in validated.values.iter() {
        content.extend_from_slice(&v.to_le_bytes());
    }
    content.extend_from_slice(&ext_id.to_le_bytes());
    let mut op = fs::OpenOptions::new()
        .append(true)
        .create(true)
        .open(log_dir.join(DELTA_LOG_FILE))
        .map_err(|e| DeltaError::IO(format!("Could not open delta file: {e}")))?;
    op.write_all(&content)
        .map_err(|e| DeltaError::IO(format!("Error while writing to file: {e}")))?;
    op.flush()
        .map_err(|e| DeltaError::IO(format!("Error while flushing: {e}")))?;
    let t_written = Instant::now();
    delta_insert(&mut delta_m, Arc::new(validated));
    drop(delta_m);
    let t_end = Instant::now();
    // a slow insert: say which stage was slow
    if t_end.duration_since(t0) > SLOW_INSERT {
        log::warn!(
            "[SLOW INSERT] {:.1}ms: validate {:.1}ms, wait for the delta lock {:.1}ms, \
             write the log {:.1}ms, add to the delta {:.1}ms",
            ms(t0, t_end),
            ms(t0, t_validated),
            ms(t_validated, t_locked),
            ms(t_locked, t_written),
            ms(t_written, t_end),
        );
    }
    Ok(ext_id)
}

pub fn persist_batch_inserts(
    vectors: Vec<Arc<SparseVector>>,
    delta: &RwLock<Delta>,
    log_dir: &Path,
    ext_base: u32,
) -> Result<Vec<u32>, DeltaError> {
    let mut delta_m = delta.write().map_err(|_| DeltaError::Mutex)?;
    if delta_m.docs.len() >= MAX_DELTA {
        drop(delta_m);
        return Err(DeltaError::IsFull);
    }
    let op = fs::OpenOptions::new()
        .append(true)
        .create(true)
        .open(log_dir.join(DELTA_LOG_FILE))
        .map_err(|e| DeltaError::IO(format!("Could not open delta file: {e}")))?;
    let mut writer = BufWriter::new(op);
    let mut ids = vec![];

    for v in vectors {
        let ext_id = ext_base + delta_m.docs.len() as u32;
        let mut content: Vec<u8> = Vec::with_capacity(v.indices.len() * 8 + 8);
        content.extend_from_slice(
            &u32::try_from(v.indices.len())
                .expect("Should convert to u32")
                .to_le_bytes(),
        );
        for i in v.indices.iter() {
            content.extend_from_slice(&(i.to_le_bytes()));
        }
        for v in v.values.iter() {
            content.extend_from_slice(&v.to_le_bytes());
        }
        content.extend_from_slice(&ext_id.to_le_bytes());
        writer
            .write_all(&content)
            .map_err(|e| DeltaError::IO(e.to_string()))?;
        delta_insert(&mut delta_m, v);
        ids.push(ext_id);
    }

    writer.flush().map_err(|e| DeltaError::IO(e.to_string()))?;

    Ok(ids)
}

fn read_u32(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(b[at..at + 4].try_into().unwrap())
}

fn read_record(buf: &[u8], pos: usize) -> Option<(SparseVector, u32, usize)> {
    if pos + 4 > buf.len() {
        // reached end
        return None;
    }

    let nnz = read_u32(buf, pos);
    let idx_start = pos + 4;
    let val_start = idx_start + 4 * (nnz as usize);
    let id_start = val_start + 4 * (nnz as usize);
    let end = id_start + 4;

    if end > buf.len() {
        // truncated record
        return None;
    }

    let indices = buf[idx_start..val_start]
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect::<Vec<u32>>();
    let values = buf[val_start..id_start]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect::<Vec<f32>>();
    let ext_id = read_u32(buf, id_start);

    Some((SparseVector { indices, values }, ext_id, end))
}

pub fn load_delta(log_dir: &Path, ext_base: u32) -> Result<Delta, String> {
    let content = match fs::read(log_dir.join(DELTA_LOG_FILE)) {
        Ok(c) => c,
        Err(e) => match e.kind() {
            io::ErrorKind::NotFound => return Ok(Delta::default()),
            _ => return Err(format!("An error occurred while reading the file: {e}")),
        },
    };
    let mut pos = 0;
    let mut docs: Vec<Arc<SparseVector>> = vec![];
    loop {
        let opt = read_record(&content, pos);
        if let Some((v, i, new_pos)) = opt {
            if i != ext_base + docs.len() as u32 {
                return Err(format!(
                    "Unexpected external ID: got {i}, expected {}",
                    ext_base + docs.len() as u32
                ));
            }
            if docs.len() == MAX_DELTA {
                return Err(format!("The log records exceed {MAX_DELTA}"));
            }
            pos = new_pos;
            docs.push(Arc::new(v));
        } else {
            if pos < content.len() {
                let fl = fs::OpenOptions::new()
                    .write(true)
                    .open(log_dir.join(DELTA_LOG_FILE))
                    .map_err(|e| {
                        format!(
                            "Error when attempting to open the delta log file for truncation: {e}"
                        )
                    })?;
                fl.set_len(pos as u64).map_err(|e| {
                    format!("Error when attempting to truncate the delta log file to {pos}: {e}")
                })?;
                drop(fl);
            }
            let postings = build_inverted_index(
                &docs
                    .iter()
                    .map(|v| v.as_ref())
                    .collect::<Vec<&SparseVector>>(),
                // uses local ids
                &docs
                    .iter()
                    .enumerate()
                    .map(|(i, _)| i as u32)
                    .collect::<Vec<u32>>(),
            );
            return Ok(Delta { docs, postings });
        }
    }
}
