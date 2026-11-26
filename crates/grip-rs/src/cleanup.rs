use std::{
    fs,
    mem::DropGuard,
    path::{Path, PathBuf},
    ptr::replace,
    sync::{Arc, RwLock, atomic::Ordering},
};

use memmap2::{Advice, Mmap};

use crate::{
    IndexOutput, SparseVector, VectorsAndSketches,
    compact::{
        COMPACT_IDS_FILE, COMPACT_INDEX_FILE, COMPACT_SKETCHES_FILE, COMPACT_VECTORS_FILE, compact,
    },
    delta::{Delta, MAX_DELTA, load_delta, persist_batch_inserts},
    healer::heal,
    read_index, read_sketches,
    tombstone::{Tombstones, delete_batch, load_tombstones},
};

#[derive(Debug)]
pub struct MutableIndex {
    pub version: usize,
    pub index_output: IndexOutput,
    pub vectors_and_sketches: VectorsAndSketches,
    pub ids: Vec<u32>,
    pub ext_base: u32,
    pub vectors_mmap: Arc<Mmap>,
    pub tombstones: Tombstones,
    pub delta: RwLock<Delta>,
}

fn version_of(path: &Path, allow_tmp: bool) -> Option<u32> {
    let mut p = path
        .file_name()? // last component (None for paths ending in `..`)
        .to_str()? // None if not valid UTF-8
        .strip_prefix('v')?;
    if allow_tmp {
        p = p.trim_suffix(".tmp");
    }
    p.parse().ok()
}

impl MutableIndex {
    pub fn load(
        base_dir: &Path,
        version: usize,
        vocab_size: usize,
        is_tmp: bool,
    ) -> Result<Self, String> {
        let base_name = if is_tmp {
            format!("v{version}.tmp")
        } else {
            format!("v{version}")
        };
        let index_dir = base_dir.join(base_name);
        let vectors_file =
            fs::File::open(index_dir.join(COMPACT_VECTORS_FILE)).map_err(|e| e.to_string())?;
        let vectors_mmap = unsafe { Mmap::map(&vectors_file).map_err(|e| e.to_string())? };
        let index_output = read_index(index_dir.join(COMPACT_INDEX_FILE), vocab_size)
            .map_err(|e| e.to_string())?;
        let vectors_and_sketches = read_sketches(index_dir.join(COMPACT_SKETCHES_FILE))?;
        let n_main = vectors_and_sketches.offsets.len() - 1;
        let (ids, ext_base) = if version > 0 {
            let id_bytes = fs::read(index_dir.join(COMPACT_IDS_FILE)).map_err(|e| e.to_string())?;
            if id_bytes.len() != (n_main + 1) * 4 {
                return Err(format!(
                    "Wrong number of stored ids in ids.bin: got {}, expected {}",
                    id_bytes.len() / 4 - 1,
                    n_main
                ));
            }
            (
                id_bytes[4..]
                    .chunks_exact(4)
                    .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
                    .collect::<Vec<u32>>(),
                u32::from_le_bytes(id_bytes[0..4].try_into().unwrap()),
            )
        } else {
            ((0..n_main).map(|i| i as u32).collect(), n_main as u32)
        };
        let delta = load_delta(&index_dir, ext_base)?;
        let tombstones = Tombstones::with_capacity(n_main + MAX_DELTA);
        load_tombstones(&tombstones, &index_dir).map_err(|e| e.to_string())?;

        Ok(Self {
            version,
            index_output,
            vectors_and_sketches,
            ids,
            vectors_mmap: Arc::new(vectors_mmap),
            tombstones,
            delta: RwLock::new(delta),
            ext_base,
        })
    }

    pub fn load_from(index_dir: &Path, vocab_size: usize) -> Result<Self, String> {
        // allows for tmp
        let ver = version_of(index_dir, true);
        if let Some(v) = ver {
            Self::load(
                index_dir
                    .parent()
                    .ok_or_else(|| "Could not find parent of index directory".to_string())?,
                v as usize,
                vocab_size,
                index_dir
                    .file_name()
                    .ok_or_else(|| "Index dir does not have a base name".to_string())?
                    .to_str()
                    .ok_or_else(|| "Index dir does not contain valid UTF-8 text".to_string())?
                    .ends_with(".tmp"),
            )
        } else {
            Err("Provided index directory is not versioned".into())
        }
    }

    pub fn from_env(base_dir: &Path, vocab_size: usize, delete_tmp: bool) -> Result<Self, String> {
        let entries = fs::read_dir(base_dir).map_err(|e| e.to_string())?;
        let mut current_version: usize = 0;
        for e in entries {
            let e = e.map_err(|er| er.to_string())?;
            let p = e.path();
            if p.is_dir()
                && let Some(v) = version_of(&p, false)
            {
                if (v as usize) > current_version {
                    current_version = v as usize;
                }
            } else if p.is_dir()
                && p.file_name()
                    .ok_or_else(|| "Path does not have a file name".to_string())?
                    .to_str()
                    .ok_or_else(|| "Path does not contain valid UTF-8 text".to_string())?
                    .ends_with(".tmp")
                && delete_tmp
            {
                fs::remove_dir_all(&p).map_err(|e| e.to_string())?;
            }
        }

        Self::load(base_dir, current_version, vocab_size, false)
    }

    /// The directory of this version, inside the base directory.
    pub fn dir(&self, base_dir: &Path) -> PathBuf {
        base_dir.join(format!("v{}", self.version))
    }

    /// Turns an id the user sees into an internal id.
    /// `delta_len` is the current number of delta documents.
    /// Returns `None` if the document does not exist (anymore).
    pub fn internal_id(&self, external: u32, delta_len: usize) -> Option<u32> {
        if external >= self.ext_base {
            // a delta document
            let pos = (external - self.ext_base) as usize;
            (pos < delta_len).then(|| (self.ids.len() + pos) as u32)
        } else {
            // a main document: `ids` is sorted
            self.ids.binary_search(&external).ok().map(|i| i as u32)
        }
    }

    /// Turns an internal id (what the search returns) into the id the user sees.
    pub fn external_id(&self, internal: u32) -> u32 {
        match self.ids.get(internal as usize) {
            // a main document
            Some(id) => *id,
            // a delta document: `ids` holds the main documents only
            None => self.ext_base + (internal - self.ids.len() as u32),
        }
    }

    pub fn reload(&mut self, index_dir: &Path, vocab_size: usize) -> Result<(), String> {
        *self = Self::load_from(index_dir, vocab_size)?;
        Ok(())
    }

    pub fn copy_from(&mut self, other: Self) {
        self.delta = other.delta;
        self.ext_base = other.ext_base;
        self.ids = other.ids;
        self.index_output = other.index_output;
        self.tombstones = other.tombstones;
        self.vectors_mmap = other.vectors_mmap;
        self.vectors_and_sketches = other.vectors_and_sketches;
        self.version = other.version;
    }
}

// function cleanup(shared, base_dir):
//     # 1. under the read lock
//     take a snapshot of the delta and the tombstones
//     healed = heal(graph, snapshot tombstones)
//     compact(base_dir/vN+1.tmp, snapshot, healed)
//     new = load_from(base_dir/vN+1.tmp)
//
//     # 2. under the write lock, short
//     copy the late inserts and deletes into new            # see below
//     rename vN+1.tmp to vN+1
//     replace the old index with new
//

pub struct CleanupParams {
    pub robust_prune_alpha: f32,
    pub max_dfs: usize,
    pub max_border_length: usize,
    pub vocab_size: usize,
    pub sketch_size: usize,
    pub ef: usize,
    pub patience: usize,
    pub max_query_dims: usize,
    pub entrypoint_seeds: usize,
}

impl CleanupParams {
    pub fn from_defaults(vocab_size: usize) -> Self {
        Self {
            robust_prune_alpha: 1.0,
            max_border_length: 2000,
            max_dfs: 1000,
            vocab_size,
            sketch_size: 80,
            ef: 100,
            patience: 20,
            max_query_dims: 2,
            entrypoint_seeds: 64,
        }
    }
}

pub fn cleanup(
    mutable_index: Arc<RwLock<MutableIndex>>,
    base_dir: &Path,
    params: &CleanupParams,
) -> Result<(), String> {
    let rlock_mi = mutable_index.read().map_err(|e| e.to_string())?;
    let rlock_delta = rlock_mi.delta.read().map_err(|e| e.to_string())?;
    let delta_snap = rlock_delta.docs.clone();
    let delta_snap_len = delta_snap.len();
    drop(rlock_delta);
    let tombstones_snap = rlock_mi.tombstones.make_copy();
    let healed = heal(
        &rlock_mi.index_output.graph,
        &tombstones_snap,
        params.robust_prune_alpha,
        params.max_dfs,
        params.max_border_length,
        params.vocab_size,
        &rlock_mi.vectors_mmap,
    )?;
    let index_dir_tmp_v = base_dir.join(format!("v{}.tmp", rlock_mi.version + 1));
    let index_dir_tmp = DropGuard::new(index_dir_tmp_v, |v| {
        // try deleting on drop.
        // if it fails, we shouldn't make it
        // a big deal. We can have a sweeper
        // that cleans leftover tmps on a regular
        // schedule, and restarting the server
        // with MutableIndex::from_env and delete_tmp = true
        // deletes leftover tmps
        let _ = fs::remove_dir_all(v);
    });
    let index_dir = base_dir.join(format!("v{}", rlock_mi.version + 1));
    let (n_main_new, new_ext_base) = compact(
        &index_dir_tmp,
        &tombstones_snap,
        &delta_snap,
        rlock_mi.ext_base,
        &rlock_mi.ids,
        &rlock_mi.vectors_mmap,
        &rlock_mi.vectors_and_sketches,
        &rlock_mi.index_output,
        &healed,
        params.vocab_size,
        params.sketch_size,
        params.entrypoint_seeds,
        params.robust_prune_alpha,
        params.ef,
        params.patience,
        params.max_query_dims,
    )?;

    let n_main_old = rlock_mi.ids.len();

    drop(rlock_mi);

    let new = MutableIndex::load_from(&index_dir_tmp, params.vocab_size)?;
    let snap_total = n_main_old + delta_snap_len;
    let mut old_to_new = vec![None; snap_total];
    let mut n_live = 0_usize;
    for old in 0..snap_total {
        if !tombstones_snap.is_deleted(old as u32) {
            old_to_new[old] = Some(n_live);
            n_live += 1;
        }
    }

    // needs write lock so we don't
    // add or delete anything while
    // backfilling intermediate updates
    // and deletions
    let mut wlocked_mi = mutable_index.write().map_err(|e| e.to_string())?;
    let rlocked_old_delta = wlocked_mi.delta.read().map_err(|e| e.to_string())?;
    let old_tombstones = wlocked_mi.tombstones.make_copy();
    let vectors_to_insert = (delta_snap_len..rlocked_old_delta.docs.len())
        .map(|i| rlocked_old_delta.docs[i].clone())
        .collect::<Vec<Arc<SparseVector>>>();
    persist_batch_inserts(vectors_to_insert, &new.delta, &index_dir_tmp, new_ext_base)
        .map_err(|e| e.to_string())?;

    let deleted_ids = old_tombstones
        .words
        .iter()
        .enumerate()
        .map(|(w, cur_unloaded)| {
            let snap = tombstones_snap.words[w].load(Ordering::Relaxed);
            let cur = cur_unloaded.load(Ordering::Relaxed);
            let mut new_bits = cur & !snap;
            let mut ids_int = vec![];
            while new_bits != 0 {
                let bit = new_bits.trailing_zeros();
                let old_internal = w as u32 * 64 + bit;
                ids_int.push(old_internal);
                new_bits &= new_bits - 1;
            }

            ids_int
        })
        .flatten()
        .collect::<Vec<u32>>();

    let current_docs = n_main_new + rlocked_old_delta.docs.len() - delta_snap_len;

    delete_batch(&deleted_ids, &new.tombstones, &index_dir_tmp, current_docs)
        .map_err(|e| e.to_string())?;

    #[cfg(target_family = "unix")]
    new.vectors_mmap
        .advise(Advice::WillNeed)
        .map_err(|e| e.to_string())?;

    fs::rename(&*index_dir_tmp, &index_dir).map_err(|e| e.to_string())?;

    drop(rlocked_old_delta);

    let old;
    unsafe {
        old = replace(&mut *wlocked_mi, new);
    }

    drop(wlocked_mi);
    drop(old);

    Ok(())
}
