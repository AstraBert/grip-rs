use std::{fs, io, path::Path};

use memmap2::Mmap;

use crate::{
    IndexOutput, QuantizedSparseVector, SparseVector, VectorsAndSketches, Visited,
    compute_entry_points,
    delta::Delta,
    disk::decode_sparse_vectors_raw,
    make_routing_sketch,
    robustprune::{MAX_GRAPH_DEGREE, robust_prune},
    run_beam_search, save_index, save_vectors_and_sketches,
    seeding::{SeedVecs, build_seed_list},
    tombstone::Tombstones,
};

pub const COMPACT_INDEX_FILE: &str = "index.bin";
pub const COMPACT_VECTORS_FILE: &str = "vectors.bin";
pub const COMPACT_SKETCHES_FILE: &str = "sketches.bin";
/// New position -> external id, 4 bytes little-endian each.
pub const COMPACT_IDS_FILE: &str = "ids.bin";

pub struct IndexVersion {
    pub n: usize,
}

fn version_of(path: &Path) -> Option<u32> {
    path.file_name()? // last component (None for paths ending in `..`)
        .to_str()? // None if not valid UTF-8
        .strip_prefix('v')? // None unless it starts with 'v'
        .parse() // None unless the rest is all digits
        .ok()
}

impl IndexVersion {
    pub fn new() -> Self {
        Self { n: 0 }
    }

    pub fn from_env(base_dir: &Path) -> io::Result<Self> {
        let entries = fs::read_dir(base_dir)?;
        let mut current_version: usize = 0;
        for e in entries {
            let e = e?;
            let p = e.path();
            if p.is_dir() {
                let v = version_of(&p);
                if let Some(ver) = v
                    && current_version < ver as usize
                {
                    current_version = ver as usize;
                }
            }
        }

        Ok(Self { n: current_version })
    }

    pub fn index_dir(&self) -> String {
        format!("v{}", self.n)
    }

    pub fn increase(&mut self) {
        self.n += 1;
    }
}

/// Packs sketches in the flat layout the beam search reads.
fn pack_sketches(sketches: &[QuantizedSparseVector], scale: f32) -> VectorsAndSketches {
    let capacity: usize = sketches.iter().map(|s| s.indices.len()).sum();
    let mut offsets = Vec::with_capacity(sketches.len() + 1);
    let mut indices = Vec::with_capacity(capacity);
    let mut values = Vec::with_capacity(capacity);
    for s in sketches {
        offsets.push(indices.len());
        indices.extend_from_slice(&s.indices);
        values.extend_from_slice(&s.values);
    }
    offsets.push(indices.len());
    VectorsAndSketches {
        offsets,
        indices,
        values,
        scale,
    }
}

/// Writes a new index into `out_dir`, without the deleted documents and with
/// the delta documents merged into the graph. `index_dir` is never touched, so
/// swapping the new directory in is up to the caller.
///
/// `index_output.graph` should be the healed graph: edges to deleted nodes are
/// dropped here, not repaired.
///
/// Ids are renumbered in old-id order (main documents first, then delta
/// documents). `ids.bin` stores the old id of every new position. It assumes
/// that old ids are the external ids, which holds until the first compaction.
///
/// Returns the number of documents in the new index.
pub fn compact(
    out_dir: &Path,
    tombstones: &Tombstones,
    delta: &Delta,
    n_main: usize,
    vectors_mmap: &Mmap,
    vectors_and_sketches: &VectorsAndSketches,
    index_output: &IndexOutput,
    vocab_size: usize,
    sketch_size: usize,
    entrypoint_seeds: usize,
    prune_alpha: f32,
    ef: usize,
    patience: usize,
    max_query_dims: usize,
) -> Result<usize, String> {
    if index_output.graph.len() != n_main * MAX_GRAPH_DEGREE {
        return Err(format!(
            "The graph has {} rows, expected {n_main}",
            index_output.graph.len() / MAX_GRAPH_DEGREE
        ));
    }
    if out_dir.join(COMPACT_INDEX_FILE).exists() {
        return Err(format!(
            "{} already holds an index, refusing to overwrite it",
            out_dir.display()
        ));
    }

    // 1. renumbering table: old id (main, then delta) -> new id
    let n = n_main + delta.docs.len();
    let mut old_to_new: Vec<Option<u32>> = vec![None; n];
    let mut n_live = 0usize;
    for old in 0..n {
        if !tombstones.is_deleted(old as u32) {
            old_to_new[old] = Some(n_live as u32);
            n_live += 1;
        }
    }
    if n_live == 0 {
        return Err("Every document is deleted, there is nothing to compact".to_string());
    }

    // 2. vectors and sketches, in new-id order
    let live_main: Vec<usize> = (0..n_main).filter(|o| old_to_new[*o].is_some()).collect();
    let main_raw = decode_sparse_vectors_raw(vectors_mmap, Some(live_main.clone()))?;

    let mut new_vectors: Vec<SparseVector> = Vec::with_capacity(n_live);
    let mut new_sketches: Vec<QuantizedSparseVector> = Vec::with_capacity(n_live);
    for (old, raw) in live_main.iter().zip(main_raw.iter()) {
        new_vectors.push(SparseVector::from_id_value_pairs_unchecked(raw));
        // the sketch only depends on its own vector: copy it
        let (idx, val) = vectors_and_sketches
            .get(*old)
            .ok_or_else(|| format!("No sketch for the document {old}"))?;
        new_sketches.push(QuantizedSparseVector::from_slices_unchecked(idx, val));
    }
    drop(main_raw);
    for old in n_main..n {
        if old_to_new[old].is_none() {
            continue;
        }
        // delta documents have no sketch yet
        let vector = delta.docs[old - n_main].clone();
        new_sketches.push(make_routing_sketch(
            &vector,
            sketch_size,
            vectors_and_sketches.scale,
        )?);
        new_vectors.push(vector);
    }

    // 3. graph rows of the main documents: translate ids, drop deleted ones,
    // keep the padding at the end of the row
    let mut new_graph = vec![u32::MAX; n_live * MAX_GRAPH_DEGREE];
    for old in 0..n_main {
        let Some(new_id) = old_to_new[old] else {
            continue;
        };
        let row = &mut new_graph
            [new_id as usize * MAX_GRAPH_DEGREE..(new_id as usize + 1) * MAX_GRAPH_DEGREE];
        let mut written = 0;
        for x in &index_output.graph[old * MAX_GRAPH_DEGREE..(old + 1) * MAX_GRAPH_DEGREE] {
            if *x == u32::MAX {
                continue;
            }
            if let Some(Some(new_x)) = old_to_new.get(*x as usize) {
                row[written] = *new_x;
                written += 1;
            }
        }
    }

    // 4. graph rows of the delta documents: insert them one by one.
    // The search starts from the old seeds, translated to the new ids, and
    // only reaches nodes that are already in the graph.
    let entry_seeds: SeedVecs = index_output
        .seeds
        .iter()
        .map(|list| {
            list.iter()
                .filter_map(|(w, id)| {
                    old_to_new
                        .get(*id as usize)
                        .copied()
                        .flatten()
                        .map(|new_id| (*w, new_id))
                })
                .collect()
        })
        .collect();

    let packed = pack_sketches(&new_sketches, vectors_and_sketches.scale);
    let no_tombstones = Tombstones::with_capacity(0);
    let borrowed: Vec<&SparseVector> = new_vectors.iter().collect();
    let mut visited = Visited {
        counter: 0,
        visited: vec![0; n_live],
    };
    let mut dense = vec![0_u8; vocab_size];
    let mut dense_prune = vec![0_f32; vocab_size];

    for old in n_main..n {
        let Some(new_id) = old_to_new[old] else {
            continue;
        };
        if new_id == 0 {
            // no main document survived and this is the first delta one:
            // the graph is empty, there is nobody to connect to
            continue;
        }
        let me = new_id as usize;

        let q8 = &new_sketches[me];
        for (t, w) in q8.indices.iter().zip(q8.values.iter()) {
            dense[*t as usize] = *w;
        }
        let mut by_impact = q8.clone();
        let mut entry_points = compute_entry_points(
            &mut by_impact,
            &entry_seeds,
            &index_output.term_max,
            max_query_dims,
        );
        let found = run_beam_search(
            q8,
            &mut visited,
            &mut entry_points,
            &packed,
            &mut dense,
            &new_graph,
            ef,
            patience,
            &no_tombstones,
        )?;

        // exact scores, like the refine step of the build
        let pool: Vec<(u32, f32)> = found
            .iter()
            .map(|f| f.1)
            .filter(|id| *id != new_id)
            .map(|id| (id, new_vectors[me].dot(&new_vectors[id as usize])))
            .collect();

        let row = robust_prune::<f32>(pool, prune_alpha, &borrowed, &mut dense_prune);
        new_graph[me * MAX_GRAPH_DEGREE..(me + 1) * MAX_GRAPH_DEGREE].copy_from_slice(&row);

        // reverse edges: let the neighbours point back to the new node
        for nb in row.iter() {
            if *nb == u32::MAX {
                break;
            }
            let nb_row = &mut new_graph
                [*nb as usize * MAX_GRAPH_DEGREE..(*nb as usize + 1) * MAX_GRAPH_DEGREE];
            if let Some(pos) = nb_row.iter().position(|x| *x == u32::MAX) {
                nb_row[pos] = new_id;
            } else {
                // the row is full: prune its neighbours plus the new node,
                // scored against the neighbour that owns the row
                let owner = &new_vectors[*nb as usize];
                let mut candidates: Vec<(u32, f32)> = nb_row
                    .iter()
                    .map(|c| (*c, owner.dot(&new_vectors[*c as usize])))
                    .collect();
                candidates.push((new_id, owner.dot(&new_vectors[me])));
                let kept =
                    robust_prune::<f32>(candidates, prune_alpha, &borrowed, &mut dense_prune);
                nb_row.copy_from_slice(&kept);
            }
        }
    }

    // 5. fresh seeds and term maxima, from the documents that are left
    let (seeds, term_max) = build_seed_list(&borrowed, entrypoint_seeds, vocab_size);
    drop(borrowed);

    // 6. ids.bin: new position -> old id
    let mut ids = vec![0_u32; n_live];
    for old in 0..n {
        if let Some(new_id) = old_to_new[old] {
            ids[new_id as usize] = old as u32;
        }
    }
    let ids_bytes: Vec<u8> = ids.iter().flat_map(|i| i.to_le_bytes()).collect();

    // 7. write everything. There are no delete.log and delta.log: a missing
    // log means "empty".
    fs::create_dir_all(out_dir).map_err(|e| e.to_string())?;
    save_vectors_and_sketches(
        new_vectors,
        new_sketches,
        vectors_and_sketches.scale,
        out_dir.join(COMPACT_VECTORS_FILE),
        out_dir.join(COMPACT_SKETCHES_FILE),
    )
    .map_err(|e| e.to_string())?;
    save_index(new_graph, seeds, term_max, out_dir.join(COMPACT_INDEX_FILE))
        .map_err(|e| e.to_string())?;
    fs::write(out_dir.join(COMPACT_IDS_FILE), ids_bytes).map_err(|e| e.to_string())?;

    Ok(n_live)
}
