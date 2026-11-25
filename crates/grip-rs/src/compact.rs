use std::{fs, path::Path};

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
/// internal position -> external id, 4 bytes little-endian each.
pub const COMPACT_IDS_FILE: &str = "ids.bin";

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

/// Writes a internal index into `out_dir`, without the deleted documents and with
/// the delta documents merged into the graph. `index_dir` is never touched, so
/// swapping the internal directory in is up to the caller.
///
/// `graph` should be the healed graph: edges to deleted nodes are
/// dropped here, not repaired.
///
/// Ids are renumbered in ext-id order (main documents first, then delta
/// documents). `ids.bin` stores the ext id of every internal position. It assumes
/// that ext ids are the external ids, which holds until the first compaction.
///
/// Returns the number of documents in the internal index.
pub fn compact(
    out_dir: &Path,
    tombstones: &Tombstones,
    delta: &Delta,
    ext_base: u32,
    old_ids: &[u32],
    vectors_mmap: &Mmap,
    vectors_and_sketches: &VectorsAndSketches,
    index_output: &IndexOutput,
    graph: &[u32],
    vocab_size: usize,
    sketch_size: usize,
    entrypoint_seeds: usize,
    prune_alpha: f32,
    ef: usize,
    patience: usize,
    max_query_dims: usize,
) -> Result<(usize, u32), String> {
    let n_main = old_ids.len();

    if graph.len() != n_main * MAX_GRAPH_DEGREE {
        return Err(format!(
            "The graph has {} rows, expected {n_main}",
            graph.len() / MAX_GRAPH_DEGREE
        ));
    }
    if out_dir.join(COMPACT_INDEX_FILE).exists() {
        return Err(format!(
            "{} already holds an index, refusing to overwrite it",
            out_dir.display()
        ));
    }

    // 1. renumbering table: ext id (main, then delta) -> internal id
    let n = n_main + delta.docs.len();
    let mut ext_to_internal: Vec<Option<u32>> = vec![None; n];
    let mut n_live = 0usize;
    for ext in 0..n {
        if !tombstones.is_deleted(ext as u32) {
            ext_to_internal[ext] = Some(n_live as u32);
            n_live += 1;
        }
    }
    if n_live == 0 {
        return Err("Every document is deleted, there is nothing to compact".to_string());
    }

    // 2. vectors and sketches, in internal-id order
    let live_main: Vec<usize> = (0..n_main)
        .filter(|o| ext_to_internal[*o].is_some())
        .collect();
    let main_raw = decode_sparse_vectors_raw(vectors_mmap, Some(live_main.clone()))?;

    let mut internal_vectors: Vec<SparseVector> = Vec::with_capacity(n_live);
    let mut internal_sketches: Vec<QuantizedSparseVector> = Vec::with_capacity(n_live);
    for (ext, raw) in live_main.iter().zip(main_raw.iter()) {
        internal_vectors.push(SparseVector::from_id_value_pairs_unchecked(raw));
        // the sketch only depends on its own vector: copy it
        let (idx, val) = vectors_and_sketches
            .get(*ext)
            .ok_or_else(|| format!("No sketch for the document {ext}"))?;
        internal_sketches.push(QuantizedSparseVector::from_slices_unchecked(idx, val));
    }
    drop(main_raw);
    for ext in n_main..n {
        if ext_to_internal[ext].is_none() {
            continue;
        }
        // delta documents have no sketch yet
        let vector = delta.docs[ext - n_main].clone();
        internal_sketches.push(make_routing_sketch(
            &vector,
            sketch_size,
            vectors_and_sketches.scale,
        )?);
        internal_vectors.push(vector);
    }

    // 3. graph rows of the main documents: translate ids, drop deleted ones,
    // keep the padding at the end of the row
    let mut internal_graph = vec![u32::MAX; n_live * MAX_GRAPH_DEGREE];
    for ext in 0..n_main {
        let Some(internal_id) = ext_to_internal[ext] else {
            continue;
        };
        let row = &mut internal_graph[internal_id as usize * MAX_GRAPH_DEGREE
            ..(internal_id as usize + 1) * MAX_GRAPH_DEGREE];
        let mut written = 0;
        for x in &graph[ext * MAX_GRAPH_DEGREE..(ext + 1) * MAX_GRAPH_DEGREE] {
            if *x == u32::MAX {
                continue;
            }
            if let Some(Some(internal_x)) = ext_to_internal.get(*x as usize) {
                row[written] = *internal_x;
                written += 1;
            }
        }
    }

    // 4. graph rows of the delta documents: insert them one by one.
    // The search starts from the ext seeds, translated to the internal ids, and
    // only reaches nodes that are already in the graph.
    let entry_seeds: SeedVecs = index_output
        .seeds
        .iter()
        .map(|list| {
            list.iter()
                .filter_map(|(w, id)| {
                    ext_to_internal
                        .get(*id as usize)
                        .copied()
                        .flatten()
                        .map(|internal_id| (*w, internal_id))
                })
                .collect()
        })
        .collect();

    let packed = pack_sketches(&internal_sketches, vectors_and_sketches.scale);
    let no_tombstones = Tombstones::with_capacity(0);
    let borrowed: Vec<&SparseVector> = internal_vectors.iter().collect();
    let mut visited = Visited {
        counter: 0,
        visited: vec![0; n_live],
    };
    let mut dense = vec![0_u8; vocab_size];
    let mut dense_prune = vec![0_f32; vocab_size];

    for ext in n_main..n {
        let Some(internal_id) = ext_to_internal[ext] else {
            continue;
        };
        if internal_id == 0 {
            // no main document survived and this is the first delta one:
            // the graph is empty, there is nobody to connect to
            continue;
        }
        let me = internal_id as usize;

        let q8 = &internal_sketches[me];
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
            &internal_graph,
            ef,
            patience,
            &no_tombstones,
        )?;

        // exact scores, like the refine step of the build
        let pool: Vec<(u32, f32)> = found
            .iter()
            .map(|f| f.1)
            .filter(|id| *id != internal_id)
            .map(|id| (id, internal_vectors[me].dot(&internal_vectors[id as usize])))
            .collect();

        let row = robust_prune::<f32>(pool, prune_alpha, &borrowed, &mut dense_prune);
        internal_graph[me * MAX_GRAPH_DEGREE..(me + 1) * MAX_GRAPH_DEGREE].copy_from_slice(&row);

        // reverse edges: let the neighbours point back to the internal node
        for nb in row.iter() {
            if *nb == u32::MAX {
                break;
            }
            let nb_row = &mut internal_graph
                [*nb as usize * MAX_GRAPH_DEGREE..(*nb as usize + 1) * MAX_GRAPH_DEGREE];
            if let Some(pos) = nb_row.iter().position(|x| *x == u32::MAX) {
                nb_row[pos] = internal_id;
            } else {
                // the row is full: prune its neighbours plus the internal node,
                // scored against the neighbour that owns the row
                let owner = &internal_vectors[*nb as usize];
                let mut candidates: Vec<(u32, f32)> = nb_row
                    .iter()
                    .map(|c| (*c, owner.dot(&internal_vectors[*c as usize])))
                    .collect();
                candidates.push((internal_id, owner.dot(&internal_vectors[me])));
                let kept =
                    robust_prune::<f32>(candidates, prune_alpha, &borrowed, &mut dense_prune);
                nb_row.copy_from_slice(&kept);
            }
        }
    }

    // 5. fresh seeds and term maxima, from the documents that are left
    let (seeds, term_max) = build_seed_list(&borrowed, entrypoint_seeds, vocab_size);
    drop(borrowed);

    // 6. record ids internal -> ext
    let mut ids = vec![0_u32; n_live];
    for ext in 0..n {
        if let Some(internal_id) = ext_to_internal[ext] {
            ids[internal_id as usize] = if ext < n_main {
                old_ids[ext]
            } else {
                ext_base + (ext - n_main) as u32
            };
        }
    }
    let new_ext_base = ext_base + delta.docs.len() as u32;
    let mut full_ids_bytes = new_ext_base.to_le_bytes().to_vec();
    full_ids_bytes.extend(ids.iter().flat_map(|i| i.to_le_bytes()));

    // 7. write everything. There are no delete.log and delta.log: a missing
    // log means "empty".
    fs::create_dir_all(out_dir).map_err(|e| e.to_string())?;
    save_vectors_and_sketches(
        internal_vectors,
        internal_sketches,
        vectors_and_sketches.scale,
        out_dir.join(COMPACT_VECTORS_FILE),
        out_dir.join(COMPACT_SKETCHES_FILE),
    )
    .map_err(|e| e.to_string())?;
    save_index(
        internal_graph,
        seeds,
        term_max,
        out_dir.join(COMPACT_INDEX_FILE),
    )
    .map_err(|e| e.to_string())?;
    fs::write(out_dir.join(COMPACT_IDS_FILE), full_ids_bytes).map_err(|e| e.to_string())?;

    Ok((n_live, new_ext_base))
}
