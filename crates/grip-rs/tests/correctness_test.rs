use std::{
    fs::File,
    io::{BufReader, Read},
    path::{Path, PathBuf},
};

use grip_rs::{
    BuildParams, SearchParams, SparseVector, Visited, build_index, disk::decode_sparse_vectors_raw,
    read_index, read_sketches, robustprune::MAX_GRAPH_DEGREE, save_index,
    save_vectors_and_sketches, search, tombstone::Tombstones,
};
use memmap2::Mmap;
use rand::{SeedableRng, rngs::StdRng, seq::IndexedRandom};
use tempfile::TempDir;

const MAX_DOCS: usize = 2000;
const MAX_QUERIES: usize = 200;
const VOCAB_SIZE: usize = 31000;

fn recall_at_k(results: &[u32], exact: &[u32]) -> f32 {
    let found: usize = results
        .iter()
        .map(|p| if exact.contains(p) { 1 } else { 0 })
        .collect::<Vec<usize>>()
        .iter()
        .sum();
    found as f32 / exact.len() as f32
}

struct Oracle {
    docs: Vec<Option<SparseVector>>,
    vocab_size: usize,
}

#[allow(dead_code)]
impl Oracle {
    fn new(docs: Vec<SparseVector>, vocab_size: usize) -> Self {
        Self {
            docs: docs.into_iter().map(Some).collect(),
            vocab_size,
        }
    }

    fn insert(&mut self, v: SparseVector) -> u32 {
        self.docs.push(Some(v));
        (self.docs.len() - 1) as u32
    }

    fn delete(&mut self, id: u32) -> bool {
        self.docs[id as usize].take().is_some()
    }

    fn search(&self, query: &SparseVector, k: usize) -> Vec<(u32, f32)> {
        let mut dense = vec![0f32; self.vocab_size];

        for (t, w) in query.indices.iter().zip(&query.values) {
            dense[*t as usize] = *w;
        }

        let mut scored: Vec<(u32, f32)> = self
            .docs
            .iter()
            .enumerate()
            .filter_map(|(id, d)| {
                let v = d.as_ref()?;
                let s: f32 = v
                    .indices
                    .iter()
                    .zip(&v.values)
                    .map(|(t, w)| dense[*t as usize] * w)
                    .sum();
                (s > 0.0).then_some((id as u32, s))
            })
            .collect();
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        scored.truncate(k);
        scored
    }
}

fn read_u32(reader: &mut impl Read) -> Result<u32, String> {
    let mut b = [0_u8; 4];
    reader
        .read_exact(&mut b)
        .map_err(|e| format!("cannot read the file: {e}"))?;
    Ok(u32::from_le_bytes(b))
}

fn read_seismic_bin(path: &Path, limit: Option<usize>) -> Result<Vec<SparseVector>, String> {
    let file = File::open(path).map_err(|e| format!("cannot open {}: {e}", path.display()))?;
    let mut reader = BufReader::with_capacity(1 << 20, file);
    let n = read_u32(&mut reader)? as usize;
    let n = limit.map_or(n, |l| l.min(n));
    let mut vectors = Vec::with_capacity(n);
    // ids and values are stored one after the other, so one read gets both
    let mut buf: Vec<u8> = Vec::new();
    for _ in 0..n {
        let len = read_u32(&mut reader)? as usize;
        buf.resize(len * 8, 0);
        reader
            .read_exact(&mut buf)
            .map_err(|e| format!("cannot read the file: {e}"))?;
        let indices: Vec<u32> = buf[..len * 4]
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let values: Vec<f32> = buf[len * 4..]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        vectors.push(SparseVector::new(indices, values)?);
    }
    Ok(vectors)
}

fn build_temp_index(documents: PathBuf, vocab_size: usize) -> TempDir {
    let vectors = read_seismic_bin(&documents, Some(MAX_DOCS)).expect("Should load vectors");
    let output =
        build_index(&vectors, BuildParams::from_defaults(vocab_size)).expect("Should build index");
    let tempdir = tempfile::tempdir_in(".").expect("Should be able to create temporary directory");
    save_index(
        output.0,
        output.1,
        output.2,
        tempdir.path().join("index.bin"),
    )
    .expect("Should save the index");
    save_vectors_and_sketches(
        vectors,
        output.3,
        output.4,
        tempdir.path().join("vectors.bin"),
        tempdir.path().join("sketches.bin"),
    )
    .expect("Should save vectors and sketches");
    tempdir
}

/// tests recall on an index with no modifications whatsoever
#[test]
fn test_unmodified_index_recall() {
    let temp_dir = build_temp_index(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("data")
            .join("documents.bin"),
        VOCAB_SIZE,
    );
    let index =
        read_index(temp_dir.path().join("index.bin"), VOCAB_SIZE).expect("Should read index");
    let vectors_and_sketches =
        read_sketches(temp_dir.path().join("sketches.bin")).expect("Should read sketches");

    let vectors_file =
        File::open(temp_dir.path().join("vectors.bin")).expect("Should open vectors file");
    // Safety: the vectors file must not change while it is mapped
    let mmap = unsafe { Mmap::map(&vectors_file).expect("Should mmap") };

    let queries = read_seismic_bin(
        &PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("data")
            .join("queries.bin"),
        Some(MAX_QUERIES),
    )
    .expect("Should read queries");
    assert!(!queries.is_empty());

    let n_docs = index.graph.len() / MAX_GRAPH_DEGREE;
    let mut visited = Visited {
        counter: 0,
        visited: vec![0; n_docs],
    };

    let mut results = Vec::with_capacity(queries.len());
    let mut dense = vec![0; VOCAB_SIZE];
    let mut dense_rescoring = vec![0_f32; VOCAB_SIZE];
    let params = SearchParams::default();
    for q in &queries {
        results.push(
            search(
                q,
                &vectors_and_sketches,
                &index,
                &params,
                &mut visited,
                &mut dense,
                &mut dense_rescoring,
                &mmap,
                &Tombstones::with_capacity(0),
            )
            .expect("Should search"),
        );
    }

    let raw = decode_sparse_vectors_raw(&mmap, None).expect("Should decode vectors");
    let docs = raw
        .iter()
        .map(|v| {
            let indices = v.iter().map(|i| i.0).collect::<Vec<u32>>();
            let values = v.iter().map(|i| i.1).collect::<Vec<f32>>();
            SparseVector::new(indices, values)
        })
        .collect::<Result<Vec<SparseVector>, String>>()
        .expect("Should load all vectors");

    let oracle = Oracle::new(docs, VOCAB_SIZE);
    let mut exacts = Vec::with_capacity(queries.len());

    for query in &queries {
        let exact = oracle.search(query, params.top_k);
        exacts.push(exact);
    }

    let mut total = 0_f32;
    for (res, ext) in results.iter().zip(exacts.iter()) {
        let recall = recall_at_k(
            &res.iter().map(|v| v.0).collect::<Vec<u32>>(),
            &ext.iter().map(|v| v.0).collect::<Vec<u32>>(),
        );
        total += recall;
    }

    assert!((total / queries.len() as f32) > 0.5);
    assert!((total / queries.len() as f32) <= 1.0);
}

/// tests recall on an index with no modifications whatsoever
#[test]
fn test_tombstone_deletions() {
    let temp_dir = build_temp_index(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("data")
            .join("documents.bin"),
        VOCAB_SIZE,
    );
    let index =
        read_index(temp_dir.path().join("index.bin"), VOCAB_SIZE).expect("Should read index");
    let vectors_and_sketches =
        read_sketches(temp_dir.path().join("sketches.bin")).expect("Should read sketches");

    let vectors_file =
        File::open(temp_dir.path().join("vectors.bin")).expect("Should open vectors file");
    // Safety: the vectors file must not change while it is mapped
    let mmap = unsafe { Mmap::map(&vectors_file).expect("Should mmap") };

    let queries = read_seismic_bin(
        &PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("data")
            .join("queries.bin"),
        Some(MAX_QUERIES),
    )
    .expect("Should read queries");
    assert!(!queries.is_empty());

    for i in &[0.05, 0.1, 0.2, 0.5] {
        let n_docs = index.graph.len() / MAX_GRAPH_DEGREE;
        let tombstones = Tombstones::with_capacity(n_docs);

        let doc_ids = (0..n_docs)
            .map(|v| u32::try_from(v).expect("Should convert to u32"))
            .collect::<Vec<u32>>();
        let mut rng = StdRng::seed_from_u64(42);
        let to_delete: Vec<u32> = doc_ids
            .choose_iter(&mut rng)
            .unwrap()
            .take((*i * n_docs as f64).floor() as usize)
            .map(|t| *t)
            .collect();

        let raw = decode_sparse_vectors_raw(&mmap, None).expect("Should decode vectors");
        let docs = raw
            .iter()
            .map(|v| {
                let indices = v.iter().map(|i| i.0).collect::<Vec<u32>>();
                let values = v.iter().map(|i| i.1).collect::<Vec<f32>>();
                SparseVector::new(indices, values)
            })
            .collect::<Result<Vec<SparseVector>, String>>()
            .expect("Should load all vectors");

        let mut oracle = Oracle::new(docs, VOCAB_SIZE);

        for d in &to_delete {
            tombstones.delete(*d);
            oracle.delete(*d);
        }

        let mut visited = Visited {
            counter: 0,
            visited: vec![0; n_docs],
        };

        let mut results = Vec::with_capacity(queries.len());
        let mut dense = vec![0; VOCAB_SIZE];
        let mut dense_rescoring = vec![0_f32; VOCAB_SIZE];
        let params = SearchParams::default();

        for q in &queries {
            let res = search(
                q,
                &vectors_and_sketches,
                &index,
                &params,
                &mut visited,
                &mut dense,
                &mut dense_rescoring,
                &mmap,
                &tombstones,
            )
            .expect("Should search");
            assert!(!res.iter().any(|v| to_delete.contains(&v.0)));
            results.push(res);
        }

        let mut exacts = Vec::with_capacity(queries.len());

        for query in &queries {
            let exact = oracle.search(query, params.top_k);
            exacts.push(exact);
        }

        let mut total = 0_f32;
        for (res, ext) in results.iter().zip(exacts.iter()) {
            let recall = recall_at_k(
                &res.iter().map(|v| v.0).collect::<Vec<u32>>(),
                &ext.iter().map(|v| v.0).collect::<Vec<u32>>(),
            );
            total += recall;
        }

        assert!((total / queries.len() as f32) > 0.5);
        assert!((total / queries.len() as f32) <= 1.0);
    }
}
