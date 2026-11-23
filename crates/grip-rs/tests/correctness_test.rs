use std::{
    collections::HashMap,
    fs::{self, File, OpenOptions},
    io::{BufReader, Read, Write},
    path::{Path, PathBuf},
    sync::RwLock,
};

use grip_rs::{
    BuildParams, IndexOutput, SearchParams, SparseVector, VectorsAndSketches, Visited, build_index,
    delta::{DELTA_LOG_FILE, Delta, MAX_DELTA, load_delta, persist_delta_insert},
    disk::decode_sparse_vectors_raw,
    read_index, read_sketches,
    robustprune::MAX_GRAPH_DEGREE,
    save_index, save_vectors_and_sketches, search, search_all,
    tombstone::Tombstones,
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
    build_temp_index_from(vectors, vocab_size)
}

fn build_temp_index_from(vectors: Vec<SparseVector>, vocab_size: usize) -> TempDir {
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

// ---------- inserts (delta index) ----------

/// Documents `0..N_MAIN` of the file go in the graph, the rest are inserted afterwards.
const N_MAIN: usize = 1500;

struct InsertEnv {
    dir: TempDir,
    index: IndexOutput,
    sketches: VectorsAndSketches,
    mmap: Mmap,
    queries: Vec<SparseVector>,
    main: Vec<SparseVector>,
    extra: Vec<SparseVector>,
}

fn insert_env() -> InsertEnv {
    let data = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data");
    let mut all =
        read_seismic_bin(&data.join("documents.bin"), Some(MAX_DOCS)).expect("Should load vectors");
    assert!(all.len() > N_MAIN, "the documents file is too small");
    let extra = all.split_off(N_MAIN);
    let main = all;

    let dir = build_temp_index_from(main.clone(), VOCAB_SIZE);
    let index = read_index(dir.path().join("index.bin"), VOCAB_SIZE).expect("Should read index");
    let sketches = read_sketches(dir.path().join("sketches.bin")).expect("Should read sketches");
    let vectors_file = File::open(dir.path().join("vectors.bin")).expect("Should open vectors");
    // Safety: the vectors file must not change while it is mapped
    let mmap = unsafe { Mmap::map(&vectors_file).expect("Should mmap") };
    let queries = read_seismic_bin(&data.join("queries.bin"), Some(MAX_QUERIES))
        .expect("Should read queries");
    assert!(!queries.is_empty());

    InsertEnv {
        dir,
        index,
        sketches,
        mmap,
        queries,
        main,
        extra,
    }
}

impl InsertEnv {
    fn n_main(&self) -> u32 {
        self.main.len() as u32
    }

    /// Inserts every extra document, and checks that the ids match the oracle's.
    fn insert_all(&self, delta: &RwLock<Delta>, oracle: &mut Oracle) {
        for v in &self.extra {
            let id = persist_delta_insert(
                v.indices.clone(),
                v.values.clone(),
                delta,
                self.dir.path(),
                self.n_main(),
            )
            .expect("Should insert");
            assert_eq!(id, oracle.insert(v.clone()));
        }
    }

    fn search_all(&self, tombstones: &Tombstones, delta: &Delta) -> Vec<Vec<(u32, f32)>> {
        let mut visited = Visited {
            counter: 0,
            visited: vec![0; self.main.len()],
        };
        let mut dense = vec![0_u8; VOCAB_SIZE];
        let mut dense_rescoring = vec![0_f32; VOCAB_SIZE];
        let mut delta_acc = vec![0_f32; MAX_DELTA];
        let params = SearchParams::default();
        self.queries
            .iter()
            .map(|q| {
                search_all(
                    q,
                    &self.sketches,
                    &self.index,
                    &params,
                    &mut visited,
                    &mut dense,
                    &mut dense_rescoring,
                    &self.mmap,
                    tombstones,
                    self.n_main(),
                    delta,
                    &mut delta_acc,
                )
                .expect("Should search")
            })
            .collect()
    }

    /// Checks the results against the oracle and returns the mean recall.
    ///
    /// - every returned score equals the oracle's score for that id
    /// - every document of the delta that is clearly in the exact top k is returned
    ///   (the delta search is exact, so it must not miss one)
    fn check(&self, results: &[Vec<(u32, f32)>], oracle: &Oracle) -> f32 {
        let k = SearchParams::default().top_k;
        let (mut total, mut counted) = (0_f32, 0_usize);
        for (q, res) in self.queries.iter().zip(results) {
            let exact = oracle.search(q, k);
            let all: HashMap<u32, f32> = oracle.search(q, usize::MAX).into_iter().collect();

            for (id, score) in res {
                if let Some(expected) = all.get(id) {
                    assert!(
                        (score - expected).abs() <= 1e-3 * expected.abs().max(1.0),
                        "document {id}: got a score of {score}, expected {expected}"
                    );
                }
            }
            assert!(res.windows(2).all(|w| w[0].1 >= w[1].1), "not sorted");

            if let Some((_, kth)) = exact.get(k - 1) {
                for (id, score) in &exact {
                    if *id >= self.n_main() && *score > kth + 1e-3 * kth.abs().max(1.0) {
                        assert!(
                            res.iter().any(|r| r.0 == *id),
                            "delta document {id} (score {score}) is in the exact top {k} but was not returned"
                        );
                    }
                }
            }

            if exact.is_empty() {
                continue;
            }
            counted += 1;
            total += recall_at_k(
                &res.iter().map(|v| v.0).collect::<Vec<u32>>(),
                &exact.iter().map(|v| v.0).collect::<Vec<u32>>(),
            );
        }
        total / counted as f32
    }
}

/// Inserted documents are searchable, with the right ids and exact scores.
#[test]
fn test_delta_inserts() {
    let env = insert_env();
    let mut oracle = Oracle::new(env.main.clone(), VOCAB_SIZE);
    let delta = RwLock::new(Delta::default());
    env.insert_all(&delta, &mut oracle);

    let tombstones = Tombstones::with_capacity(N_MAIN + MAX_DELTA);
    let results = env.search_all(&tombstones, &delta.read().unwrap());

    let total = N_MAIN + env.extra.len();
    assert!(
        results
            .iter()
            .flatten()
            .all(|(id, _)| (*id as usize) < total)
    );
    assert!(
        results.iter().flatten().any(|(id, _)| *id >= env.n_main()),
        "no document of the delta was ever returned"
    );

    let recall = env.check(&results, &oracle);
    assert!(recall > 0.5, "recall is {recall}");
    assert!(recall <= 1.0);
}

/// Deletions work on both the graph and the delta, and the delta ids are accepted.
#[test]
fn test_delta_deletions() {
    let env = insert_env();
    let mut oracle = Oracle::new(env.main.clone(), VOCAB_SIZE);
    let delta = RwLock::new(Delta::default());
    env.insert_all(&delta, &mut oracle);

    let tombstones = Tombstones::with_capacity(N_MAIN + MAX_DELTA);
    let total = N_MAIN + env.extra.len();
    let deleted: Vec<u32> = (0..total as u32).filter(|id| id % 10 == 3).collect();
    assert!(deleted.iter().any(|id| *id >= env.n_main()));
    for id in &deleted {
        assert!(tombstones.delete(*id));
        assert!(oracle.delete(*id));
    }

    let results = env.search_all(&tombstones, &delta.read().unwrap());
    assert!(
        !results.iter().flatten().any(|(id, _)| deleted.contains(id)),
        "a deleted document was returned"
    );

    let recall = env.check(&results, &oracle);
    assert!(recall > 0.5, "recall is {recall}");
}

/// A delta that is loaded again from its log gives the same results.
#[test]
fn test_delta_survives_restart() {
    let env = insert_env();
    let mut oracle = Oracle::new(env.main.clone(), VOCAB_SIZE);
    let delta = RwLock::new(Delta::default());
    env.insert_all(&delta, &mut oracle);

    let tombstones = Tombstones::with_capacity(N_MAIN + MAX_DELTA);
    let before = env.search_all(&tombstones, &delta.read().unwrap());
    drop(delta);

    let reloaded = load_delta(env.dir.path(), env.n_main()).expect("Should load the delta");
    assert_eq!(reloaded.docs.len(), env.extra.len());
    let expected_ids: Vec<u32> = (0..env.extra.len() as u32)
        .map(|i| env.n_main() + i)
        .collect();
    assert_eq!(reloaded.ext_ids, expected_ids);

    let after = env.search_all(&tombstones, &reloaded);
    assert_eq!(before, after);
}

fn some_vector(seed: u32) -> (Vec<u32>, Vec<f32>) {
    (vec![seed, seed + 7, seed + 100], vec![0.5, 1.0, 2.0])
}

/// A record cut short by a crash is dropped, and the next insert does not corrupt the log.
#[test]
fn test_delta_torn_log() {
    let dir = tempfile::tempdir().expect("Should create a temporary directory");
    let base = 100_u32;
    let delta = RwLock::new(Delta::default());
    for i in 0..3 {
        let (indices, values) = some_vector(i);
        let id = persist_delta_insert(indices, values, &delta, dir.path(), base).unwrap();
        assert_eq!(id, base + i);
    }
    let log = dir.path().join(DELTA_LOG_FILE);
    let good_len = fs::metadata(&log).unwrap().len();
    // the first 6 bytes of a record: nnz = 3, and half of the first index
    let mut f = OpenOptions::new().append(true).open(&log).unwrap();
    f.write_all(&[3, 0, 0, 0, 1, 0]).unwrap();
    drop(f);

    let loaded = load_delta(dir.path(), base).expect("Should load despite the torn record");
    assert_eq!(loaded.docs.len(), 3);
    assert_eq!(fs::metadata(&log).unwrap().len(), good_len, "not truncated");

    // inserting after the repair gives a log that loads cleanly
    let delta = RwLock::new(loaded);
    let (indices, values) = some_vector(3);
    assert_eq!(
        persist_delta_insert(indices, values, &delta, dir.path(), base).unwrap(),
        base + 3
    );
    let again = load_delta(dir.path(), base).unwrap();
    assert_eq!(again.docs.len(), 4);
    assert_eq!(again.ext_ids, vec![100, 101, 102, 103]);
}

/// A log that does not match the index it is loaded for is rejected.
#[test]
fn test_delta_wrong_base_is_rejected() {
    let dir = tempfile::tempdir().expect("Should create a temporary directory");
    let delta = RwLock::new(Delta::default());
    let (indices, values) = some_vector(1);
    persist_delta_insert(indices, values, &delta, dir.path(), 100).unwrap();
    assert!(load_delta(dir.path(), 101).is_err());
    assert!(load_delta(dir.path(), 100).is_ok());
}

/// No log means an empty delta.
#[test]
fn test_delta_missing_log_is_empty() {
    let dir = tempfile::tempdir().expect("Should create a temporary directory");
    let delta = load_delta(dir.path(), 10).expect("A missing log is not an error");
    assert!(delta.docs.is_empty());
}
