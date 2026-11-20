use std::{
    fs::{self, OpenOptions},
    io::{self, Write},
    path::Path,
    sync::atomic::{AtomicU64, AtomicUsize, Ordering},
};

pub const DELETION_LOG_FILE: &'static str = "delete.log";

#[derive(Debug)]
pub struct Tombstones {
    words: Vec<AtomicU64>,
    count: AtomicUsize,
}

impl Tombstones {
    pub fn with_capacity(n: usize) -> Self {
        let n_words = n.div_ceil(64);
        Self {
            words: (0..n_words).map(|_| AtomicU64::new(0)).collect(),
            count: AtomicUsize::new(0),
        }
    }

    pub fn is_deleted(&self, id: u32) -> bool {
        let word = self.words.get(id as usize >> 6);
        if word.is_none() {
            return false;
        }
        word.unwrap().load(Ordering::Relaxed) >> (id & 63) & 1 == 1
    }

    /// Returns true if the document was alive before this call.
    pub fn delete(&self, id: u32) -> bool {
        let mask = 1u64 << (id & 63);
        let prev = self.words[id as usize >> 6].fetch_or(mask, Ordering::Relaxed);
        let newly = prev & mask == 0;
        if newly {
            self.count.fetch_add(1, Ordering::Relaxed);
        }
        newly
    }

    pub fn ratio(&self, existing: usize) -> f64 {
        self.count.load(Ordering::Relaxed) as f64 / existing as f64
    }
}

pub fn delete(
    id: u32,
    tombstones: &Tombstones,
    log_dir: &Path,
    documents_count: usize,
) -> io::Result<bool> {
    if id as usize >= documents_count {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            "Point ID not among the documents",
        ));
    }

    let deleted = tombstones.delete(id);
    if !deleted {
        return Ok(false);
    }

    let mut fl = OpenOptions::new()
        .append(true)
        .create(true)
        .open(log_dir.join(DELETION_LOG_FILE))?;

    fl.write_all(&id.to_le_bytes())?;
    fl.flush()?;

    Ok(true)
}

pub fn load_tombstones(tombstones: &Tombstones, log_dir: &Path) -> io::Result<()> {
    let content = match fs::read(log_dir.join(DELETION_LOG_FILE)) {
        Ok(c) => c,
        Err(e) => match e.kind() {
            io::ErrorKind::NotFound => return Ok(()),
            _ => return Err(e),
        },
    };

    for chunk in content.chunks_exact(4) {
        let id = u32::from_le_bytes(chunk.try_into().expect("Should convert to [u8; 4]"));
        tombstones.delete(id);
    }

    Ok(())
}
