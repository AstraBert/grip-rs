//! Timing of the index build phases.
//!
//! With the `performance-timing` feature, [`Phases`] prints the wall time of each
//! phase and the peak memory so far to stderr, and [`CpuClock`] adds up the time
//! that all threads spend in one piece of code. Without the feature, both types
//! do nothing and the compiler removes the calls.

#[cfg(feature = "performance-timing")]
mod enabled {
    use std::{
        sync::atomic::{AtomicU64, Ordering},
        time::Instant,
    };

    /// Prints the time since the previous mark each time `mark` is called.
    pub struct Phases {
        label: &'static str,
        start: Instant,
        last: Instant,
    }

    impl Phases {
        pub fn new(label: &'static str) -> Self {
            let now = Instant::now();
            Self {
                label,
                start: now,
                last: now,
            }
        }

        pub fn mark(&mut self, phase: &str) {
            let now = Instant::now();
            eprintln!(
                "[timing] {}: {phase}: {:.2?} (total {:.2?}, peak memory {})",
                self.label,
                now - self.last,
                now - self.start,
                peak_memory()
            );
            // the printing time must not count for the next phase
            self.last = Instant::now();
        }

        /// Prints an extra line. `text` only runs when the feature is on.
        pub fn note(&self, text: impl FnOnce() -> String) {
            eprintln!("[timing] {}:     {}", self.label, text());
        }
    }

    /// Adds up the time that all threads spend inside `time`.
    /// Call it once for each chunk of work, not once for each tiny operation,
    /// because reading the clock has a cost.
    #[derive(Default)]
    pub struct CpuClock(AtomicU64);

    impl CpuClock {
        pub fn time<T>(&self, f: impl FnOnce() -> T) -> T {
            let t = Instant::now();
            let out = f();
            self.0
                .fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
            out
        }

        pub fn seconds(&self) -> f64 {
            self.0.load(Ordering::Relaxed) as f64 / 1e9
        }
    }

    #[cfg(unix)]
    fn peak_memory() -> String {
        // SAFETY: `usage` is a plain C struct, and zero is a valid value for all its fields
        let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
        // SAFETY: `usage` is valid for writes
        if unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) } != 0 {
            return "unknown".to_string();
        }
        let raw = usage.ru_maxrss as f64;
        // macOS reports bytes, Linux reports KiB
        let bytes = if cfg!(target_os = "macos") {
            raw
        } else {
            raw * 1024.0
        };
        format!("{:.2} GiB", bytes / (1u64 << 30) as f64)
    }

    #[cfg(not(unix))]
    fn peak_memory() -> String {
        "unknown".to_string()
    }
}

#[cfg(not(feature = "performance-timing"))]
mod disabled {
    pub struct Phases;

    impl Phases {
        pub fn new(_label: &'static str) -> Self {
            Self
        }
        pub fn mark(&mut self, _phase: &str) {}
        pub fn note(&self, _text: impl FnOnce() -> String) {}
    }

    #[derive(Default)]
    pub struct CpuClock;

    impl CpuClock {
        pub fn time<T>(&self, f: impl FnOnce() -> T) -> T {
            f()
        }
        pub fn seconds(&self) -> f64 {
            0.0
        }
    }
}

#[cfg(feature = "performance-timing")]
pub use enabled::*;

#[cfg(not(feature = "performance-timing"))]
pub use disabled::*;
