//! Side-channel counter for progress reporting. It never takes part in the
//! computation it accompanies; only observers reading `snapshot` use it.

use std::sync::atomic::{AtomicU64, Ordering};

/// Units of work done; the total is `None` until known. A reader that learns
/// its size only after opening the file sets it with `set_total`.
pub struct Progress {
    done: AtomicU64,
    total: AtomicU64,
}

const UNKNOWN: u64 = u64::MAX;

impl Progress {
    pub fn new(total: Option<u64>) -> Self {
        Self {
            done: AtomicU64::new(0),
            total: AtomicU64::new(total.unwrap_or(UNKNOWN)),
        }
    }

    pub fn set_total(&self, total: u64) {
        self.total.store(total, Ordering::Relaxed);
    }

    pub fn inc(&self) {
        self.inc_by(1);
    }

    pub fn inc_by(&self, n: u64) {
        self.done.fetch_add(n, Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> (u64, Option<u64>) {
        let total = self.total.load(Ordering::Relaxed);
        (
            self.done.load(Ordering::Relaxed),
            (total != UNKNOWN).then_some(total),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn inc_counts_every_call_under_concurrency() {
        let progress = Arc::new(Progress::new(Some(100)));
        let handles: Vec<_> = (0..10)
            .map(|_| {
                let p = Arc::clone(&progress);
                thread::spawn(move || (0..10).for_each(|_| p.inc()))
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(progress.snapshot(), (100, Some(100)));
    }

    #[test]
    fn inc_by_adds_several_units() {
        let progress = Progress::new(None);
        progress.inc_by(7);
        progress.inc_by(3);
        assert_eq!(progress.snapshot(), (10, None));
        progress.set_total(12);
        assert_eq!(progress.snapshot(), (10, Some(12)));
    }
}
