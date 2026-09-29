//! Side-channel counter for progress reporting. It never takes part in the
//! computation it accompanies; only observers reading `snapshot` use it.

use std::sync::atomic::{AtomicU64, Ordering};

/// Units of work done; `total` is `None` when unknown upfront.
pub struct Progress {
    done: AtomicU64,
    total: Option<u64>,
}

impl Progress {
    pub fn new(total: Option<u64>) -> Self {
        Self {
            done: AtomicU64::new(0),
            total,
        }
    }

    pub fn inc(&self) {
        self.inc_by(1);
    }

    pub fn inc_by(&self, n: u64) {
        self.done.fetch_add(n, Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> (u64, Option<u64>) {
        (self.done.load(Ordering::Relaxed), self.total)
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
    }
}
