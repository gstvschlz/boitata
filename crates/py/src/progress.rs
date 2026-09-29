//! Shared driver for the `progress` keyword: a `tqdm` bar updated from a
//! monitor thread while the computation runs with the GIL released.

use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

use ceres_core::Progress;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Runs `f` with a counter of `total` units (`None` when unknown upfront).
/// Without `show`, `f` runs with no counter and no extra thread.
pub fn with_progress<T: Send>(
    py: Python<'_>,
    total: Option<u64>,
    show: bool,
    f: impl FnOnce(Option<&Progress>) -> T + Send,
) -> PyResult<T> {
    if !show {
        return Ok(py.detach(|| f(None)));
    }
    let kwargs = PyDict::new(py);
    if let Some(total) = total {
        kwargs.set_item("total", total)?;
    }
    let bar = py
        .import("tqdm")?
        .getattr("tqdm")?
        .call((), Some(&kwargs))?
        .unbind();
    let counter = Progress::new(total);
    let stop = AtomicBool::new(false);
    let update = |delta: u64| {
        Python::attach(|py| {
            if let Err(e) = bar.bind(py).call_method1("update", (delta,)) {
                e.print(py);
            }
        })
    };
    let result = py.detach(|| {
        thread::scope(|scope| {
            let monitor = scope.spawn(|| {
                let mut last = 0;
                while !stop.load(Ordering::Acquire) {
                    thread::park_timeout(Duration::from_millis(50));
                    let (done, _) = counter.snapshot();
                    if done > last {
                        update(done - last);
                        last = done;
                    }
                }
                let (done, _) = counter.snapshot();
                if done > last {
                    update(done - last);
                }
            });
            let result = f(Some(&counter));
            stop.store(true, Ordering::Release);
            monitor.thread().unpark();
            let _ = monitor.join();
            result
        })
    });
    Python::attach(|py| {
        if let Err(e) = bar.bind(py).call_method0("close") {
            e.print(py);
        }
    });
    Ok(result)
}
