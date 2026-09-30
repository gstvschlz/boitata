//! Shared driver for the `progress` keyword: a `tqdm` bar updated from a
//! monitor thread while the computation runs with the GIL released.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

use boitata_core::Progress;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Runs `f` with a counter of `total` units. When `total` is `None` the bar
/// opens on the first tick, by which time a reader has usually set it.
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
    let tqdm = py.import("tqdm.auto")?.getattr("tqdm")?.unbind();
    let open = |py: Python, total: Option<u64>| -> PyResult<Py<PyAny>> {
        let kwargs = PyDict::new(py);
        if let Some(total) = total {
            kwargs.set_item("total", total)?;
        }
        Ok(tqdm.bind(py).call((), Some(&kwargs))?.unbind())
    };
    let bar = OnceLock::new();
    if total.is_some() {
        let _ = bar.set(open(py, total)?);
    }
    let counter = Progress::new(total);
    let stop = AtomicBool::new(false);
    let update = |delta: u64, total: Option<u64>, found: bool| {
        Python::attach(|py| {
            let result = (|| -> PyResult<()> {
                let bar = match bar.get() {
                    Some(bar) if found => {
                        bar.bind(py).call_method1("reset", (total,))?;
                        bar
                    }
                    Some(bar) => bar,
                    None => {
                        let opened = open(py, total)?;
                        bar.get_or_init(|| opened)
                    }
                };
                bar.bind(py).call_method1("update", (delta,))?;
                Ok(())
            })();
            if let Err(e) = result {
                e.print(py);
            }
        })
    };
    let result = py.detach(|| {
        thread::scope(|scope| {
            let monitor = scope.spawn(|| {
                let (mut last, mut known) = (0, total);
                let mut tick = || {
                    let (done, total) = counter.snapshot();
                    let found = total.is_some() && known.is_none();
                    if found {
                        (last, known) = (0, total);
                    }
                    if done > last || found {
                        update(done - last, total, found);
                        last = done;
                    }
                };
                while !stop.load(Ordering::Acquire) {
                    thread::park_timeout(Duration::from_millis(50));
                    tick();
                }
                tick();
            });
            let result = f(Some(&counter));
            let (done, total) = counter.snapshot();
            if let Some(total) = total {
                counter.inc_by(total.saturating_sub(done));
            }
            stop.store(true, Ordering::Release);
            monitor.thread().unpark();
            let _ = monitor.join();
            result
        })
    });
    if let Some(bar) = bar.get() {
        Python::attach(|py| {
            if let Err(e) = bar.bind(py).call_method0("close") {
                e.print(py);
            }
        });
    }
    Ok(result)
}
