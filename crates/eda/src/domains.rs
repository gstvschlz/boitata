//! Domains from ensemble spatial clustering, after Martin and Boisvert
//! (2018), "Towards justifying unsupervised stationary decisions for
//! geostatistical modeling: Ensemble spatial and multivariate clustering with
//! geomodeling specific clustering metrics", Computers & Geosciences 120,
//! 82–96, doi:10.1016/j.cageo.2018.08.005.
//!
//! Every column is standardized to mean 0 and variance 1 over the samples
//! where all columns are finite; the others get no domain. Neighbors are the
//! nearest samples in space (Euclidean, isotropic), the sample excluded.
//! Each clustering is k-means (Lloyd iterations) from a k-means++ start.
//!
//! - **Autocorrelation clustering** clusters every sample on, for each
//!   column, its value and the mean of its `neighbors` (the two axes of the
//!   Moran scatterplot). The paper clusters local autocorrelation statistics;
//!   the local Moran's I is the product of those two coordinates, which alone
//!   would put high-high and low-low samples together, so the coordinates are
//!   clustered instead.
//! - **Dual-space search** takes, among the `2m` nearest samples in space,
//!   the `m` nearest in the space of the columns, and clusters every sample
//!   on the mean of the columns over itself and those `m`: averaging within
//!   a geographic neighborhood, but only over samples that also look alike.
//! - **Ensembles** repeat a clustering over randomized parameters: each
//!   realization draws `m` between `neighbors / 2` and `2 × neighbors` and a
//!   random subset of at least half of the columns.
//! - **Consensus** clusters the pairing frequency `p_ij`, the share of the
//!   realizations that put samples `i` and `j` in the same cluster. The
//!   `n × n` matrix is never formed: with every realization's labels one-hot
//!   encoded, the squared distance between two samples' encodings is
//!   `2R(1 − p_ij)` for `R` realizations, so k-means on the encodings
//!   minimizes, within each domain, the mean over its pairs of `1 − p_ij`.
//!   Memory is `n × R` labels and time `n × R × k` per iteration, with no
//!   quadratic ceiling. The ensemble mixes both clusterings for
//!   [`Clustering::Consensus`] and holds only dual-space searches for
//!   [`Clustering::Dual`].
//!
//! The randomized ranges, the scatterplot coordinates and the k-means
//! consensus are this crate's reading of the method, not the paper's exact
//! recipe. Every realization has its own seed from the master seed, so the
//! domains do not depend on the number of threads.
//!
//! Two metrics score a domaining: [`wcss`], the within-cluster sum of
//! squares of the standardized columns per sample, and [`spatial_entropy`],
//! the mean over samples of the entropy of the domains among a sample and its
//! neighbors. Low values of both mean domains that are compact in the columns
//! and contiguous in space.

use boitata_core::rng::realization_seed;
use kiddo::{ImmutableKdTree, SquaredEuclidean};
use rayon::prelude::*;

use crate::{EdaError, Result, invalid};

const MAX_ITERATIONS: usize = 100;
const TRIES: usize = 10;

/// How [`domains`] clusters the samples.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Clustering {
    /// One autocorrelation clustering over `neighbors`, all columns.
    Autocorrelation,
    /// Consensus of an ensemble of dual-space searches.
    Dual,
    /// Consensus of an ensemble of autocorrelation clusterings and dual-space
    /// searches, alternating.
    Consensus,
}

/// Result of [`domains`].
#[derive(Clone, Debug, PartialEq)]
pub struct Domains {
    /// Domain of each sample, `0..k` by increasing mean of the first column;
    /// `None` where a column is NaN.
    pub labels: Vec<Option<u32>>,
    /// Mean pairing frequency of each sample with the other samples of its
    /// domain, over the realizations; `None` for one clustering.
    pub agreement: Option<Vec<Option<f64>>>,
    pub wcss: f64,
    pub spatial_entropy: f64,
}

struct Draws {
    seed: u64,
    count: u64,
}

impl Draws {
    fn new(seed: u64) -> Self {
        Self { seed, count: 0 }
    }

    fn unit(&mut self) -> f64 {
        self.count += 1;
        (realization_seed(self.seed, self.count) >> 11) as f64 / (1u64 << 53) as f64
    }

    fn below(&mut self, n: usize) -> usize {
        ((self.unit() * n as f64) as usize).min(n - 1)
    }

    /// Between `lo` and `hi`, both included.
    fn between(&mut self, lo: usize, hi: usize) -> usize {
        lo + self.below(hi - lo + 1)
    }

    /// A random subset of at least half of `0..m`, in increasing order.
    fn subset(&mut self, m: usize) -> Vec<usize> {
        let size = self.between(m.div_ceil(2), m);
        let mut all: Vec<usize> = (0..m).collect();
        for i in 0..size {
            let j = i + self.below(m - i);
            all.swap(i, j);
        }
        all.truncate(size);
        all.sort_unstable();
        all
    }
}

trait Space: Sync {
    type Center: Clone + Send + Sync;
    fn len(&self) -> usize;
    fn point(&self, i: usize) -> Self::Center;
    fn mean(&self, members: &[usize]) -> Self::Center;
    fn distance(&self, i: usize, center: &Self::Center) -> f64;
}

/// Rows of `d` features.
struct Dense {
    data: Vec<f64>,
    d: usize,
}

impl Space for Dense {
    type Center = Vec<f64>;

    fn len(&self) -> usize {
        self.data.len() / self.d
    }

    fn point(&self, i: usize) -> Vec<f64> {
        self.data[i * self.d..(i + 1) * self.d].to_vec()
    }

    fn mean(&self, members: &[usize]) -> Vec<f64> {
        let mut c = vec![0.0; self.d];
        for &i in members {
            for (c, x) in c.iter_mut().zip(&self.data[i * self.d..(i + 1) * self.d]) {
                *c += x;
            }
        }
        c.iter_mut().for_each(|c| *c /= members.len() as f64);
        c
    }

    fn distance(&self, i: usize, center: &Vec<f64>) -> f64 {
        self.data[i * self.d..(i + 1) * self.d]
            .iter()
            .zip(center)
            .map(|(x, c)| (x - c).powi(2))
            .sum()
    }
}

/// Labels of `r` realizations per sample, one-hot encoded in `k` classes.
struct OneHot {
    labels: Vec<u32>,
    r: usize,
    k: usize,
}

impl Space for OneHot {
    /// Class frequencies per realization, and their sum of squares.
    type Center = (Vec<f64>, f64);

    fn len(&self) -> usize {
        self.labels.len() / self.r
    }

    fn point(&self, i: usize) -> (Vec<f64>, f64) {
        self.mean(&[i])
    }

    fn mean(&self, members: &[usize]) -> (Vec<f64>, f64) {
        let mut f = vec![0.0; self.r * self.k];
        for &i in members {
            for (q, &l) in self.labels[i * self.r..(i + 1) * self.r].iter().enumerate() {
                f[q * self.k + l as usize] += 1.0;
            }
        }
        f.iter_mut().for_each(|f| *f /= members.len() as f64);
        let norm = f.iter().map(|f| f * f).sum();
        (f, norm)
    }

    fn distance(&self, i: usize, (f, norm): &(Vec<f64>, f64)) -> f64 {
        (self.r as f64 - 2.0 * self.hits(i, f) + norm).max(0.0)
    }
}

impl OneHot {
    /// Sum over realizations of the frequency, in `f`, of sample `i`'s class.
    fn hits(&self, i: usize, f: &[f64]) -> f64 {
        self.labels[i * self.r..(i + 1) * self.r]
            .iter()
            .enumerate()
            .map(|(q, &l)| f[q * self.k + l as usize])
            .sum()
    }
}

/// Lloyd's k-means from a k-means++ start: labels and the sum of squared
/// distances to the centers.
fn kmeans<S: Space>(space: &S, k: usize, draws: &mut Draws) -> (Vec<u32>, f64) {
    let n = space.len();
    let mut centers = vec![space.point(draws.below(n))];
    let mut nearest: Vec<f64> = (0..n)
        .into_par_iter()
        .map(|i| space.distance(i, &centers[0]))
        .collect();
    while centers.len() < k {
        let total: f64 = nearest.iter().sum();
        let pick = if total > 0.0 {
            let target = draws.unit() * total;
            let mut sum = 0.0;
            nearest
                .iter()
                .position(|d| {
                    sum += d;
                    sum > target
                })
                .unwrap_or(n - 1)
        } else {
            draws.below(n)
        };
        let center = space.point(pick);
        nearest = nearest
            .par_iter()
            .enumerate()
            .map(|(i, &d)| d.min(space.distance(i, &center)))
            .collect();
        centers.push(center);
    }
    let mut labels = vec![u32::MAX; n];
    let mut distances = vec![0.0; n];
    for _ in 0..MAX_ITERATIONS {
        let assigned: Vec<(u32, f64)> = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut best = (0, f64::INFINITY);
                for (c, center) in centers.iter().enumerate() {
                    let d = space.distance(i, center);
                    if d < best.1 {
                        best = (c as u32, d);
                    }
                }
                best
            })
            .collect();
        let changed = assigned.iter().zip(&labels).any(|(a, l)| a.0 != *l);
        (labels, distances) = assigned.into_iter().unzip();
        if !changed {
            break;
        }
        let mut members = vec![vec![]; k];
        for (i, &l) in labels.iter().enumerate() {
            members[l as usize].push(i);
        }
        for (c, m) in members.iter().enumerate() {
            if m.is_empty() {
                let far = (0..n).fold(0, |a, i| if distances[i] > distances[a] { i } else { a });
                distances[far] = 0.0;
                centers[c] = space.point(far);
            } else {
                centers[c] = space.mean(m);
            }
        }
    }
    (labels, distances.iter().sum())
}

/// The k-means of `tries` starts with the lowest sum of squares.
fn best_kmeans<S: Space>(space: &S, k: usize, tries: usize, draws: &mut Draws) -> Vec<u32> {
    let mut best = (vec![], f64::INFINITY);
    for _ in 0..tries {
        let run = kmeans(space, k, draws);
        if run.1 < best.1 {
            best = run;
        }
    }
    best.0
}

/// Indices of the `m` nearest other samples of every sample, nearest first.
fn neighbor_lists(coords: &[[f64; 3]], m: usize) -> Result<Vec<Vec<usize>>> {
    let tree = ImmutableKdTree::<f64, 3>::new_from_slice(coords)
        .map_err(|e| EdaError::InvalidInput(format!("{e:?}")))?;
    let want = std::num::NonZero::new(m + 1).expect("m + 1 > 0");
    Ok((0..coords.len())
        .into_par_iter()
        .map(|i| {
            let mut found: Vec<(f64, usize)> = tree
                .query(&coords[i])
                .nearest_n::<SquaredEuclidean<f64>>(want)
                .execute()
                .into_iter()
                .map(|r| (r.distance, r.item as usize))
                .filter(|&(_, j)| j != i)
                .collect();
            found.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            found.truncate(m);
            found.into_iter().map(|(_, j)| j).collect()
        })
        .collect())
}

/// Columns standardized to mean 0, variance 1, as rows of `m` values.
fn standardize(columns: &[Vec<f64>], rows: &[usize]) -> Vec<f64> {
    let m = columns.len();
    let mut z = vec![0.0; rows.len() * m];
    for (v, column) in columns.iter().enumerate() {
        let n = rows.len() as f64;
        let mean = rows.iter().map(|&i| column[i]).sum::<f64>() / n;
        let var = rows
            .iter()
            .map(|&i| (column[i] - mean).powi(2))
            .sum::<f64>()
            / n;
        let std = if var > 0.0 { var.sqrt() } else { 1.0 };
        for (r, &i) in rows.iter().enumerate() {
            z[r * m + v] = (column[i] - mean) / std;
        }
    }
    z
}

struct Samples<'a> {
    z: &'a [f64],
    m: usize,
    near: &'a [Vec<usize>],
}

impl Samples<'_> {
    fn autocorrelation(&self, size: usize, columns: &[usize]) -> Dense {
        let d = 2 * columns.len();
        let data = (0..self.near.len())
            .into_par_iter()
            .flat_map_iter(|i| {
                let near = &self.near[i][..size.min(self.near[i].len())];
                columns.iter().flat_map(move |&v| {
                    let lag = near.iter().map(|&j| self.z[j * self.m + v]).sum::<f64>()
                        / near.len() as f64;
                    [self.z[i * self.m + v], lag]
                })
            })
            .collect();
        Dense { data, d }
    }

    fn dual(&self, size: usize, columns: &[usize]) -> Dense {
        let distance = |i: usize, j: usize| -> f64 {
            columns
                .iter()
                .map(|&v| (self.z[i * self.m + v] - self.z[j * self.m + v]).powi(2))
                .sum()
        };
        let data = (0..self.near.len())
            .into_par_iter()
            .flat_map_iter(|i| {
                let mut near: Vec<usize> =
                    self.near[i][..(2 * size).min(self.near[i].len())].to_vec();
                near.sort_by(|&a, &b| distance(i, a).total_cmp(&distance(i, b)));
                near.truncate(size);
                near.push(i);
                columns.iter().map(move |&v| {
                    near.iter().map(|&j| self.z[j * self.m + v]).sum::<f64>() / near.len() as f64
                })
            })
            .collect();
        Dense {
            data,
            d: columns.len(),
        }
    }
}

/// Domains of samples at `coords` from the values of `columns`, in `k`
/// domains; see the module docs. `neighbors` sizes the neighborhoods and
/// `realizations` the ensemble (unused by [`Clustering::Autocorrelation`]).
pub fn domains(
    coords: &[[f64; 3]],
    columns: &[Vec<f64>],
    k: usize,
    method: Clustering,
    neighbors: usize,
    realizations: usize,
    seed: u64,
) -> Result<Domains> {
    let n = coords.len();
    if columns.is_empty() {
        return invalid("give at least one column");
    }
    if columns.iter().any(|c| c.len() != n) {
        return invalid(format!("expected {n} values in every column"));
    }
    if coords.iter().flatten().any(|x| !x.is_finite()) {
        return invalid("coordinates must be finite");
    }
    if columns.iter().flatten().any(|v| v.is_infinite()) {
        return invalid("values must be finite or NaN");
    }
    if neighbors == 0 || (method != Clustering::Autocorrelation && realizations == 0) {
        return invalid("neighbors and realizations must be positive");
    }
    let rows: Vec<usize> = (0..n)
        .filter(|&i| columns.iter().all(|c| !c[i].is_nan()))
        .collect();
    if k == 0 || k > rows.len() || rows.len() < 2 {
        return invalid(format!(
            "k must be between 1 and the {} samples without NaN, and there must be two",
            rows.len()
        ));
    }
    let m = columns.len();
    let z = standardize(columns, &rows);
    let points: Vec<[f64; 3]> = rows.iter().map(|&i| coords[i]).collect();
    let near = neighbor_lists(&points, (4 * neighbors).min(rows.len() - 1))?;
    let samples = Samples {
        z: &z,
        m,
        near: &near,
    };
    let all: Vec<usize> = (0..m).collect();
    let mut last = Draws::new(realization_seed(seed, u64::MAX));
    let (labels, agreement) = if method == Clustering::Autocorrelation {
        let features = samples.autocorrelation(neighbors, &all);
        (best_kmeans(&features, k, TRIES, &mut last), None)
    } else {
        let ensemble: Vec<Vec<u32>> = (0..realizations)
            .into_par_iter()
            .map(|r| {
                let mut draws = Draws::new(realization_seed(seed, r as u64));
                let size = draws.between(neighbors.div_ceil(2), 2 * neighbors);
                let subset = draws.subset(m);
                let features = if method == Clustering::Consensus && r % 2 == 0 {
                    samples.autocorrelation(size, &subset)
                } else {
                    samples.dual(size, &subset)
                };
                kmeans(&features, k, &mut draws).0
            })
            .collect();
        let encoded = OneHot {
            labels: (0..rows.len())
                .flat_map(|i| ensemble.iter().map(move |e| e[i]))
                .collect(),
            r: realizations,
            k,
        };
        let labels = best_kmeans(&encoded, k, TRIES, &mut last);
        let mut members = vec![vec![]; k];
        for (i, &l) in labels.iter().enumerate() {
            members[l as usize].push(i);
        }
        let centers: Vec<_> = members.iter().map(|m| encoded.mean(m)).collect();
        let agreement: Vec<f64> = (0..rows.len())
            .map(|i| {
                let size = members[labels[i] as usize].len() as f64;
                if size < 2.0 {
                    return 1.0;
                }
                let hits = encoded.hits(i, &centers[labels[i] as usize].0);
                (size * hits / realizations as f64 - 1.0) / (size - 1.0)
            })
            .collect();
        (labels, Some(agreement))
    };
    let mut means: Vec<(f64, usize)> = (0..k)
        .map(|c| {
            let (sum, count) = labels
                .iter()
                .enumerate()
                .filter(|&(_, &l)| l as usize == c)
                .fold((0.0, 0usize), |(s, n), (i, _)| (s + z[i * m], n + 1));
            (
                if count > 0 {
                    sum / count as f64
                } else {
                    f64::INFINITY
                },
                c,
            )
        })
        .collect();
    means.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    let mut order = vec![0u32; k];
    for (rank, &(_, c)) in means.iter().enumerate() {
        order[c] = rank as u32;
    }
    let mut out = vec![None; n];
    for (r, &i) in rows.iter().enumerate() {
        out[i] = Some(order[labels[r] as usize]);
    }
    let agreement = agreement.map(|a| {
        let mut full = vec![None; n];
        for (r, &i) in rows.iter().enumerate() {
            full[i] = Some(a[r]);
        }
        full
    });
    Ok(Domains {
        wcss: wcss(columns, &out)?,
        spatial_entropy: spatial_entropy(coords, &out, neighbors)?,
        labels: out,
        agreement,
    })
}

/// Within-cluster sum of squares of the columns, standardized over the
/// labelled samples, divided by their number: the number of columns for a
/// single domain, 0 when every domain is constant. Unlabelled samples are
/// left out.
pub fn wcss(columns: &[Vec<f64>], labels: &[Option<u32>]) -> Result<f64> {
    let n = labels.len();
    if columns.iter().any(|c| c.len() != n) {
        return invalid(format!("expected {n} values in every column"));
    }
    let rows: Vec<usize> = (0..n).filter(|&i| labels[i].is_some()).collect();
    if rows.is_empty()
        || rows
            .iter()
            .any(|&i| columns.iter().any(|c| !c[i].is_finite()))
    {
        return invalid("labelled samples need finite values");
    }
    let m = columns.len();
    let z = standardize(columns, &rows);
    let k = rows
        .iter()
        .filter_map(|&i| labels[i])
        .max()
        .map_or(0, |c| c as usize + 1);
    let mut sums = vec![0.0; k * m];
    let mut counts = vec![0usize; k];
    for (r, &i) in rows.iter().enumerate() {
        let c = labels[i].expect("labelled") as usize;
        counts[c] += 1;
        for v in 0..m {
            sums[c * m + v] += z[r * m + v];
        }
    }
    let mut total = 0.0;
    for (r, &i) in rows.iter().enumerate() {
        let c = labels[i].expect("labelled") as usize;
        for v in 0..m {
            total += (z[r * m + v] - sums[c * m + v] / counts[c] as f64).powi(2);
        }
    }
    Ok(total / rows.len() as f64)
}

/// Mean over the labelled samples of the entropy of the labels among each
/// sample and its `neighbors` nearest labelled samples, divided by `ln k`
/// for `k` domains so that it lies in [0, 1]: 0 when no neighborhood
/// straddles a contact, 1 when every neighborhood holds all domains equally.
pub fn spatial_entropy(
    coords: &[[f64; 3]],
    labels: &[Option<u32>],
    neighbors: usize,
) -> Result<f64> {
    if coords.len() != labels.len() {
        return invalid(format!("expected {} labels", coords.len()));
    }
    let rows: Vec<usize> = (0..labels.len()).filter(|&i| labels[i].is_some()).collect();
    if rows.len() < 2 || neighbors == 0 {
        return invalid("needs two labelled samples and neighbors > 0");
    }
    let k = rows
        .iter()
        .filter_map(|&i| labels[i])
        .max()
        .map_or(0, |c| c as usize + 1);
    if k < 2 {
        return Ok(0.0);
    }
    let points: Vec<[f64; 3]> = rows.iter().map(|&i| coords[i]).collect();
    let near = neighbor_lists(&points, neighbors.min(rows.len() - 1))?;
    let local: Vec<f64> = (0..rows.len())
        .into_par_iter()
        .map(|r| {
            let mut counts = vec![0usize; k];
            for &j in near[r].iter().chain([&r]) {
                counts[labels[rows[j]].expect("labelled") as usize] += 1;
            }
            let total = (near[r].len() + 1) as f64;
            -counts
                .iter()
                .filter(|&&c| c > 0)
                .map(|&c| {
                    let p = c as f64 / total;
                    p * p.ln()
                })
                .sum::<f64>()
                / (k as f64).ln()
        })
        .collect();
    Ok(local.iter().sum::<f64>() / rows.len() as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use boitata_core::rng::gaussian;

    /// Two stationary regions split at x = 50 on a 60 × 40 grid: the columns
    /// differ by `shift` standard deviations between them.
    fn planted(shift: f64) -> (Vec<[f64; 3]>, Vec<Vec<f64>>, Vec<u32>) {
        let mut coords = vec![];
        let mut truth = vec![];
        let mut columns = vec![vec![], vec![], vec![]];
        for (i, (x, y)) in (0..60)
            .flat_map(|x| (0..40).map(move |y| (x, y)))
            .enumerate()
        {
            let (x, y) = (2.0 * x as f64 + 1.0, 2.0 * y as f64 + 1.0);
            let region = u32::from(x + 10.0 * (y / 15.0).sin() > 60.0);
            coords.push([x, y, 0.0]);
            truth.push(region);
            let s = if region == 1 { shift } else { 0.0 };
            columns[0].push(s + gaussian(realization_seed(1, i as u64)));
            columns[1].push(-s + gaussian(realization_seed(2, i as u64)));
            columns[2].push(gaussian(realization_seed(3, i as u64)));
        }
        (coords, columns, truth)
    }

    fn agreement(labels: &[Option<u32>], truth: &[u32]) -> f64 {
        let same = labels
            .iter()
            .zip(truth)
            .filter(|(l, t)| l.unwrap() == **t)
            .count() as f64;
        let share = same / truth.len() as f64;
        share.max(1.0 - share)
    }

    #[test]
    fn recovers_two_planted_domains() {
        let (coords, columns, truth) = planted(1.5);
        let mut pointwise = Draws::new(9);
        let z = standardize(&columns, &(0..truth.len()).collect::<Vec<_>>());
        let plain = best_kmeans(&Dense { data: z, d: 3 }, 2, TRIES, &mut pointwise);
        let plain: Vec<Option<u32>> = plain.into_iter().map(Some).collect();
        let base = agreement(&plain, &truth);
        for method in [
            Clustering::Autocorrelation,
            Clustering::Dual,
            Clustering::Consensus,
        ] {
            let d = domains(&coords, &columns, 2, method, 16, 40, 7).unwrap();
            let score = agreement(&d.labels, &truth);
            assert!(score >= 0.95, "{method:?} agrees {score:.3}");
            assert!(
                score > base + 0.1,
                "{method:?} {score:.3}, pointwise {base:.3}"
            );
            assert!(
                d.spatial_entropy < spatial_entropy(&coords, &plain, 16).unwrap(),
                "{method:?}"
            );
            if let Some(a) = &d.agreement {
                assert!(
                    a.iter()
                        .all(|a| a.is_some_and(|a| (0.0..=1.0).contains(&a)))
                );
            }
        }
    }

    #[test]
    fn same_domains_at_any_thread_count() {
        let (coords, columns, _) = planted(0.8);
        let run = |threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| domains(&coords, &columns, 3, Clustering::Consensus, 12, 24, 3))
                .unwrap()
        };
        assert_eq!(run(1), run(4));
        assert_ne!(
            run(1).labels,
            domains(&coords, &columns, 3, Clustering::Consensus, 12, 24, 4)
                .unwrap()
                .labels
        );
    }

    #[test]
    fn metrics_on_known_labels() {
        let coords: Vec<[f64; 3]> = (0..8).map(|i| [i as f64, 0.0, 0.0]).collect();
        let values = vec![vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]];
        let split: Vec<Option<u32>> = (0..8).map(|i| Some(u32::from(i >= 4))).collect();
        assert!(wcss(&values, &split).unwrap().abs() < 1e-12);
        assert!((wcss(&values, &[Some(0); 8]).unwrap() - 1.0).abs() < 1e-12);
        let alternating: Vec<Option<u32>> = (0..8).map(|i| Some(i % 2)).collect();
        let contiguous = spatial_entropy(&coords, &split, 1).unwrap();
        assert!(contiguous < spatial_entropy(&coords, &alternating, 1).unwrap());
        assert_eq!(spatial_entropy(&coords, &[Some(0); 8], 2).unwrap(), 0.0);
    }

    #[test]
    fn rows_with_nan_get_no_domain() {
        let (coords, mut columns, _) = planted(1.5);
        columns[1][5] = f64::NAN;
        let d = domains(&coords, &columns, 2, Clustering::Dual, 8, 10, 1).unwrap();
        assert_eq!(d.labels[5], None);
        assert_eq!(d.agreement.unwrap()[5], None);
        assert!(d.labels.iter().filter(|l| l.is_none()).count() == 1);
        assert!(domains(&coords, &columns, 0, Clustering::Dual, 8, 10, 1).is_err());
    }
}
