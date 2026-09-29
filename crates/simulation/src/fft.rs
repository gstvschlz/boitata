//! Cross-correlations by FFT, and the image-quilting cost map computed with
//! them.
//!
//! A direct scan compares a template with an image once per position. A
//! correlation of arrays `a` (the image) and `b` (a kernel over the patch box)
//! gives a sum at every position at once,
//!
//! ```text
//! corr[p] = Σ_l a[p + l] · b[l]
//! ```
//!
//! and every cost [`cost_map`] sums is a handful of them. A squared difference
//! expands into three,
//!
//! ```text
//! Σ w (a − t)²  =  Σ w a²  −  2 Σ a (w t)  +  Σ w t²
//! ```
//!
//! correlations of the image, its square and its mask with kernels built from
//! the template. A categorical mismatch count is one correlation per code,
//! between that code's indicator image and the weights of the template cells
//! holding it. Everything runs in `f64`; continuous values are centered on the
//! middle of their range first, which keeps the three terms small and their
//! difference accurate.
//!
//! [`cost_map`]: crate::quilting::cost_map

use std::sync::Arc;

use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};

use crate::quilting::Term;

/// Cells along x, y and z, x fastest.
pub type Dims = [usize; 3];

/// The next product of 2, 3 and 5 from `n`: rustfft transforms any length,
/// but a smooth one avoids its slower generic path, and the padding is zeros
/// anyway.
fn next_fast(mut n: usize) -> usize {
    n = n.max(1);
    loop {
        let mut m = n;
        for f in [2, 3, 5] {
            while m.is_multiple_of(f) {
                m /= f;
            }
        }
        if m == 1 {
            return n;
        }
        n += 1;
    }
}

/// The padded grid a correlation of these shapes runs on: `image + kernel - 1`
/// along each axis, rounded up to a smooth length.
fn pad_dims(image: Dims, kernel: Dims) -> Dims {
    [0, 1, 2].map(|a| next_fast(image[a] + kernel[a] - 1))
}

/// Cells of the padded grid of a correlation of these shapes: what one
/// spectrum holds.
pub fn padded_cells(image: Dims, kernel: Dims) -> usize {
    pad_dims(image, kernel).iter().product()
}

/// An image spectrum, a kernel spectrum, and the coefficient their
/// correlation carries into a sum.
pub type CorrelationTerm<'a> = (&'a [Complex<f64>], &'a [Complex<f64>], f64);

/// FFT plans for every transform of one (image, kernel) shape. Planning costs
/// more than a transform, so it is done once.
#[derive(Clone)]
pub struct Correlator {
    pad: Dims,
    forward: [Arc<dyn Fft<f64>>; 3],
    inverse: [Arc<dyn Fft<f64>>; 3],
}

impl std::fmt::Debug for Correlator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Correlator")
            .field("pad", &self.pad)
            .finish()
    }
}

impl Correlator {
    /// A correlator for an image of `image` cells and a kernel of `kernel`
    /// cells. The padded grid holds `image + kernel - 1` cells along each
    /// axis, so the circular correlation agrees with the linear one at every
    /// position where the kernel lies in the image.
    pub fn new(image: Dims, kernel: Dims) -> Self {
        let mut planner = FftPlanner::new();
        let pad = pad_dims(image, kernel);
        Self {
            pad,
            forward: pad.map(|n| planner.plan_fft_forward(n)),
            inverse: pad.map(|n| planner.plan_fft_inverse(n)),
        }
    }

    /// Complex values in one spectrum.
    pub fn cells(&self) -> usize {
        self.pad.iter().product()
    }

    /// The spectrum of a real array of `dims` cells at the start of the
    /// padded grid; `value(i)` is its cell `i`, x fastest, read once.
    pub fn spectrum(&self, dims: Dims, value: impl Fn(usize) -> f64) -> Vec<Complex<f64>> {
        let mut out = Vec::new();
        self.spectrum_into(dims, value, &mut out, &mut Vec::new());
        out
    }

    /// [`Correlator::spectrum`] into `out`, through `work`; both keep their
    /// memory from call to call.
    pub fn spectrum_into(
        &self,
        dims: Dims,
        value: impl Fn(usize) -> f64,
        out: &mut Vec<Complex<f64>>,
        work: &mut Vec<Complex<f64>>,
    ) {
        out.clear();
        out.resize(self.cells(), Complex::new(0.0, 0.0));
        let mut i = 0;
        for z in 0..dims[2] {
            for y in 0..dims[1] {
                let row = (z * self.pad[1] + y) * self.pad[0];
                for x in 0..dims[0] {
                    out[row + x].re = value(i);
                    i += 1;
                }
            }
        }
        self.transform(out, work, &self.forward);
    }

    /// `Σ c · corr(image, kernel)` over `terms`, into `out` through `buf` and
    /// `work`. A correlation is linear, so the whole sum takes one inverse
    /// transform. `out` lies on the padded grid: `corr[p]` is at
    /// `out[self.at(p)]`.
    pub fn combine(
        &self,
        terms: &[CorrelationTerm<'_>],
        buf: &mut Vec<Complex<f64>>,
        work: &mut Vec<Complex<f64>>,
        out: &mut Vec<f64>,
    ) {
        buf.clear();
        buf.resize(self.cells(), Complex::new(0.0, 0.0));
        for &(image, kernel, coefficient) in terms {
            for ((acc, a), b) in buf.iter_mut().zip(image).zip(kernel) {
                *acc += a * b.conj() * coefficient;
            }
        }
        self.transform(buf, work, &self.inverse);
        // rustfft leaves the inverse unscaled.
        let scale = 1.0 / self.cells() as f64;
        out.clear();
        out.extend(buf.iter().map(|c| c.re * scale));
    }

    /// Index of position `p` in the output of [`Correlator::combine`].
    pub fn at(&self, p: Dims) -> usize {
        (p[2] * self.pad[1] + p[1]) * self.pad[0] + p[0]
    }

    /// A 3D transform as 1D transforms along each axis in turn. After each
    /// axis the array is rotated, `(x, y, z)` to `(y, z, x)`, so the next axis
    /// is contiguous; three rotations restore it.
    fn transform(
        &self,
        buf: &mut Vec<Complex<f64>>,
        work: &mut Vec<Complex<f64>>,
        plans: &[Arc<dyn Fft<f64>>; 3],
    ) {
        let mut dims = self.pad;
        work.resize(buf.len(), Complex::new(0.0, 0.0));
        for plan in plans {
            let [nx, ny, nz] = dims;
            dims = [ny, nz, nx];
            if nx == 1 {
                // Moving an axis of 1 cell leaves the array as it is.
                continue;
            }
            plan.process(buf);
            // Eight columns at a time, which reads whole cache lines.
            let rows = ny * nz;
            for x0 in (0..nx).step_by(8) {
                let x1 = (x0 + 8).min(nx);
                for i in 0..rows {
                    for x in x0..x1 {
                        work[x * rows + i] = buf[x + nx * i];
                    }
                }
            }
            std::mem::swap(buf, work);
        }
    }
}

/// The spectra of one image a cost reads.
#[derive(Debug, Clone)]
enum ImageSpectra {
    /// One indicator per code, 1 where the image holds it; a no-data cell is
    /// 0 in every one, so it matches nothing.
    Categorical(Vec<Vec<Complex<f64>>>),
    /// The centered values, their squares, and the mask of informed cells.
    Continuous {
        center: f64,
        /// Largest distance of a value from `center`.
        spread: f64,
        spectra: [Vec<Complex<f64>>; 3],
    },
}

/// Spectra of a continuous image, in [`ImageSpectra::Continuous`] order.
const VALUES: usize = 0;
const SQUARES: usize = 1;
const INFORMED: usize = 2;

impl ImageSpectra {
    fn len(&self) -> usize {
        match self {
            Self::Categorical(indicators) => indicators.len(),
            Self::Continuous { .. } => 3,
        }
    }

    fn get(&self, k: usize) -> &[Complex<f64>] {
        match self {
            Self::Categorical(indicators) => &indicators[k],
            Self::Continuous { spectra, .. } => &spectra[k],
        }
    }
}

/// Buffers one realization reuses from patch to patch.
#[derive(Debug, Default)]
pub struct CostScratch {
    kernels: Vec<Vec<Complex<f64>>>,
    buf: Vec<Complex<f64>>,
    work: Vec<Complex<f64>>,
    corr: Vec<f64>,
}

/// [`cost_map`] by FFT: the spectra of the images the costs read, transformed
/// once and read at every patch.
///
/// A cost costs one forward transform per kernel it needs and one inverse
/// transform, whatever the number of cells it compares: a kernel per code of
/// a categorical image that the templates hold, three per continuous image.
///
/// [`cost_map`]: crate::quilting::cost_map
#[derive(Debug, Clone)]
pub struct CostFft {
    corr: Correlator,
    image_dims: Dims,
    patch: Dims,
    images: Vec<ImageSpectra>,
}

impl CostFft {
    /// Transforms `images`, each as (values, categorical), x fastest, of
    /// `image_dims` cells, NaN where there is no data, for the costs of
    /// patches of `patch` cells.
    pub fn new(image_dims: Dims, patch: Dims, images: &[(&[f32], bool)]) -> Self {
        let corr = Correlator::new(image_dims, patch);
        let spectra = |values: &[f32], categorical: bool| {
            if categorical {
                let k = values
                    .iter()
                    .filter(|v| !v.is_nan())
                    .fold(-1.0f32, |a, &b| a.max(b));
                let k = (k + 1.0).max(0.0) as usize;
                let indicators = (0..k)
                    .map(|c| {
                        let c = c as f32;
                        corr.spectrum(image_dims, |i| f64::from(u8::from(values[i] == c)))
                    })
                    .collect();
                return ImageSpectra::Categorical(indicators);
            }
            let [lo, hi] = values
                .iter()
                .filter(|v| !v.is_nan())
                .fold([f64::INFINITY, f64::NEG_INFINITY], |[lo, hi], &v| {
                    [lo.min(f64::from(v)), hi.max(f64::from(v))]
                });
            let center = if lo <= hi { 0.5 * (lo + hi) } else { 0.0 };
            let at = |i: usize| match values[i].is_nan() {
                true => 0.0,
                false => f64::from(values[i]) - center,
            };
            ImageSpectra::Continuous {
                center,
                spread: if lo <= hi { 0.5 * (hi - lo) } else { 0.0 },
                spectra: [
                    corr.spectrum(image_dims, at),
                    corr.spectrum(image_dims, |i| at(i) * at(i)),
                    corr.spectrum(image_dims, |i| f64::from(u8::from(!values[i].is_nan()))),
                ],
            }
        };
        Self {
            images: images.iter().map(|&(v, c)| spectra(v, c)).collect(),
            corr,
            image_dims,
            patch,
        }
    }

    /// The costs of [`cost_map`] for `terms`, to rounding (see
    /// [`CostFft::error_bound`]). Term `i` reads image `images[i]` of those
    /// given to [`CostFft::new`], and must be of the same kind.
    ///
    /// [`cost_map`]: crate::quilting::cost_map
    pub fn costs(
        &self,
        terms: &[Term<'_>],
        images: &[usize],
        scratch: &mut CostScratch,
    ) -> Vec<f64> {
        let cells: usize = self.patch.iter().product();
        // One kernel per image spectrum a term meets, coefficients folded in.
        let offsets: Vec<usize> = self
            .images
            .iter()
            .scan(0, |next, image| {
                let at = *next;
                *next += image.len();
                Some(at)
            })
            .collect();
        let n_spectra = self.images.iter().map(ImageSpectra::len).sum();
        let mut kernels: Vec<Option<Vec<f64>>> = vec![None; n_spectra];
        let mut add = |slot: usize, cell: usize, value: f64| {
            kernels[slot].get_or_insert_with(|| vec![0.0; cells])[cell] += value;
        };
        // Every compared cell counts a full mismatch; the correlations take
        // back what the image matches.
        let mut total = 0.0;
        for (term, &image) in terms.iter().zip(images) {
            let compared = term
                .template
                .iter()
                .zip(term.weights)
                .enumerate()
                .filter(|(_, (value, weight))| !value.is_nan() && **weight > 0.0);
            match &self.images[image] {
                ImageSpectra::Categorical(indicators) => {
                    for (cell, (&value, &weight)) in compared {
                        total += weight;
                        let code = value as usize;
                        // A code the image never holds matches nothing.
                        if code < indicators.len() {
                            add(offsets[image] + code, cell, -weight);
                        }
                    }
                }
                ImageSpectra::Continuous { center, .. } => {
                    let r2 = term.inv_range * term.inv_range;
                    for (cell, (&value, &weight)) in compared {
                        total += weight;
                        let t = f64::from(value) - center;
                        add(offsets[image] + SQUARES, cell, r2 * weight);
                        add(offsets[image] + VALUES, cell, -2.0 * r2 * weight * t);
                        // A no-data image cell stays a full mismatch.
                        add(
                            offsets[image] + INFORMED,
                            cell,
                            r2 * weight * t * t - weight,
                        );
                    }
                }
            }
        }

        let used: Vec<(usize, Vec<f64>)> = kernels
            .into_iter()
            .enumerate()
            .filter_map(|(slot, k)| k.map(|k| (slot, k)))
            .collect();
        scratch
            .kernels
            .resize_with(used.len().max(scratch.kernels.len()), Vec::new);
        for ((_, kernel), out) in used.iter().zip(&mut scratch.kernels) {
            self.corr
                .spectrum_into(self.patch, |c| kernel[c], out, &mut scratch.work);
        }
        let spectrum = |slot: usize| {
            let image = offsets.partition_point(|&o| o <= slot) - 1;
            self.images[image].get(slot - offsets[image])
        };
        let parts: Vec<CorrelationTerm<'_>> = used
            .iter()
            .zip(&scratch.kernels)
            .map(|((slot, _), kernel)| (spectrum(*slot), kernel.as_slice(), 1.0))
            .collect();
        self.corr.combine(
            &parts,
            &mut scratch.buf,
            &mut scratch.work,
            &mut scratch.corr,
        );

        let [mx, my, mz]: Dims = std::array::from_fn(|a| self.image_dims[a] - self.patch[a] + 1);
        let mut costs = Vec::with_capacity(mx * my * mz);
        for tz in 0..mz {
            for ty in 0..my {
                let row = self.corr.at([0, ty, tz]);
                costs.extend(scratch.corr[row..row + mx].iter().map(|c| total + c));
            }
        }
        costs
    }

    /// A bound on how far a cost of [`CostFft::costs`] lies from the direct
    /// scan's: a tiny share of the largest sum the correlations meet.
    pub fn error_bound(&self, terms: &[Term<'_>], images: &[usize]) -> f64 {
        let mut largest = 0.0;
        for (term, &image) in terms.iter().zip(images) {
            let scale = match &self.images[image] {
                ImageSpectra::Categorical(_) => 1.0,
                ImageSpectra::Continuous { center, spread, .. } => {
                    let t = term
                        .template
                        .iter()
                        .filter(|v| !v.is_nan())
                        .fold(0.0f64, |m, &v| m.max((f64::from(v) - center).abs()));
                    let r = term.inv_range;
                    1f64.max((spread * r).powi(2)).max((t * r).powi(2))
                }
            };
            let weight: f64 = term
                .template
                .iter()
                .zip(term.weights)
                .filter(|(v, w)| !v.is_nan() && **w > 0.0)
                .map(|(_, w)| w)
                .sum();
            largest += scale * weight;
        }
        1e-7 * largest
    }
}

/// Kernels a patch's costs transform: one per code of a categorical image its
/// terms hold, three per continuous image. Term `i` reads image `images[i]`.
pub fn n_kernels(terms: &[Term<'_>], images: &[usize]) -> usize {
    let mut slots: Vec<(usize, u32)> = Vec::new();
    for (term, &image) in terms.iter().zip(images) {
        let cells = term.template.iter().zip(term.weights);
        for (value, _) in cells.filter(|(v, w)| !v.is_nan() && **w > 0.0) {
            let slot = match term.categorical {
                true => (image, *value as u32),
                false => (image, u32::MAX),
            };
            if !slots.contains(&slot) {
                slots.push(slot);
            }
        }
    }
    slots
        .iter()
        .map(|&(_, code)| if code == u32::MAX { 3 } else { 1 })
        .sum()
}

/// What one cell of one transform costs in cells of the direct scan, on a
/// padded grid of one layer and of several. The transforms run on one
/// thread, the direct scan on all of them, and a 3D array pays three
/// rotations per transform against the scan's short, cached rows.
///
/// Measured with the `image quilting` group of the simulation bench on 24
/// threads, milliseconds for one patch's costs:
///
/// ```text
/// image          patch     compared  kernels  direct   FFT  break-even
/// 250x250        40x40          444        2    2.73   1.23               0.5
/// 250x250        40x40   444 + 1600        5    8.43   2.62               0.3
/// 64x64x32       20x20x10      5399        2    4.76   9.30                24
/// ```
///
/// The last column is the value of the constant at which the two paths take
/// the same time.
const FFT_WORK_FLAT: f64 = 0.5;
const FFT_WORK_DEEP: f64 = 20.0;

/// Whether one patch's costs are cheaper by FFT than by the direct scan: the
/// scan costs `positions × compared` operations, the FFT a transform per
/// kernel and one more, each `padded × log2(padded)`.
pub fn fft_pays(image_dims: Dims, patch: Dims, compared: usize, n_kernels: usize) -> bool {
    let positions: usize = (0..3).map(|a| image_dims[a] - patch[a] + 1).product();
    let padded = padded_cells(image_dims, patch);
    let per_cell = match image_dims[2] > 1 {
        true => FFT_WORK_DEEP,
        false => FFT_WORK_FLAT,
    };
    let fft = per_cell * ((n_kernels + 1) * padded) as f64 * (padded.max(2) as f64).log2();
    positions as f64 * compared as f64 > fft
}

/// Bytes the FFT path holds: `n_spectra` image spectra for the run, and for
/// each of `threads` realizations at once up to `n_kernels` kernel spectra,
/// the buffer they are summed in and the map that comes back.
pub fn fft_bytes(
    image_dims: Dims,
    patch: Dims,
    n_spectra: usize,
    n_kernels: usize,
    threads: usize,
) -> usize {
    let padded = padded_cells(image_dims, patch);
    let complex = size_of::<Complex<f64>>();
    let working = padded * (complex * (n_kernels + 1) + size_of::<f64>());
    padded * complex * n_spectra + working * threads
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quilting::cost_map;

    /// A seeded pseudo-random value in `[0, 1)`.
    fn noise(i: usize, seed: u64) -> f64 {
        let h = ceres_core::rng::splitmix(i as u64 ^ seed.wrapping_mul(0x9e37_79b9));
        (h >> 11) as f64 / (1u64 << 53) as f64
    }

    fn continuous_image(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| match i % 13 {
                5 => f32::NAN,
                _ => (1000.0 + 4.0 * noise(i, 1)) as f32,
            })
            .collect()
    }

    fn categorical_image(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| match i % 17 {
                3 => f32::NAN,
                _ => (noise(i, 2) * 3.0).floor() as f32,
            })
            .collect()
    }

    /// A template of `cells` with no data at every fifth cell and weight 0 at
    /// every seventh.
    fn template(cells: usize, categorical: bool, seed: u64) -> (Vec<f32>, Vec<f64>) {
        let values = (0..cells)
            .map(|c| match (c % 5, categorical) {
                (0, _) => f32::NAN,
                (_, true) => (noise(c, seed) * 4.0).floor() as f32,
                (_, false) => (999.0 + 6.0 * noise(c, seed)) as f32,
            })
            .collect();
        let weights = (0..cells)
            .map(|c| if c % 7 == 0 { 0.0 } else { noise(c, seed + 9) })
            .collect();
        (values, weights)
    }

    #[test]
    fn next_fast_lengths_are_products_of_2_3_and_5() {
        assert_eq!(
            [1, 7, 11, 13, 31, 97].map(next_fast),
            [1, 8, 12, 15, 32, 100]
        );
    }

    #[test]
    fn fft_costs_equal_the_direct_scan_to_rounding() {
        for (dims, patch) in [([13, 11, 1], [5, 4, 1]), ([9, 7, 5], [3, 3, 2])] {
            let n: usize = dims.iter().product();
            let cells: usize = patch.iter().product();
            let (cont, cat) = (continuous_image(n), categorical_image(n));
            let fft = CostFft::new(dims, patch, &[(&cont, false), (&cat, true)]);
            let t: Vec<_> = (0..3).map(|s| template(cells, s != 0, s)).collect();
            let term = |image: &'static str, s: usize| Term {
                image: if image == "cont" { &cont } else { &cat },
                categorical: image == "cat",
                inv_range: if image == "cat" { 0.0 } else { 0.25 },
                template: &t[s].0,
                weights: &t[s].1,
            };
            let cases: [(Vec<Term<'_>>, Vec<usize>); 4] = [
                (vec![term("cont", 0)], vec![0]),
                (vec![term("cat", 1)], vec![1]),
                (vec![term("cat", 1), term("cont", 0)], vec![1, 0]),
                // Two terms on one image, as the soft probabilities are.
                (vec![term("cat", 1), term("cat", 2)], vec![1, 1]),
            ];
            let mut scratch = CostScratch::default();
            for (terms, images) in &cases {
                let want = cost_map(dims, patch, terms);
                let got = fft.costs(terms, images, &mut scratch);
                let bound = fft.error_bound(terms, images);
                assert_eq!(want.len(), got.len());
                for (w, g) in want.iter().zip(&got) {
                    assert!((w - g).abs() < 1e-9 * w.abs().max(1.0), "{w} vs {g}");
                    assert!((w - g).abs() < bound, "{w} vs {g}, bound {bound}");
                }
            }
        }
    }

    #[test]
    fn a_patch_pays_for_its_transforms_when_it_compares_many_cells() {
        let (image, patch) = ([250, 250, 1], [40, 40, 1]);
        assert!(!fft_pays(image, patch, 20, 2));
        assert!(fft_pays(image, patch, 1600, 2));
        assert!(fft_bytes(image, patch, 2, 3, 4) > fft_bytes(image, patch, 2, 3, 1));
    }
}
