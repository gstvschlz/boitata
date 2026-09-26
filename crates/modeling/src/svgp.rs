//! Sparse variational Gaussian process — the implicit engine that does not
//! decimate.
//!
//! [`crate::rbf`] and [`crate::hermite`] both factorize one dense `n × n`
//! system, so both cost `O(n³)` time and `O(n²)` memory, and both have to be
//! handed a decimated constraint set. This engine replaces the dense solve with
//! `m` *inducing* values `u = f(Z)`: `O(nm²)` time and `O(m²)` memory, with `m`
//! in the hundreds whatever `n` is. Every composite informs the fit, and the
//! posterior *variance* falls out of the formulation rather than being a second
//! computation bolted onto the mean.
//!
//! # The bound
//!
//! Titsias (2009), "Variational Learning of Inducing Variables in Sparse
//! Gaussian Processes", *AISTATS*, in its **collapsed** form: the variational
//! distribution `q(u)` is eliminated analytically, leaving a bound in the
//! hyperparameters alone,
//!
//! ```text
//! L = log N(ŷ | 0, Qff + Σ) − ½ tr(Σ⁻¹(K̃ff − Q̃ff)),    Qff = Kfu Kuu⁻¹ Kuf
//! ```
//!
//! where `~` marks the diagonal. The first term is the DTC / projected-process
//! likelihood; the second is the correction that makes this a *bound* on the
//! exact marginal likelihood rather than a different model — it charges the fit
//! for the variance the inducing set fails to explain, which is what stops the
//! lengthscales collapsing onto a degenerate solution.
//!
//! Everything is evaluated through `Luu = chol(Kuu)` and `B = I + ÃÃᵀ`, so no
//! `n × n` object is ever formed. See [`Fitter::elbo_and_gradient`].
//!
//! # Why the gradients are hand-derived
//!
//! The choice #320 asked to be made deliberately, recorded here rather than in
//! a planning document:
//!
//! - Reverse-mode autodiff costs 2–4× a forward pass. So do these analytic
//!   gradients, because they reuse `Luu` and `Lb` — the two factorizations the
//!   bound has already paid for. There is no speed to buy.
//! - The expensive operation in the bound is a Cholesky, and no Rust tensor
//!   framework (`candle`, `burn`, `dfdx`) ships a differentiable one. Its
//!   backward pass would have to be hand-derived regardless, so the framework
//!   would be carried for the easy half of the work.
//! - Those frameworks are f32-first. A kernel matrix that needs jitter to
//!   factorize at all is exactly the case where f32 is not enough.
//!
//! The collapsed bound is what makes this tractable: `q(u)`'s mean and
//! covariance are closed-form, so the optimizer only ever sees five scalars —
//! the signal variance, three lengthscales and the noise.
//!
//! # Covariance
//!
//! The kernel is a `variogram::Model` shape read as a correlation function,
//! `ρ(r) = 1 − g(r)`, over the *reduced* lag `r = ‖D(x − x′)‖` with
//! `D = diag(1/ℓ)·R` ([`AnisoTransform`]). Reusing the variogram shapes means
//! the learned `ℓ` are practical ranges on the same convention as a fitted
//! variogram — directly comparable with a Variography step — and that
//! `variogram::is_differentiable` stays the single authority on which kernels
//! can take a structural reading. Hensman, Fusi & Lawrence (2013) is the
//! stochastic route to the same bound; it is not used here, because a
//! deterministic optimizer over five parameters is both reproducible and
//! enough.

use std::collections::VecDeque;

use nalgebra::{DMatrix, DVector, Matrix3, Vector3};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use variogram::{Angles, Model, is_differentiable, shape, shape_d1, shape_d2};

use crate::aniso::AnisoTransform;
use crate::constraint::{ConstraintSet, ValueKind};
use crate::error::{ModelError, Result};
use crate::implicit::decimate;

/// Lags below this are treated as coincident. Every quantity that divides by
/// `r` has a finite limit at zero; this is where the limit is taken instead.
const ZERO_LAG: f64 = 1e-12;

/// Observations processed per pass over the data. Bounds the working set to
/// `O(m · CHUNK)`, so `n` never appears in the memory footprint.
const CHUNK: usize = 1024;

/// Noise weight of a boundary pick relative to an ordinary sample when the
/// caller asks for zero tolerance. A pick is a known zero of the field, so it
/// is trusted far harder than a coding — but not infinitely, because `Σ` has to
/// stay invertible.
const BOUNDARY_WEIGHT_FLOOR: f64 = 0.01;

/// Points the inducing placement clusters over. `k`-means on 10⁶ composites
/// would cost more than the fit it initializes, and the placement does not need
/// every point to find the shape of the data.
const PLACEMENT_SAMPLE: usize = 20_000;

/// Lloyd iterations for the inducing placement.
const PLACEMENT_ITERATIONS: usize = 10;

/// L-BFGS history length.
const LBFGS_MEMORY: usize = 8;

/// Backtracking steps before a line search is called stalled.
const LINE_SEARCH_STEPS: usize = 24;

/// Largest move any one log-parameter may make in a single iteration — one
/// decade.
///
/// Without it the first step is raw steepest descent, and the bound's gradient
/// on a real dataset is `O(10³)`: the trial point lands thousands of decades
/// away, the projection pins it to a corner of the parameter box, and because
/// that corner still beats the starting point the Armijo test *accepts* it. The
/// fit then reports a converged optimum that is nothing but the bounds.
const MAX_LOG_STEP: f64 = std::f64::consts::LN_10;

/// Jitter ladder, relative to the signal variance. `Kuu` is a kernel matrix
/// over points the placement deliberately spread out, but two inducing points
/// can still land close enough to make it numerically singular.
const JITTER_LADDER: [f64; 4] = [1e-10, 1e-8, 1e-6, 1e-4];

/// How far each log-parameter may travel from where it started, in decades.
const PARAMETER_DECADES: f64 = 3.0;

// ------------------------------------------------------------------- kernel

/// `ρ(r)` — correlation at reduced (dimensionless) lag.
fn rho(model: Model, r: f64) -> f64 {
    1.0 - shape(model, r, 1.0)
}

/// `ρ′(r)`.
fn rho_d1(model: Model, r: f64) -> f64 {
    -shape_d1(model, r, 1.0)
}

/// `ρ″(r)`.
fn rho_d2(model: Model, r: f64) -> f64 {
    -shape_d2(model, r, 1.0)
}

/// `g(r) = ρ′(r)/r`, the coefficient of the separation vector in `∇ρ`.
///
/// Split out for the same reason [`crate::rbf::Kernel::dphi_over_r`] is: its
/// limit at the origin is the whole question of whether the kernel admits a
/// derivative observation. For a differentiable model `ρ′(0) = 0` and the limit
/// is `ρ″(0)`; for the rest `ρ′(0) ≠ 0` and there is no limit, which is why
/// [`SvgpSpec::validate`] refuses those models by name the moment a structural
/// reading is present.
fn rho_g(model: Model, r: f64) -> f64 {
    if r <= ZERO_LAG {
        rho_d2(model, 0.0)
    } else {
        rho_d1(model, r) / r
    }
}

/// `g′(r) = (ρ″(r) − g(r))/r`.
///
/// Zero at the origin: `g` is even in `r`, so its slope there vanishes.
fn rho_g_d1(model: Model, r: f64) -> f64 {
    if r <= ZERO_LAG {
        0.0
    } else {
        (rho_d2(model, r) - rho_g(model, r)) / r
    }
}

/// Prior variance of `∇f·d`, per unit signal variance, for a reduced direction
/// `w = Dd`: `−ρ″(0)‖w‖²`. Positive for every kernel this engine accepts.
fn derivative_prior(model: Model, w: &Vector3<f64>) -> f64 {
    -rho_d2(model, 0.0) * w.norm_squared()
}

// --------------------------------------------------------------- parameters

/// How the fit ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Convergence {
    /// The gradient, or the change in the bound, fell under the tolerance.
    Converged,
    /// The iteration cap was reached with the bound still moving.
    MaxIterations,
    /// The line search could not find an improving step — usually a bound
    /// already flat to machine precision.
    LineSearchStalled,
    /// Hyperparameter learning was switched off; the starting values were used
    /// as given.
    NotAttempted,
}

impl Convergence {
    pub fn as_str(self) -> &'static str {
        match self {
            Convergence::Converged => "converged",
            Convergence::MaxIterations => "maxIterations",
            Convergence::LineSearchStalled => "lineSearchStalled",
            Convergence::NotAttempted => "notAttempted",
        }
    }

    /// Whether the fit is one to take at face value. `MaxIterations` is the one
    /// that means "the answer was still moving when we stopped".
    pub fn is_settled(self) -> bool {
        !matches!(self, Convergence::MaxIterations)
    }
}

/// What the fit did, for the result card. An iterative fit can converge slowly,
/// land in a poor optimum or stall, and none of that shows in the surface it
/// produces — so it is reported rather than hidden behind a spinner.
#[derive(Debug, Clone)]
pub struct FitReport {
    pub status: Convergence,
    pub iterations: usize,
    /// Evidence lower bound at the returned hyperparameters, over the full
    /// observation set.
    pub elbo: f64,
    /// Infinity norm of the bound's gradient when the optimizer stopped.
    pub gradient_norm: f64,
    pub inducing: usize,
    /// Observation rows the final fit conditioned on — values, boundary picks
    /// and derivative rows, undecimated.
    pub observations: usize,
    /// Rows the hyperparameter search itself saw (see [`SvgpSpec::hyper_sample`]).
    pub hyper_observations: usize,
    pub signal_variance: f64,
    pub noise_variance: f64,
    /// Learned practical ranges along the ARD axes, in meters.
    pub lengthscales: [f64; 3],
    /// Relative jitter the Cholesky of `Kuu` needed.
    pub jitter: f64,
}

/// Everything the sparse engine is configured by.
#[derive(Debug, Clone)]
pub struct SvgpSpec {
    /// Correlation shape. Any `variogram::Model` except `Power`, which is not a
    /// stationary covariance.
    pub model: Model,
    /// Number of inducing values `m`. Cost is `O(nm²)` to fit and `O(m²)` per
    /// evaluated node, so this is the one knob that buys accuracy with time.
    pub inducing: usize,
    /// Rotation (azimuth, dip, rake) of the ARD frame. The three
    /// lengthscales along those axes are learned; the rotation is not — a
    /// structural frame is something the geologist knows and the bound is poor
    /// at recovering.
    pub rotation: [f64; 3],
    /// 0 = constant mean, 1 = a linear trend removed before the GP and added
    /// back after.
    pub drift_degree: usize,
    /// Noise weight of a boundary pick relative to a sample; 0 means "as close
    /// to exact as the system allows" (see [`BOUNDARY_WEIGHT_FLOOR`]).
    pub boundary_tolerance: f64,
    /// Extra slack on structural rows, as a multiple of a sample's noise. 0
    /// trusts a dip exactly as far as an assay.
    pub gradient_nugget: f64,
    /// Maximize the bound over the hyperparameters. Off keeps the starting
    /// values, which makes this a plain sparse GP at a fixed covariance.
    pub learn_hyperparameters: bool,
    pub max_iterations: usize,
    /// Stop when the gradient's infinity norm falls below this.
    pub tolerance: f64,
    /// Seeds the inducing placement. Persisted by the caller, because a
    /// resource model that does not reproduce is not a resource model.
    pub seed: u64,
    /// Rows the hyperparameter search sees, sample rows decimated to fit. The
    /// *fit* always uses everything; five scalars are identifiable from far
    /// fewer rows than the posterior needs, and every search step costs
    /// `O(nm²)`.
    pub hyper_sample: usize,
    /// Starting lengthscales in meters. `None` derives them from the extent and
    /// the inducing count (see [`default_lengthscales`]).
    pub initial_lengthscales: Option<[f64; 3]>,
    /// Starting noise as a fraction of the signal variance. A tenth: generous
    /// enough that the trace correction does not dominate the first step, small
    /// enough that the data still has to be explained.
    pub initial_noise: f64,
}

impl Default for SvgpSpec {
    fn default() -> Self {
        Self {
            model: Model::Matern { order: 2.5 },
            inducing: 256,
            rotation: [0.0; 3],
            drift_degree: 0,
            boundary_tolerance: 0.0,
            gradient_nugget: 0.0,
            learn_hyperparameters: true,
            max_iterations: 100,
            tolerance: 1e-5,
            seed: 42,
            hyper_sample: 5_000,
            initial_lengthscales: None,
            initial_noise: 0.1,
        }
    }
}

impl SvgpSpec {
    fn validate(&self, has_derivatives: bool) -> Result<()> {
        if matches!(self.model, Model::Power { .. }) {
            return Err(ModelError::InvalidParameter(
                "the power model is not a stationary covariance and cannot drive a Gaussian \
                 process; pick a bounded kernel"
                    .into(),
            ));
        }
        if self.inducing < 2 {
            return Err(ModelError::InvalidParameter(
                "need at least 2 inducing points".into(),
            ));
        }
        if self.boundary_tolerance < 0.0 || self.gradient_nugget < 0.0 {
            return Err(ModelError::InvalidParameter(
                "boundary tolerance and structural nugget must not be negative".into(),
            ));
        }
        if !self.initial_noise.is_finite() || self.initial_noise <= 0.0 {
            return Err(ModelError::InvalidParameter(
                "initial noise must be positive".into(),
            ));
        }
        if let Some(l) = self.initial_lengthscales
            && l.iter().any(|v| !v.is_finite() || *v <= 0.0)
        {
            return Err(ModelError::InvalidParameter(
                "initial lengthscales must be positive".into(),
            ));
        }
        // Same rule, same authority and the same by-name refusal as `hermite`:
        // a gradient observation on a nowhere-differentiable field constrains a
        // derivative that does not exist.
        if has_derivatives && !is_differentiable(self.model) {
            return Err(ModelError::InvalidParameter(format!(
                "the {:?} kernel cannot take structural data: it leaves the origin with a \
                 finite slope, so a field drawn from it is continuous but nowhere \
                 differentiable and its gradient does not exist. Use a Gaussian, cubic or \
                 Matérn (ν ≥ 3/2) kernel.",
                self.model
            )));
        }
        Ok(())
    }
}

// ------------------------------------------------------------ observations

/// The fit's rows, values first and derivatives after, so an index tells the
/// kind. Both are linear functionals of the field, which is what lets one
/// Gaussian model carry an assay and a dip in the same system.
#[derive(Debug, Clone)]
struct Observations {
    at: Vec<[f64; 3]>,
    /// World-space unit directions, one per *derivative* row; indexed by
    /// `i − values`.
    dirs: Vec<[f64; 3]>,
    /// Right-hand side, already detrended.
    y: Vec<f64>,
    /// Noise weight: `σᵢ² = σ²·wᵢ`.
    weight: Vec<f64>,
    /// True for an ordinary sample — the only kind the search decimates.
    sample: Vec<bool>,
    /// Rows before the first derivative row.
    values: usize,
}

impl Observations {
    fn len(&self) -> usize {
        self.at.len()
    }

    fn is_derivative(&self, i: usize) -> bool {
        i >= self.values
    }

    fn direction(&self, i: usize) -> [f64; 3] {
        self.dirs[i - self.values]
    }
}

/// The mean function removed before the GP sees the data and added back after.
///
/// A GP models departures from a mean; leaving a deposit-wide trend in the
/// residual would force the lengthscales to grow until the kernel could
/// represent it, which is a worse fit *and* a worse variance. Degree 1 is the
/// same linear drift the other two engines offer.
#[derive(Debug, Clone)]
struct Trend {
    /// `[c, bx, by, bz]`; the last three are zero for a constant mean.
    coefficients: [f64; 4],
}

impl Trend {
    fn constant(mean: f64) -> Self {
        Self {
            coefficients: [mean, 0.0, 0.0, 0.0],
        }
    }

    fn value(&self, p: &[f64; 3]) -> f64 {
        let c = &self.coefficients;
        c[0] + c[1] * p[0] + c[2] * p[1] + c[3] * p[2]
    }

    /// Directional derivative of the trend — what a structural row has to have
    /// removed from it, since the GP only ever sees the residual.
    fn slope(&self, d: &[f64; 3]) -> f64 {
        let c = &self.coefficients;
        c[1] * d[0] + c[2] * d[1] + c[3] * d[2]
    }

    /// Least squares over the value rows. Falls back to the mean when the
    /// system is rank-deficient — five collinear drillholes should not fail a
    /// run, they should get a constant.
    fn fit(locations: &[[f64; 3]], values: &[f64], degree: usize) -> Self {
        let n = values.len();
        let mean = values.iter().sum::<f64>() / n as f64;
        // A degree-1 drift has four coefficients; fitting it to fewer than
        // twice that is fitting the noise.
        if degree == 0 || n < 8 {
            return Trend::constant(mean);
        }
        // Centered coordinates: a mine-grid easting of 5·10⁵ against a residual
        // of ±1 loses most of the mantissa in the normal equations.
        let mut origin = [0.0; 3];
        for p in locations {
            for k in 0..3 {
                origin[k] += p[k] / n as f64;
            }
        }
        let mut ata = DMatrix::<f64>::zeros(4, 4);
        let mut atb = DVector::<f64>::zeros(4);
        for (p, v) in locations.iter().zip(values) {
            let row = [1.0, p[0] - origin[0], p[1] - origin[1], p[2] - origin[2]];
            for a in 0..4 {
                for b in 0..4 {
                    ata[(a, b)] += row[a] * row[b];
                }
                atb[a] += row[a] * v;
            }
        }
        match ata.lu().solve(&atb) {
            Some(c) if c.iter().all(|v| v.is_finite()) => Self {
                // Undo the centering: `c₀ − b·origin` is the world intercept.
                coefficients: [
                    c[0] - c[1] * origin[0] - c[2] * origin[1] - c[3] * origin[2],
                    c[1],
                    c[2],
                    c[3],
                ],
            },
            _ => Trend::constant(mean),
        }
    }
}

// -------------------------------------------------------------- the engine

/// A fitted sparse variational GP. Evaluate the posterior with [`Svgp::value`]
/// and [`Svgp::variance`].
#[derive(Debug, Clone)]
pub struct Svgp {
    model: Model,
    metric: AnisoTransform,
    signal_variance: f64,
    trend: Trend,
    /// Inducing locations `Z`, in world coordinates.
    z: Vec<[f64; 3]>,
    /// `Luu⁻ᵀ Lb⁻ᵀ c` — contract with `k(x*, Z)` for the posterior mean.
    alpha: DVector<f64>,
    /// `Kuu⁻¹ − Luu⁻ᵀ B⁻¹ Luu⁻¹` — the quadratic form the posterior variance
    /// subtracts from the prior. Symmetric positive semi-definite.
    var_form: DMatrix<f64>,
    report: FitReport,
}

impl Svgp {
    /// Fit the sparse GP to `set`.
    ///
    /// The constraint set is used whole — there is no sample cap here, which is
    /// the entire point of the engine.
    pub fn fit(set: &ConstraintSet, spec: &SvgpSpec) -> Result<Self> {
        set.validate()?;
        spec.validate(!set.derivatives.is_empty())?;

        let value_locs: Vec<[f64; 3]> = set.values.iter().map(|v| v.at).collect();
        let raw_values: Vec<f64> = set.values.iter().map(|v| v.value).collect();
        let trend = Trend::fit(&value_locs, &raw_values, spec.drift_degree.min(1));

        let extent = extent_of(set);
        let lengthscales = spec
            .initial_lengthscales
            .unwrap_or_else(|| default_lengthscales(&extent, spec.inducing));
        let metric0 = transform(spec.rotation, lengthscales)?;

        // The residual spread sets the scale of everything: the signal variance
        // starts there and the noise is a fraction of it.
        let residuals: Vec<f64> = value_locs
            .iter()
            .zip(&raw_values)
            .map(|(p, v)| v - trend.value(p))
            .collect();
        let mean_residual = residuals.iter().sum::<f64>() / residuals.len() as f64;
        let spread = residuals
            .iter()
            .map(|v| (v - mean_residual).powi(2))
            .sum::<f64>()
            / residuals.len().max(2) as f64;
        // A perfectly constant field — every sample coded +1, say — has zero
        // spread. That is still a legal fit, the surface is just empty, so give
        // the kernel a scale rather than dividing by zero.
        let signal_variance = if spread > 0.0 { spread } else { 1.0 };

        let observations = assemble(set, &trend, &residuals, &metric0, spec);

        let z = place_inducing(&observations.at, spec.inducing, &metric0, spec.seed);
        if z.len() < 2 {
            return Err(ModelError::InsufficientData(
                "the constraints collapse onto fewer than two distinct locations".into(),
            ));
        }

        // θ = [ln σf², ln ℓ₁, ln ℓ₂, ln ℓ₃, ln σ²]. Log space because each is a
        // positive scale, and because a multiplicative step is the natural one
        // for a range.
        let theta0 = [
            signal_variance.ln(),
            lengthscales[0].ln(),
            lengthscales[1].ln(),
            lengthscales[2].ln(),
            (signal_variance * spec.initial_noise).ln(),
        ];

        let fitter = Fitter {
            model: spec.model,
            rotation: spec.rotation,
            z: &z,
        };

        let (theta, status, iterations, gradient_norm, hyper_observations) =
            if spec.learn_hyperparameters {
                let search = subsample(&observations, spec.hyper_sample);
                let seen = search.len();
                let (theta, status, iterations, gnorm) = fitter.maximize(&search, &theta0, spec)?;
                (theta, status, iterations, gnorm, seen)
            } else {
                (theta0, Convergence::NotAttempted, 0, f64::NAN, 0)
            };

        // The reported bound and the posterior itself are always over the whole
        // observation set, whatever the search was allowed to see.
        let (elbo, _, factors) = fitter.elbo_and_gradient(&observations, &theta, false)?;
        let (alpha, var_form) = fitter.posterior(&factors)?;

        let metric = transform(
            spec.rotation,
            [theta[1].exp(), theta[2].exp(), theta[3].exp()],
        )?;
        let report = FitReport {
            status,
            iterations,
            elbo,
            gradient_norm,
            inducing: z.len(),
            observations: observations.len(),
            hyper_observations,
            signal_variance: theta[0].exp(),
            noise_variance: theta[4].exp(),
            lengthscales: [theta[1].exp(), theta[2].exp(), theta[3].exp()],
            jitter: factors.jitter,
        };

        Ok(Self {
            model: spec.model,
            metric,
            signal_variance: theta[0].exp(),
            trend,
            z,
            alpha,
            var_form,
            report,
        })
    }

    /// Posterior mean of the field at a world-space point. `O(m)`.
    pub fn value(&self, p: &[f64; 3]) -> f64 {
        let k = self.k_star(p);
        self.trend.value(p) + k.dot(&self.alpha)
    }

    /// Posterior variance of the *latent field* at a world-space point, in the
    /// field's units squared. `O(m²)`.
    ///
    /// This is the variance of `f`, not of a future observation of it — the
    /// noise term is deliberately left out, because what the isosurface is
    /// drawn through is the field, not a measurement of it. It is largest where
    /// the data is furthest away, which is exactly the "how much of this shell
    /// did we actually drill" question.
    pub fn variance(&self, p: &[f64; 3]) -> f64 {
        let k = self.k_star(p);
        let quad = k.dot(&(&self.var_form * &k));
        // Close to the data those two are nearly equal; clamp rather than hand
        // back a negative variance.
        (self.signal_variance - quad).max(0.0)
    }

    pub fn report(&self) -> &FitReport {
        &self.report
    }

    pub fn inducing_points(&self) -> &[[f64; 3]] {
        &self.z
    }

    /// `k(x*, Z)` — prior covariance between a query point and the inducing
    /// values.
    fn k_star(&self, p: &[f64; 3]) -> DVector<f64> {
        DVector::from_iterator(
            self.z.len(),
            self.z.iter().map(|z| {
                let r = reduced(&self.metric, z, p).norm();
                self.signal_variance * rho(self.model, r)
            }),
        )
    }
}

/// Everything the bound needs that does not change between evaluations.
struct Fitter<'a> {
    model: Model,
    rotation: [f64; 3],
    z: &'a [[f64; 3]],
}

/// The factorized bound at one set of hyperparameters — the pieces prediction
/// and the gradient both need.
struct Factors {
    jitter: f64,
    luu: DMatrix<f64>,
    b_inv: DMatrix<f64>,
    /// `Lb⁻ᵀ Lb⁻¹ Ãỹ`.
    v: DVector<f64>,
}

/// Cached kernel quantities for one (inducing, observation) pair. Every
/// derivative of the bound is assembled from these, and they are cheap enough
/// to recompute per chunk rather than hold `n × m` of them.
struct Pair {
    /// Reduced separation `u = D(z − x)`.
    u: Vector3<f64>,
    r: f64,
    /// The covariance entry per unit signal variance, `Kuf[a, i] / σf²`.
    corr: f64,
    g: f64,
    g_d1: f64,
}

impl Fitter<'_> {
    /// `Kuu`'s Cholesky factor and the jitter it needed.
    fn kuu(&self, metric: &AnisoTransform, sf2: f64) -> Result<(DMatrix<f64>, f64)> {
        let m = self.z.len();
        let mut base = DMatrix::<f64>::zeros(m, m);
        for a in 0..m {
            base[(a, a)] = sf2;
            for b in (a + 1)..m {
                let r = reduced(metric, &self.z[a], &self.z[b]).norm();
                let c = sf2 * rho(self.model, r);
                base[(a, b)] = c;
                base[(b, a)] = c;
            }
        }
        for jitter in JITTER_LADDER {
            let mut trial = base.clone();
            for a in 0..m {
                trial[(a, a)] += sf2 * jitter;
            }
            if let Some(chol) = trial.cholesky() {
                return Ok((chol.l(), jitter));
            }
        }
        Err(ModelError::Singular(
            "the inducing covariance would not factorize even with jitter — the inducing \
             points are degenerate"
                .into(),
        ))
    }

    /// Kernel quantities for every (inducing, observation) pair in a chunk,
    /// laid out observation-major.
    fn pairs(
        &self,
        obs: &Observations,
        rows: &[usize],
        metric: &AnisoTransform,
        d_matrix: &Matrix3<f64>,
        out: &mut Vec<Pair>,
        reduced_dirs: &mut Vec<Vector3<f64>>,
    ) {
        let m = self.z.len();
        out.clear();
        out.reserve(m * rows.len());
        reduced_dirs.clear();
        reduced_dirs.reserve(rows.len());
        for &i in rows {
            let at = obs.at[i];
            let w = if obs.is_derivative(i) {
                let d = obs.direction(i);
                d_matrix * Vector3::new(d[0], d[1], d[2])
            } else {
                Vector3::zeros()
            };
            reduced_dirs.push(w);
            for a in 0..m {
                let u = reduced(metric, &self.z[a], &at);
                let r = u.norm();
                let g = rho_g(self.model, r);
                // A value row is a plain correlation; a derivative row is
                // `∇_x k · d = −g(r)·(u·Dd)`, the covariance differentiated
                // once in its second argument.
                let corr = if obs.is_derivative(i) {
                    -g * u.dot(&w)
                } else {
                    rho(self.model, r)
                };
                out.push(Pair {
                    u,
                    r,
                    corr,
                    g,
                    g_d1: rho_g_d1(self.model, r),
                });
            }
        }
    }

    /// `Ã = Luu⁻¹ Kuf Σ^{-1/2}` for one chunk.
    fn a_tilde(
        &self,
        obs: &Observations,
        rows: &[usize],
        pairs: &[Pair],
        luu: &DMatrix<f64>,
        sf2: f64,
        noise: f64,
    ) -> Result<DMatrix<f64>> {
        let m = self.z.len();
        let mut scaled = DMatrix::<f64>::zeros(m, rows.len());
        for (col, &i) in rows.iter().enumerate() {
            let inv_sigma = 1.0 / (noise * obs.weight[i]).sqrt();
            for a in 0..m {
                scaled[(a, col)] = sf2 * pairs[col * m + a].corr * inv_sigma;
            }
        }
        luu.solve_lower_triangular(&scaled)
            .ok_or_else(|| ModelError::Singular("the inducing covariance is singular".into()))
    }

    /// The collapsed bound and, optionally, its gradient in `θ`.
    ///
    /// Two chunked passes. The first accumulates `B = I + ÃÃᵀ`, `Ãỹ` and the
    /// trace correction; the second recomputes each chunk to contract the
    /// bound's derivative with the kernel's. Nothing of size `n × n` — or even
    /// `m × n` — is ever held.
    ///
    /// The derivatives, all reached through `Luu` and `B`:
    ///
    /// ```text
    /// dL/dKuf[:, i] = Luu⁻ᵀ[ v·ε̃ᵢ − (B⁻¹Ã)ᵢ + Ãᵢ ] / σᵢ
    /// dL/dKuu       = Luu⁻ᵀ[ I − ½vvᵀ − ½B⁻¹ − ½B ] Luu⁻¹
    /// dL/dkffᵢ      = −½/σᵢ²
    /// dL/dσᵢ²       = ½[ ε̃ᵢ² − 1 + ÃᵢᵀB⁻¹Ãᵢ + kffᵢ/σᵢ² − ‖Ãᵢ‖² ] / σᵢ²
    /// ```
    fn elbo_and_gradient(
        &self,
        obs: &Observations,
        theta: &[f64; 5],
        want_gradient: bool,
    ) -> Result<(f64, [f64; 5], Factors)> {
        let m = self.z.len();
        let n = obs.len();
        let sf2 = theta[0].exp();
        let lengthscales = [theta[1].exp(), theta[2].exp(), theta[3].exp()];
        let noise = theta[4].exp();
        let metric = transform(self.rotation, lengthscales)?;
        let d_matrix = metric.matrix();

        let (luu, jitter) = self.kuu(&metric, sf2)?;

        let indices: Vec<usize> = (0..n).collect();
        let mut pairs: Vec<Pair> = Vec::new();
        let mut reduced_dirs: Vec<Vector3<f64>> = Vec::new();

        // ---- pass 1: B, Ãỹ, ‖ỹ‖², log|Σ| and the trace correction.
        let mut b = DMatrix::<f64>::identity(m, m);
        let mut a_y = DVector::<f64>::zeros(m);
        let mut yy = 0.0;
        let mut log_det_sigma = 0.0;
        let mut trace = 0.0;

        for rows in indices.chunks(CHUNK) {
            self.pairs(obs, rows, &metric, &d_matrix, &mut pairs, &mut reduced_dirs);
            let a_tilde = self.a_tilde(obs, rows, &pairs, &luu, sf2, noise)?;
            b += &a_tilde * a_tilde.transpose();
            for (col, &i) in rows.iter().enumerate() {
                let sigma2 = noise * obs.weight[i];
                let y_tilde = obs.y[i] / sigma2.sqrt();
                yy += y_tilde * y_tilde;
                log_det_sigma += sigma2.ln();
                let a_col = a_tilde.column(col);
                a_y.axpy(y_tilde, &a_col, 1.0);
                let kff = sf2 * kff_unit(self.model, obs, i, &reduced_dirs[col]);
                trace += kff / sigma2 - a_col.norm_squared();
            }
        }

        let chol_b = b
            .clone()
            .cholesky()
            .ok_or_else(|| ModelError::Singular("the variational system is singular".into()))?;
        let lb = chol_b.l();
        let c = lb
            .solve_lower_triangular(&a_y)
            .ok_or_else(|| ModelError::Singular("variational back-substitution failed".into()))?;
        let v = lb
            .tr_solve_lower_triangular(&c)
            .ok_or_else(|| ModelError::Singular("variational back-substitution failed".into()))?;

        let log_det_b: f64 = (0..m).map(|a| lb[(a, a)].ln()).sum::<f64>() * 2.0;
        let elbo = -0.5
            * (n as f64 * std::f64::consts::TAU.ln() + log_det_sigma + log_det_b + yy - c.dot(&c))
            - 0.5 * trace;

        let b_inv = chol_b.inverse();
        let factors = Factors {
            jitter,
            luu,
            b_inv,
            v,
        };

        if !want_gradient {
            return Ok((elbo, [0.0; 5], factors));
        }

        let luu = &factors.luu;
        let b_inv = &factors.b_inv;
        let v = &factors.v;

        // ---- pass 2: contract dL/dK with dK/dθ.
        let mut grad = [0.0f64; 5];
        for rows in indices.chunks(CHUNK) {
            self.pairs(obs, rows, &metric, &d_matrix, &mut pairs, &mut reduced_dirs);
            let a_tilde = self.a_tilde(obs, rows, &pairs, luu, sf2, noise)?;
            let cols = rows.len();

            let mut eps = DVector::<f64>::zeros(cols);
            for (col, &i) in rows.iter().enumerate() {
                let sigma2 = noise * obs.weight[i];
                eps[col] = obs.y[i] / sigma2.sqrt() - a_tilde.column(col).dot(v);
            }

            // `B⁻¹Ã` is needed twice — once for the Kuf gradient and once for
            // the per-row quadratic form in the noise gradient — so it is one
            // gemm rather than `n` separate `O(m²)` products.
            let binv_a = b_inv * &a_tilde;
            let t = v * eps.transpose() - &binv_a + &a_tilde;
            let mut g_uf = luu
                .tr_solve_lower_triangular(&t)
                .ok_or_else(|| ModelError::Singular("gradient back-substitution failed".into()))?;
            for (col, &i) in rows.iter().enumerate() {
                let inv_sigma = 1.0 / (noise * obs.weight[i]).sqrt();
                g_uf.column_mut(col).scale_mut(inv_sigma);
            }

            for (col, &i) in rows.iter().enumerate() {
                let sigma2 = noise * obs.weight[i];
                let derivative = obs.is_derivative(i);
                let w = &reduced_dirs[col];
                let kff = sf2 * kff_unit(self.model, obs, i, w);

                for a in 0..m {
                    let p = &pairs[col * m + a];
                    let g = g_uf[(a, col)];
                    // ∂Kuf/∂ln σf² is Kuf itself — everything is linear in it.
                    grad[0] += g * sf2 * p.corr;
                    for (k, slot) in grad[1..4].iter_mut().enumerate() {
                        *slot += g * sf2 * dcorr_dln_l(p, w, k, derivative);
                    }
                }

                // The prior-variance diagonal.
                let dl_dkff = -0.5 / sigma2;
                grad[0] += dl_dkff * kff;
                if derivative {
                    // kff = −ρ″(0)·σf²·‖w‖², and ∂‖w‖²/∂ln ℓₖ = −2wₖ².
                    let norm2 = w.norm_squared().max(f64::MIN_POSITIVE);
                    for (k, slot) in grad[1..4].iter_mut().enumerate() {
                        *slot += dl_dkff * kff * (-2.0 * w[k] * w[k] / norm2);
                    }
                }

                // The noise. σᵢ² = σ²·wᵢ, so ∂σᵢ²/∂ln σ² = σᵢ² and the weight
                // falls straight out of the chain rule.
                let a_col = a_tilde.column(col);
                let quad = a_col.dot(&binv_a.column(col));
                grad[4] +=
                    0.5 * (eps[col] * eps[col] - 1.0 + quad + kff / sigma2 - a_col.norm_squared());
            }
        }

        // dL/dKuu, contracted over the m × m block.
        let mut inner = DMatrix::<f64>::identity(m, m);
        inner -= (v * v.transpose()) * 0.5;
        inner -= b_inv * 0.5;
        inner -= &b * 0.5;
        let half = luu
            .tr_solve_lower_triangular(&inner)
            .ok_or_else(|| ModelError::Singular("gradient back-substitution failed".into()))?;
        let g_uu = luu
            .tr_solve_lower_triangular(&half.transpose())
            .ok_or_else(|| ModelError::Singular("gradient back-substitution failed".into()))?
            .transpose();
        for a in 0..m {
            // The diagonal carries the jitter, which scales with σf² and not
            // with the lengthscales.
            grad[0] += g_uu[(a, a)] * sf2 * (1.0 + jitter);
            for b_idx in (a + 1)..m {
                let u = reduced(&metric, &self.z[a], &self.z[b_idx]);
                let r = u.norm();
                let g = rho_g(self.model, r);
                let entry = g_uu[(a, b_idx)] + g_uu[(b_idx, a)];
                grad[0] += entry * sf2 * rho(self.model, r);
                for (k, slot) in grad[1..4].iter_mut().enumerate() {
                    // ∂ρ/∂ln ℓₖ = −g(r)·uₖ².
                    *slot += entry * sf2 * (-g * u[k] * u[k]);
                }
            }
        }

        Ok((elbo, grad, factors))
    }

    /// The two `m`-sized objects prediction needs.
    fn posterior(&self, factors: &Factors) -> Result<(DVector<f64>, DMatrix<f64>)> {
        let m = self.z.len();
        let alpha = factors
            .luu
            .tr_solve_lower_triangular(&factors.v)
            .ok_or_else(|| ModelError::Singular("posterior back-substitution failed".into()))?;
        let eye = DMatrix::<f64>::identity(m, m);
        let luu_inv = factors
            .luu
            .solve_lower_triangular(&eye)
            .ok_or_else(|| ModelError::Singular("the inducing covariance is singular".into()))?;
        let kuu_inv = luu_inv.tr_mul(&luu_inv);
        let var_form = kuu_inv - luu_inv.tr_mul(&factors.b_inv) * &luu_inv;
        Ok((alpha, var_form))
    }

    /// Maximize the bound by L-BFGS with an Armijo backtracking line search.
    ///
    /// Projected: every parameter is clamped to a band around where it started,
    /// which is what keeps a poorly conditioned dataset from walking the
    /// lengthscale off to infinity instead of reporting that it could not do
    /// better.
    fn maximize(
        &self,
        obs: &Observations,
        theta0: &[f64; 5],
        spec: &SvgpSpec,
    ) -> Result<([f64; 5], Convergence, usize, f64)> {
        let bounds = parameter_bounds(theta0);
        let mut theta = clamp(*theta0, &bounds);
        let (elbo0, grad0, _) = self.elbo_and_gradient(obs, &theta, true)?;
        // The optimizer minimizes, so it carries −L throughout.
        let mut f = -elbo0;
        let mut grad = grad0;
        let mut g: Vec<f64> = grad0.iter().map(|v| -v).collect();

        let mut history: VecDeque<(Vec<f64>, Vec<f64>)> = VecDeque::new();
        let mut status = Convergence::MaxIterations;
        let mut iterations = 0;

        for _ in 0..spec.max_iterations {
            if infinity_norm(&g) < spec.tolerance {
                status = Convergence::Converged;
                break;
            }
            let mut direction = lbfgs_direction(&g, &history);
            // A non-descent direction means the history has gone stale; drop it
            // rather than trust it.
            if dot(&direction, &g) >= 0.0 {
                history.clear();
                direction = g.iter().map(|v| -v).collect();
            }
            let reach = infinity_norm(&direction);
            if reach > MAX_LOG_STEP {
                let scale = MAX_LOG_STEP / reach;
                for v in direction.iter_mut() {
                    *v *= scale;
                }
            }

            let mut step = 1.0;
            let mut moved = false;
            for _ in 0..LINE_SEARCH_STEPS {
                let mut trial = theta;
                for (k, slot) in trial.iter_mut().enumerate() {
                    *slot += step * direction[k];
                }
                let trial = clamp(trial, &bounds);
                // Armijo against the step actually taken, not the one proposed:
                // the projection onto the parameter box can shorten it, or turn
                // it into no step at all, and testing the proposal would accept
                // moves the fit never made.
                let s: Vec<f64> = (0..5).map(|k| trial[k] - theta[k]).collect();
                let slope = dot(&s, &g);
                // A step that lands where the Cholesky fails is not an error,
                // it is a step that was too long.
                if slope < 0.0
                    && let Ok((elbo, gradient, _)) = self.elbo_and_gradient(obs, &trial, true)
                {
                    let ft = -elbo;
                    if ft.is_finite() && ft <= f + 1e-4 * slope {
                        let gt: Vec<f64> = gradient.iter().map(|v| -v).collect();
                        let yv: Vec<f64> = (0..5).map(|k| gt[k] - g[k]).collect();
                        if dot(&s, &yv) > 1e-12 {
                            history.push_back((s, yv));
                            if history.len() > LBFGS_MEMORY {
                                history.pop_front();
                            }
                        }
                        let improvement = f - ft;
                        theta = trial;
                        f = ft;
                        g = gt;
                        grad = gradient;
                        moved = true;
                        iterations += 1;
                        if improvement.abs() <= spec.tolerance * (1.0 + f.abs()) {
                            status = Convergence::Converged;
                        }
                        break;
                    }
                }
                step *= 0.5;
            }
            if !moved {
                status = Convergence::LineSearchStalled;
                break;
            }
            if status == Convergence::Converged {
                break;
            }
        }

        Ok((theta, status, iterations, infinity_norm(&grad)))
    }
}

/// Prior variance of row `i`, per unit signal variance.
fn kff_unit(model: Model, obs: &Observations, i: usize, w: &Vector3<f64>) -> f64 {
    if obs.is_derivative(i) {
        derivative_prior(model, w)
    } else {
        1.0
    }
}

/// `∂(Kuf entry / σf²)/∂ln ℓₖ`.
///
/// For a value row that is `−g(r)·uₖ²`. For a derivative row the direction
/// rescales too, which is what the second term carries.
fn dcorr_dln_l(p: &Pair, w: &Vector3<f64>, k: usize, derivative: bool) -> f64 {
    if !derivative {
        return -p.g * p.u[k] * p.u[k];
    }
    if p.r <= ZERO_LAG {
        // `u → 0` kills both terms faster than `g′` can grow.
        return 0.0;
    }
    let uw = p.u.dot(w);
    p.g_d1 * p.u[k] * p.u[k] * uw / p.r + 2.0 * p.g * p.u[k] * w[k]
}

/// Two-loop recursion. Returns `−H·g`, the quasi-Newton descent direction.
fn lbfgs_direction(g: &[f64], history: &VecDeque<(Vec<f64>, Vec<f64>)>) -> Vec<f64> {
    let mut q: Vec<f64> = g.to_vec();
    let mut alphas = Vec::with_capacity(history.len());
    for (s, y) in history.iter().rev() {
        let rho_i = 1.0 / dot(s, y);
        let alpha = rho_i * dot(s, &q);
        for (k, slot) in q.iter_mut().enumerate() {
            *slot -= alpha * y[k];
        }
        alphas.push((rho_i, alpha));
    }
    if let Some((s, y)) = history.back() {
        let yy = dot(y, y);
        if yy > 0.0 {
            let scale = dot(s, y) / yy;
            for v in q.iter_mut() {
                *v *= scale;
            }
        }
    }
    // `alphas` was filled walking the history backwards, so reversing it lines
    // the pairs back up with `history` in order.
    for ((rho_i, alpha), (s, y)) in alphas.into_iter().rev().zip(history.iter()) {
        let beta = rho_i * dot(y, &q);
        for (k, slot) in q.iter_mut().enumerate() {
            *slot += (alpha - beta) * s[k];
        }
    }
    q.iter().map(|v| -v).collect()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn infinity_norm(v: &[f64]) -> f64 {
    v.iter().fold(0.0f64, |acc, x| acc.max(x.abs()))
}

// -------------------------------------------------------------- assembly

/// `D(a − b)` — the reduced separation vector.
fn reduced(metric: &AnisoTransform, a: &[f64; 3], b: &[f64; 3]) -> Vector3<f64> {
    let d = metric.apply(&[a[0] - b[0], a[1] - b[1], a[2] - b[2]]);
    Vector3::new(d[0], d[1], d[2])
}

fn transform(rotation: [f64; 3], lengthscales: [f64; 3]) -> Result<AnisoTransform> {
    AnisoTransform::new(&Angles {
        azimuth: rotation[0],
        dip: rotation[1],
        rake: rotation[2],
        major: lengthscales[0],
        semi: lengthscales[1],
        minor: lengthscales[2],
    })
}

fn extent_of(set: &ConstraintSet) -> [f64; 3] {
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    for p in set
        .values
        .iter()
        .map(|v| v.at)
        .chain(set.derivatives.iter().map(|d| d.at))
    {
        for k in 0..3 {
            min[k] = min[k].min(p[k]);
            max[k] = max[k].max(p[k]);
        }
    }
    let mut extent = [0.0; 3];
    for k in 0..3 {
        extent[k] = (max[k] - min[k]).max(0.0);
    }
    extent
}

/// Starting ranges tied to the resolution the inducing set can actually carry.
///
/// This is the initialization #320 asks to be got right, and the one place the
/// engine is genuinely easy to break. `m` inducing points spread over the data
/// sit about `extent/∛m` apart on each axis; a range shorter than that
/// describes structure the sparse approximation cannot represent, so the trace
/// correction `−½tr(Σ⁻¹(Kff − Qff))` — which charges the fit for exactly that
/// unexplained variance — dominates the bound from the first step. The optimizer
/// then buys its improvement the cheap way, by shrinking the signal variance
/// until the data is all noise, and reports a perfectly converged fit to an
/// empty field.
///
/// Twice the inducing spacing starts on the right side of that. A degenerate
/// axis — a single bench, a 2D grid — borrows the largest.
fn default_lengthscales(extent: &[f64; 3], inducing: usize) -> [f64; 3] {
    let per_axis = (inducing.max(1) as f64).cbrt().max(1.0);
    let largest = extent.iter().cloned().fold(0.0f64, f64::max);
    let fallback = if largest > 0.0 {
        2.0 * largest / per_axis
    } else {
        1.0
    };
    let mut out = [0.0; 3];
    for k in 0..3 {
        out[k] = if extent[k] > 0.0 {
            2.0 * extent[k] / per_axis
        } else {
            fallback
        };
    }
    out
}

/// How far each parameter may travel from where it started, in decades.
///
/// The ranges are given the most room, because a real deposit's continuity
/// genuinely spans orders of magnitude and the starting guess is only a
/// heuristic. The noise gets the least room *upwards*: `σ²` starts at a tenth of
/// the data's variance and may reach it but not exceed it, because a noise
/// larger than the whole signal is not a fit, it is the optimizer declining to
/// fit — the degenerate optimum [`default_lengthscales`] describes, reached from
/// the other side.
fn parameter_bounds(theta0: &[f64; 5]) -> [(f64, f64); 5] {
    let decades: [(f64, f64); 5] = [
        (2.0, 2.0),
        (PARAMETER_DECADES, PARAMETER_DECADES),
        (PARAMETER_DECADES, PARAMETER_DECADES),
        (PARAMETER_DECADES, PARAMETER_DECADES),
        (4.0, 1.0),
    ];
    let mut out = [(0.0, 0.0); 5];
    for (k, slot) in out.iter_mut().enumerate() {
        let (down, up) = decades[k];
        *slot = (
            theta0[k] - down * std::f64::consts::LN_10,
            theta0[k] + up * std::f64::consts::LN_10,
        );
    }
    out
}

fn clamp(mut theta: [f64; 5], bounds: &[(f64, f64); 5]) -> [f64; 5] {
    for (k, slot) in theta.iter_mut().enumerate() {
        *slot = slot.clamp(bounds[k].0, bounds[k].1);
    }
    theta
}

/// Turn the constraint set into rows the bound can walk: detrended, and with
/// the noise weight each kind of observation earns.
///
/// A derivative residual carries units of field per meter, so it cannot share a
/// variance with a value residual as it stands. The row's own prior variance is
/// the conversion — dividing by it puts a structural row on the same footing as
/// a sample, after which `gradient_nugget` reads as "how much slacker than an
/// assay" and nothing else. That conversion is fixed at the *initial*
/// lengthscales and held there, so the learned noise stays a single scalar and
/// `∂σᵢ²/∂θ` does not pick up a kernel term.
fn assemble(
    set: &ConstraintSet,
    trend: &Trend,
    residuals: &[f64],
    metric: &AnisoTransform,
    spec: &SvgpSpec,
) -> Observations {
    let capacity = set.values.len() + set.derivatives.len();
    let mut at = Vec::with_capacity(capacity);
    let mut y = Vec::with_capacity(capacity);
    let mut weight = Vec::with_capacity(capacity);
    let mut sample = Vec::with_capacity(capacity);

    for (v, residual) in set.values.iter().zip(residuals) {
        at.push(v.at);
        y.push(*residual);
        match v.kind {
            ValueKind::Sample => {
                weight.push(1.0);
                sample.push(true);
            }
            ValueKind::Boundary => {
                weight.push(spec.boundary_tolerance.max(BOUNDARY_WEIGHT_FLOOR));
                sample.push(false);
            }
        }
    }

    let values = at.len();
    let d_matrix = metric.matrix();
    let mut dirs = Vec::with_capacity(set.derivatives.len());
    for d in &set.derivatives {
        let w = d_matrix * Vector3::new(d.direction[0], d.direction[1], d.direction[2]);
        let prior = derivative_prior(spec.model, &w).max(f64::MIN_POSITIVE);
        at.push(d.at);
        dirs.push(d.direction);
        y.push(d.value - trend.slope(&d.direction));
        weight.push((1.0 + spec.gradient_nugget) * prior);
        sample.push(false);
    }

    Observations {
        at,
        dirs,
        y,
        weight,
        sample,
        values,
    }
}

/// The rows the hyperparameter search sees: every boundary pick and every
/// structural row, with the ordinary samples strided down to fit.
///
/// The same policy as the dense engines' decimation, for the same reason — the
/// rows worth spending a scarce budget on are the ones there are fewest of.
/// Unlike those engines this is *only* the search; the fit itself sees
/// everything.
fn subsample(obs: &Observations, max: usize) -> Observations {
    if obs.values <= max {
        return obs.clone();
    }
    let samples: Vec<usize> = (0..obs.values).filter(|&i| obs.sample[i]).collect();
    let picks: Vec<usize> = (0..obs.values).filter(|&i| !obs.sample[i]).collect();
    let budget = max.saturating_sub(picks.len()).max(2);
    let kept = decimate(&samples, budget);

    let mut at = Vec::new();
    let mut y = Vec::new();
    let mut weight = Vec::new();
    let mut sample = Vec::new();
    for &i in kept.iter().chain(&picks) {
        at.push(obs.at[i]);
        y.push(obs.y[i]);
        weight.push(obs.weight[i]);
        sample.push(obs.sample[i]);
    }
    let values = at.len();
    for i in obs.values..obs.len() {
        at.push(obs.at[i]);
        y.push(obs.y[i]);
        weight.push(obs.weight[i]);
        sample.push(obs.sample[i]);
    }
    Observations {
        at,
        dirs: obs.dirs.clone(),
        y,
        weight,
        sample,
        values,
    }
}

/// Place `k` inducing points by seeded `k`-means++ over the constraint
/// locations, in the *reduced* metric so a flat deposit does not spend all its
/// resolution on the vertical.
///
/// Deterministic given the seed: `StdRng` is ChaCha12, reproducible across
/// platforms and thread counts, and Lloyd's iteration has no other source of
/// randomness.
fn place_inducing(
    locations: &[[f64; 3]],
    k: usize,
    metric: &AnisoTransform,
    seed: u64,
) -> Vec<[f64; 3]> {
    let pool = decimate(locations, PLACEMENT_SAMPLE);
    if pool.len() <= k {
        return dedup(&pool);
    }
    let reduced_pool: Vec<Vector3<f64>> = pool
        .iter()
        .map(|p| {
            let r = metric.apply(p);
            Vector3::new(r[0], r[1], r[2])
        })
        .collect();

    let mut rng = StdRng::seed_from_u64(seed);
    let first = rng.gen_range(0..pool.len());
    let mut chosen: Vec<usize> = vec![first];
    let mut best: Vec<f64> = reduced_pool
        .iter()
        .map(|p| (p - reduced_pool[first]).norm_squared())
        .collect();

    while chosen.len() < k {
        let total: f64 = best.iter().sum();
        if total <= 0.0 || !total.is_finite() {
            // Every remaining point coincides with a chosen one; there is
            // nothing left to spread over.
            break;
        }
        let target = rng.gen_range(0.0..total);
        let mut acc = 0.0;
        let mut pick = pool.len() - 1;
        for (i, d) in best.iter().enumerate() {
            acc += d;
            if acc >= target {
                pick = i;
                break;
            }
        }
        chosen.push(pick);
        for (i, p) in reduced_pool.iter().enumerate() {
            best[i] = best[i].min((p - reduced_pool[pick]).norm_squared());
        }
    }

    let mut centroids: Vec<Vector3<f64>> = chosen.iter().map(|&i| reduced_pool[i]).collect();
    let mut world: Vec<[f64; 3]> = chosen.iter().map(|&i| pool[i]).collect();
    for _ in 0..PLACEMENT_ITERATIONS {
        let mut sums = vec![Vector3::zeros(); centroids.len()];
        let mut world_sums = vec![[0.0f64; 3]; centroids.len()];
        let mut counts = vec![0usize; centroids.len()];
        for (i, p) in reduced_pool.iter().enumerate() {
            let mut nearest = 0;
            let mut best_d = f64::INFINITY;
            for (c, q) in centroids.iter().enumerate() {
                let d = (p - q).norm_squared();
                if d < best_d {
                    best_d = d;
                    nearest = c;
                }
            }
            sums[nearest] += p;
            counts[nearest] += 1;
            for axis in 0..3 {
                world_sums[nearest][axis] += pool[i][axis];
            }
        }
        let mut moved = false;
        for c in 0..centroids.len() {
            if counts[c] == 0 {
                continue;
            }
            let inv = 1.0 / counts[c] as f64;
            let next = sums[c] * inv;
            if (next - centroids[c]).norm_squared() > 0.0 {
                moved = true;
            }
            centroids[c] = next;
            world[c] = [
                world_sums[c][0] * inv,
                world_sums[c][1] * inv,
                world_sums[c][2] * inv,
            ];
        }
        if !moved {
            break;
        }
    }
    dedup(&world)
}

/// Drop points that coincide to within a relative tolerance — two identical
/// inducing points make `Kuu` singular however much jitter is added.
fn dedup(points: &[[f64; 3]]) -> Vec<[f64; 3]> {
    let mut out: Vec<[f64; 3]> = Vec::with_capacity(points.len());
    for p in points {
        if !p.iter().all(|v| v.is_finite()) {
            continue;
        }
        let duplicate = out.iter().any(|q| {
            (0..3).all(|k| (q[k] - p[k]).abs() <= 1e-9 * (1.0 + q[k].abs().max(p[k].abs())))
        });
        if !duplicate {
            out.push(*p);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraint::PlaneEncoding;
    use crate::orientation::Plane;

    /// Indicator codings of a ball of `radius` on a 4 m lattice over ±16 m,
    /// with the samples nearest the contact dropped so the interpolant, not the
    /// sampling, decides where the boundary lands. The same fixture the dense
    /// engines are tested against, so the three are directly comparable.
    fn shell_set(radius: f64) -> ConstraintSet {
        let mut set = ConstraintSet::new();
        for i in -4..=4 {
            for j in -4..=4 {
                for k in -4..=4 {
                    let p = [i as f64 * 4.0, j as f64 * 4.0, k as f64 * 4.0];
                    let d = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
                    if (d - radius).abs() > 1.5 {
                        set.push_sample(p, if d < radius { 1.0 } else { -1.0 });
                    }
                }
            }
        }
        set
    }

    fn spec(inducing: usize) -> SvgpSpec {
        SvgpSpec {
            inducing,
            max_iterations: 60,
            ..Default::default()
        }
    }

    /// Rebuild the pieces `Svgp::fit` assembles, so a test can drive the bound
    /// directly instead of through the whole fit.
    fn bench(set: &ConstraintSet, spec: &SvgpSpec) -> (Observations, Vec<[f64; 3]>, [f64; 5]) {
        let locs: Vec<[f64; 3]> = set.values.iter().map(|v| v.at).collect();
        let raw: Vec<f64> = set.values.iter().map(|v| v.value).collect();
        let trend = Trend::fit(&locs, &raw, spec.drift_degree);
        let lengthscales = spec
            .initial_lengthscales
            .unwrap_or_else(|| default_lengthscales(&extent_of(set), spec.inducing));
        let metric = transform(spec.rotation, lengthscales).unwrap();
        let residuals: Vec<f64> = locs
            .iter()
            .zip(&raw)
            .map(|(p, v)| v - trend.value(p))
            .collect();
        let mean = residuals.iter().sum::<f64>() / residuals.len() as f64;
        let sf2 = residuals.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
            / residuals.len().max(2) as f64;
        let obs = assemble(set, &trend, &residuals, &metric, spec);
        let z = place_inducing(&obs.at, spec.inducing, &metric, spec.seed);
        let theta = [
            sf2.ln(),
            lengthscales[0].ln(),
            lengthscales[1].ln(),
            lengthscales[2].ln(),
            (sf2 * spec.initial_noise).ln(),
        ];
        (obs, z, theta)
    }

    /// The load-bearing test of this module. Every gradient in
    /// [`Fitter::elbo_and_gradient`] is hand-derived, and a sign error in any
    /// of them produces an optimizer that quietly walks the wrong way rather
    /// than anything that looks like a failure. Central differences on the
    /// bound itself are the only honest check.
    ///
    /// The set carries structural readings deliberately: the derivative rows
    /// exercise the `∂Kuf/∂ln ℓ` term that a value-only fit never reaches.
    #[test]
    fn analytic_gradients_match_finite_differences() {
        let mut set = shell_set(10.0);
        for i in -1..=1 {
            set.push_plane(
                [i as f64 * 6.0, 0.0, 0.0],
                Plane {
                    dip: 30.0,
                    dip_direction: 120.0,
                },
                PlaneEncoding::Both,
                0.02,
            )
            .unwrap();
        }
        let spec = spec(24);
        let (obs, z, theta) = bench(&set, &spec);
        let fitter = Fitter {
            model: spec.model,
            rotation: spec.rotation,
            z: &z,
        };

        let (_, analytic, _) = fitter.elbo_and_gradient(&obs, &theta, true).unwrap();

        let h = 1e-6;
        for k in 0..5 {
            let mut up = theta;
            let mut down = theta;
            up[k] += h;
            down[k] -= h;
            let (lu, _, _) = fitter.elbo_and_gradient(&obs, &up, false).unwrap();
            let (ld, _, _) = fitter.elbo_and_gradient(&obs, &down, false).unwrap();
            let numeric = (lu - ld) / (2.0 * h);
            let scale = numeric.abs().max(analytic[k].abs()).max(1.0);
            assert!(
                (numeric - analytic[k]).abs() / scale < 1e-4,
                "θ[{k}]: analytic {} vs finite-difference {numeric}",
                analytic[k]
            );
        }
    }

    /// Same check on a value-only set, so a bug that happens to cancel between
    /// the value and derivative paths cannot hide.
    #[test]
    fn analytic_gradients_match_finite_differences_without_derivatives() {
        let set = shell_set(10.0);
        let spec = spec(16);
        let (obs, z, theta) = bench(&set, &spec);
        let fitter = Fitter {
            model: spec.model,
            rotation: spec.rotation,
            z: &z,
        };
        let (_, analytic, _) = fitter.elbo_and_gradient(&obs, &theta, true).unwrap();
        let h = 1e-6;
        for k in 0..5 {
            let mut up = theta;
            let mut down = theta;
            up[k] += h;
            down[k] -= h;
            let (lu, _, _) = fitter.elbo_and_gradient(&obs, &up, false).unwrap();
            let (ld, _, _) = fitter.elbo_and_gradient(&obs, &down, false).unwrap();
            let numeric = (lu - ld) / (2.0 * h);
            let scale = numeric.abs().max(analytic[k].abs()).max(1.0);
            assert!(
                (numeric - analytic[k]).abs() / scale < 1e-4,
                "θ[{k}]: analytic {} vs finite-difference {numeric}",
                analytic[k]
            );
        }
    }

    /// The optimizer has to actually raise the bound it is handed, and say so.
    #[test]
    fn the_fit_raises_the_bound_and_reports_how_it_ended() {
        let set = shell_set(10.0);
        let spec = spec(48);
        let (obs, z, theta0) = bench(&set, &spec);
        let fitter = Fitter {
            model: spec.model,
            rotation: spec.rotation,
            z: &z,
        };
        let (before, _, _) = fitter.elbo_and_gradient(&obs, &theta0, false).unwrap();

        let gp = Svgp::fit(&set, &spec).unwrap();
        assert!(
            gp.report().elbo > before,
            "the fit lowered the bound: {} -> {}",
            before,
            gp.report().elbo
        );
        assert!(gp.report().iterations > 0);
        assert!(
            gp.report().status.is_settled(),
            "did not settle: {:?}",
            gp.report().status
        );
    }

    /// Radius of the ball with the same volume as the modeled shell.
    fn effective_radius(set: &ConstraintSet, spec: &SvgpSpec) -> f64 {
        let grid = GridSpec {
            origin: [-20.0; 3],
            spacing: [2.0; 3],
            counts: [21; 3],
        };
        let surface = crate::implicit::build_surface_constrained(
            set,
            &Engine::SparseGp {
                spec: spec.clone(),
                emit_variance: false,
            },
            &grid,
            0.0,
        )
        .unwrap();
        (surface.mesh.enclosed_volume() * 3.0 / (4.0 * std::f64::consts::PI)).cbrt()
    }

    use crate::implicit::{Engine, GridSpec};

    #[test]
    fn recovers_a_spherical_domain() {
        let r = effective_radius(&shell_set(10.0), &spec(64));
        assert!(
            (8.0..11.32).contains(&r),
            "contact at r = {r} contradicts the samples"
        );
        assert!((r - 10.0).abs() < 2.0, "contact at r = {r}, want 10");
    }

    /// The issue's acceptance criterion: on a dataset small enough for both,
    /// the sparse engine and the dual-kriging engine must agree.
    #[test]
    fn agrees_with_dual_kriging_on_a_dataset_small_enough_for_both() {
        use variogram::{Model as VgModel, Variogram};

        let set = shell_set(10.0);
        let grid = GridSpec {
            origin: [-20.0; 3],
            spacing: [2.0; 3],
            counts: [21; 3],
        };
        let volume = |engine: &Engine| {
            crate::implicit::build_surface_constrained(&set, engine, &grid, 0.0)
                .unwrap()
                .mesh
                .enclosed_volume()
        };
        let dense = volume(&Engine::DualKriging {
            variogram: Variogram::single(VgModel::Spherical, 1.0, 30.0),
            spec: crate::hermite::HermiteSpec::default(),
        });
        let sparse = volume(&Engine::SparseGp {
            spec: spec(64),
            emit_variance: false,
        });
        let dense_r = (dense * 3.0 / (4.0 * std::f64::consts::PI)).cbrt();
        let sparse_r = (sparse * 3.0 / (4.0 * std::f64::consts::PI)).cbrt();
        assert!(
            (dense_r - sparse_r).abs() < 1.5,
            "engines disagree by {} m of radius (dense {dense_r}, sparse {sparse_r})",
            (dense_r - sparse_r).abs()
        );
    }

    /// A full-data sparse fit and one on a decimated subset have to land in the
    /// same place — otherwise the extra data the engine exists to use is
    /// changing the answer for the wrong reasons.
    #[test]
    fn the_full_fit_agrees_with_one_on_a_decimated_subset() {
        let full = shell_set(10.0);
        let thinned = ConstraintSet {
            values: decimate(&full.values, full.values.len() / 3),
            derivatives: Vec::new(),
        };
        let a = effective_radius(&full, &spec(64));
        let b = effective_radius(&thinned, &spec(64));
        assert!(
            (a - b).abs() < 1.5,
            "full fit gives r = {a}, decimated gives r = {b}"
        );
    }

    /// Non-reproducible resource models are not acceptable: the same seed has
    /// to give bit-identical output.
    #[test]
    fn the_same_seed_reproduces_the_field_exactly() {
        let set = shell_set(10.0);
        let probes = [[0.0, 0.0, 0.0], [7.0, 3.0, -2.0], [15.0, 15.0, 15.0]];
        let a = Svgp::fit(&set, &spec(48)).unwrap();
        let b = Svgp::fit(&set, &spec(48)).unwrap();
        for p in &probes {
            assert_eq!(a.value(p), b.value(p), "not reproducible at {p:?}");
            assert_eq!(a.variance(p), b.variance(p));
        }
        assert_eq!(a.report().elbo, b.report().elbo);

        // A different seed moves the inducing points, so it may not reproduce
        // bit-for-bit — but it must still model the same body.
        let other = Svgp::fit(
            &set,
            &SvgpSpec {
                seed: 7,
                ..spec(48)
            },
        )
        .unwrap();
        for p in &probes {
            assert!(
                (a.value(p) - other.value(p)).abs() < 0.35,
                "seed changed the field at {p:?}: {} vs {}",
                a.value(p),
                other.value(p)
            );
        }
    }

    /// The posterior variance has to be small where the data is and rise where
    /// it is not — that is the whole claim #319 rests on.
    #[test]
    fn the_posterior_variance_rises_away_from_the_data() {
        let set = shell_set(10.0);
        let gp = Svgp::fit(&set, &spec(64)).unwrap();
        // A sampled node, versus a point far outside the drilled extent.
        let inside = gp.variance(&[4.0, 4.0, 4.0]);
        let outside = gp.variance(&[120.0, 120.0, 120.0]);
        assert!(inside >= 0.0 && outside >= 0.0);
        assert!(
            outside > inside * 2.0,
            "variance did not rise away from the data: {inside} inside, {outside} outside"
        );
        // Far enough out, the posterior forgets the data and returns the prior.
        assert!(
            (outside - gp.report().signal_variance).abs() < 0.05 * gp.report().signal_variance,
            "the far-field variance {outside} is not the prior {}",
            gp.report().signal_variance
        );
    }

    /// Same rule and the same by-name refusal as the dual-kriging engine.
    #[test]
    fn a_non_differentiable_kernel_is_refused_by_name_for_structural_data() {
        let mut set = shell_set(10.0);
        set.push_lineation(
            [0.0, 0.0, 0.0],
            crate::orientation::Lineation {
                plunge: 10.0,
                trend: 40.0,
            },
        )
        .unwrap();
        let err = Svgp::fit(
            &set,
            &SvgpSpec {
                model: Model::Exponential,
                ..spec(24)
            },
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("Exponential"), "unhelpful message: {err}");
        assert!(err.contains("nowhere"), "unhelpful message: {err}");

        // The same kernel is fine with no structural data to constrain.
        assert!(
            Svgp::fit(
                &shell_set(10.0),
                &SvgpSpec {
                    model: Model::Exponential,
                    ..spec(24)
                }
            )
            .is_ok()
        );
    }

    /// The power model is unbounded, so it is not a covariance at all.
    #[test]
    fn the_power_model_is_refused() {
        let err = Svgp::fit(
            &shell_set(10.0),
            &SvgpSpec {
                model: Model::Power { exponent: 1.5 },
                ..spec(16)
            },
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("stationary covariance"), "{err}");
    }

    /// Central difference of the posterior mean along `direction`.
    fn slope_at(gp: &Svgp, at: [f64; 3], direction: [f64; 3], h: f64) -> f64 {
        let step = |s: f64| {
            [
                at[0] + s * direction[0],
                at[1] + s * direction[1],
                at[2] + s * direction[2],
            ]
        };
        (gp.value(&step(h)) - gp.value(&step(-h))) / (2.0 * h)
    }

    /// A structural reading constrains the *gradient*, so that is what the test
    /// measures. The value samples are symmetric east–west, so the
    /// unconstrained fit has no easterly slope at the origin at all; a
    /// derivative row asking for one has to produce it.
    #[test]
    fn a_structural_reading_steers_the_gradient() {
        let mut base = ConstraintSet::new();
        for i in -2..=2 {
            for j in -2..=2 {
                let (x, y) = (i as f64 * 8.0, j as f64 * 8.0);
                base.push_sample([x, y, -8.0], 1.0);
                base.push_sample([x, y, 8.0], -1.0);
            }
        }
        let mut steered = base.clone();
        let east = [1.0, 0.0, 0.0];
        for i in -1..=1 {
            steered
                .push_derivative([i as f64 * 8.0, 0.0, 0.0], east, 0.05)
                .unwrap();
        }

        let plain = Svgp::fit(&base, &spec(32)).unwrap();
        let steered = Svgp::fit(&steered, &spec(32)).unwrap();

        let flat_slope = slope_at(&plain, [0.0, 0.0, 0.0], east, 0.5);
        let asked_slope = slope_at(&steered, [0.0, 0.0, 0.0], east, 0.5);
        assert!(
            flat_slope.abs() < 1e-6,
            "the value-only field already slopes east: {flat_slope}"
        );
        // Not just "it moved" — it has to move *towards the reading*, and get a
        // useful part of the way there.
        assert!(
            asked_slope > 0.02,
            "the structural reading asked for a slope of 0.05/m and got {asked_slope}"
        );
    }

    /// A tangent reading is the other half: it says the field does *not* change
    /// along the structure, and has to flatten a field that otherwise would.
    #[test]
    fn a_tangent_reading_flattens_the_field_along_it() {
        // A field that falls to the east on its own.
        let mut base = ConstraintSet::new();
        for i in -3..=3 {
            for j in -2..=2 {
                for k in -1..=1 {
                    let p = [i as f64 * 6.0, j as f64 * 6.0, k as f64 * 6.0];
                    base.push_sample(p, -0.04 * p[0]);
                }
            }
        }
        // A horizontal plane read at every sample: both its tangents are
        // horizontal, so this says the field does not change in the x–y plane.
        // Read densely on purpose — a handful of gradient rows against a
        // hundred consistent assays should *not* win, and a test that demanded
        // they did would be testing the wrong thing.
        let mut flattened = base.clone();
        let locations: Vec<[f64; 3]> = base.values.iter().map(|v| v.at).collect();
        for at in locations {
            flattened
                .push_plane(
                    at,
                    Plane {
                        dip: 0.0,
                        dip_direction: 90.0,
                    },
                    PlaneEncoding::Tangents,
                    1.0,
                )
                .unwrap();
        }
        let east = [1.0, 0.0, 0.0];
        let plain = slope_at(&Svgp::fit(&base, &spec(32)).unwrap(), [0.0; 3], east, 0.5);
        let held = slope_at(
            &Svgp::fit(&flattened, &spec(32)).unwrap(),
            [0.0; 3],
            east,
            0.5,
        );
        assert!(plain < -0.02, "the base field does not fall east: {plain}");
        assert!(
            held.abs() < plain.abs() * 0.6,
            "the tangent reading did not flatten the field: {plain} -> {held}"
        );
    }

    /// Learning off is a legitimate configuration — a plain sparse GP at the
    /// covariance it was handed — and has to report itself as such.
    #[test]
    fn learning_can_be_switched_off() {
        let gp = Svgp::fit(
            &shell_set(10.0),
            &SvgpSpec {
                learn_hyperparameters: false,
                initial_lengthscales: Some([25.0, 25.0, 25.0]),
                ..spec(32)
            },
        )
        .unwrap();
        assert_eq!(gp.report().status, Convergence::NotAttempted);
        assert_eq!(gp.report().iterations, 0);
        assert!((gp.report().lengthscales[0] - 25.0).abs() < 1e-9);
        assert_eq!(gp.report().hyper_observations, 0);
    }

    /// The search may be capped, but the fit never is.
    #[test]
    fn the_search_is_capped_and_the_fit_is_not() {
        let set = shell_set(10.0);
        let n = set.values.len();
        let gp = Svgp::fit(
            &set,
            &SvgpSpec {
                hyper_sample: 100,
                ..spec(32)
            },
        )
        .unwrap();
        assert_eq!(gp.report().observations, n);
        assert!(gp.report().hyper_observations <= 100);
        assert!(gp.report().hyper_observations < n);
    }

    /// Mine-grid coordinates are ~10⁵ with a field of ±1; the reduced metric
    /// and the centered drift are what keep that conditioned.
    #[test]
    fn survives_mine_grid_coordinates() {
        let mut set = ConstraintSet::new();
        for i in -4..=4 {
            for j in -4..=4 {
                for k in -2..=2 {
                    let p = [
                        512_340.0 + i as f64 * 4.0,
                        7_384_120.0 + j as f64 * 4.0,
                        1_240.0 + k as f64 * 4.0,
                    ];
                    let d = ((p[0] - 512_340.0).powi(2)
                        + (p[1] - 7_384_120.0).powi(2)
                        + (p[2] - 1_240.0).powi(2))
                    .sqrt();
                    set.push_sample(p, if d < 10.0 { 1.0 } else { -1.0 });
                }
            }
        }
        let gp = Svgp::fit(&set, &spec(48)).unwrap();
        assert!(gp.value(&[512_340.0, 7_384_120.0, 1_240.0]) > 0.0);
        assert!(gp.value(&[512_340.0 + 16.0, 7_384_120.0 + 16.0, 1_240.0]) < 0.0);
    }

    /// A linear drift is removed before the GP and added back after, so a field
    /// with a trend running clean through it is reproduced outside the data
    /// rather than reverting to the mean.
    #[test]
    fn a_linear_drift_is_carried_through() {
        let mut set = ConstraintSet::new();
        for i in -4..=4 {
            for j in -4..=4 {
                for k in -4..=4 {
                    let p = [i as f64 * 5.0, j as f64 * 5.0, k as f64 * 5.0];
                    set.push_sample(p, 0.05 * p[0]);
                }
            }
        }
        let with_drift = Svgp::fit(
            &set,
            &SvgpSpec {
                drift_degree: 1,
                ..spec(32)
            },
        )
        .unwrap();
        let without = Svgp::fit(&set, &spec(32)).unwrap();

        // Well outside the data the GP alone has to guess; the drift knows.
        let far = [60.0, 0.0, 0.0];
        let truth = 0.05 * far[0];
        assert!(
            (with_drift.value(&far) - truth).abs() < 0.05,
            "the drifted fit extrapolates to {}, want {truth}",
            with_drift.value(&far)
        );
        assert!(
            (with_drift.value(&far) - truth).abs() < (without.value(&far) - truth).abs(),
            "the drift did not help: with {} vs without {}, truth {truth}",
            with_drift.value(&far),
            without.value(&far)
        );
    }

    /// #320's scale criterion: 10⁵ samples fitted without decimation, in time
    /// comparable to the dense engines' capped run.
    ///
    /// Ignored because it is a minute of work in debug and meaningless there
    /// anyway — run it in release:
    /// `cargo test -p modeling --release fits_a_hundred_thousand -- --ignored --nocapture`
    #[test]
    #[ignore = "scale benchmark; run in release"]
    fn fits_a_hundred_thousand_samples_without_decimation() {
        // A ball of radius 40 sampled on a 100 × 100 × 10 lattice: 100k
        // composites, the size of a real drillhole database.
        let mut set = ConstraintSet::new();
        for i in 0..100 {
            for j in 0..100 {
                for k in 0..10 {
                    let p = [i as f64 * 2.0, j as f64 * 2.0, k as f64 * 5.0];
                    let d =
                        ((p[0] - 100.0).powi(2) + (p[1] - 100.0).powi(2) + (p[2] - 25.0).powi(2))
                            .sqrt();
                    set.push_sample(p, if d < 40.0 { 1.0 } else { -1.0 });
                }
            }
        }
        assert_eq!(set.values.len(), 100_000);

        let started = std::time::Instant::now();
        let gp = Svgp::fit(&set, &spec(128)).unwrap();
        let elapsed = started.elapsed();
        let report = gp.report();
        println!(
            "{} samples, m = {}, {} iterations ({}), search saw {} rows, {:.1?}",
            report.observations,
            report.inducing,
            report.iterations,
            report.status.as_str(),
            report.hyper_observations,
            elapsed
        );
        println!(
            "  ranges {:?}, sill {:.4}, nugget {:.4}, elbo {:.1}",
            report.lengthscales, report.signal_variance, report.noise_variance, report.elbo
        );

        // Every sample informed the fit — that is the whole point.
        assert_eq!(report.observations, 100_000);
        assert!(report.hyper_observations <= 5_000);
        // And it modeled the ball rather than giving up on it.
        assert!(
            gp.value(&[100.0, 100.0, 25.0]) > 0.0,
            "center reads outside"
        );
        assert!(gp.value(&[0.0, 0.0, 0.0]) < 0.0, "corner reads inside");
        assert!(
            report.noise_variance < report.signal_variance,
            "the fit explained the data as noise"
        );
    }

    #[test]
    fn an_empty_constraint_set_is_refused() {
        assert!(Svgp::fit(&ConstraintSet::new(), &spec(16)).is_err());
    }

    #[test]
    fn nonsense_parameters_are_refused() {
        let set = shell_set(10.0);
        assert!(
            Svgp::fit(
                &set,
                &SvgpSpec {
                    inducing: 1,
                    ..spec(16)
                }
            )
            .is_err()
        );
        assert!(
            Svgp::fit(
                &set,
                &SvgpSpec {
                    initial_noise: 0.0,
                    ..spec(16)
                }
            )
            .is_err()
        );
        assert!(
            Svgp::fit(
                &set,
                &SvgpSpec {
                    initial_lengthscales: Some([10.0, -1.0, 10.0]),
                    ..spec(16)
                }
            )
            .is_err()
        );
    }

    /// The inducing placement is what makes `m ≪ n` work; it has to spread over
    /// the data rather than pile up.
    #[test]
    fn the_inducing_placement_spreads_over_the_data() {
        let set = shell_set(10.0);
        let gp = Svgp::fit(&set, &spec(32)).unwrap();
        let z = gp.inducing_points();
        assert_eq!(z.len(), 32);
        let mut min = [f64::INFINITY; 3];
        let mut max = [f64::NEG_INFINITY; 3];
        for p in z {
            for k in 0..3 {
                min[k] = min[k].min(p[k]);
                max[k] = max[k].max(p[k]);
            }
        }
        // The samples span ±16 m; the placement should cover most of that.
        for k in 0..3 {
            assert!(
                max[k] - min[k] > 20.0,
                "axis {k} only spans {}",
                max[k] - min[k]
            );
        }
    }

    /// `ρ(0) = 1` and the derivative prior is positive for every kernel the
    /// engine accepts — the pair of facts the derivative rows rest on.
    #[test]
    fn the_accepted_kernels_have_a_finite_positive_gradient_variance() {
        let w = Vector3::new(0.3, -0.2, 0.1);
        for model in [
            Model::Gaussian,
            Model::Cubic,
            Model::Matern { order: 1.5 },
            Model::Matern { order: 2.5 },
        ] {
            assert!((rho(model, 0.0) - 1.0).abs() < 1e-12, "{model:?}");
            let prior = derivative_prior(model, &w);
            assert!(
                prior.is_finite() && prior > 0.0,
                "{model:?} gives a gradient variance of {prior}"
            );
        }
    }
}
