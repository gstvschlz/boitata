//! A variogram volume recovers the anisotropy a field was simulated with.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use simulation::{Bands, TurningBandsParams};
use variogram::aniso::{Angles, Anisotropy};
use variogram::{LagBins, Model, Variogram, VolumeParams, variogram_volume};

fn field(
    rotation: (f64, f64, f64),
    ratios: (f64, f64),
    seed: u64,
) -> (Vec<(f64, f64, f64)>, Vec<f64>) {
    let angles = Angles {
        azimuth: rotation.0,
        dip: rotation.1,
        rake: rotation.2,
        major: 1.0,
        semi: ratios.0,
        minor: ratios.1,
    };
    let vg = Variogram::single(Model::Spherical, 1.0, 50.0)
        .with_anisotropy(Anisotropy::new(angles).unwrap());
    let mut rng = StdRng::seed_from_u64(seed);
    let points: Vec<_> = (0..5000)
        .map(|_| {
            (
                rng.gen_range(0.0..200.0),
                rng.gen_range(0.0..200.0),
                rng.gen_range(0.0..200.0),
            )
        })
        .collect();
    let params = TurningBandsParams {
        n_bands: 500,
        ..Default::default()
    };
    let bands = Bands::new([0.0; 3], [200.0; 3], &vg, &params, &mut rng);
    let values = bands.field(&points);
    (points, values)
}

fn params() -> VolumeParams {
    VolumeParams {
        bins: LagBins {
            max_lag: 70.0,
            lag_width: 5.0,
        },
        ..Default::default()
    }
}

fn angle_between(a: (f64, f64), b: (f64, f64)) -> f64 {
    let u = variogram::unit_vector(a.0, a.1);
    let v = variogram::unit_vector(b.0, b.1);
    (u.0 * v.0 + u.1 * v.1 + u.2 * v.2)
        .abs()
        .min(1.0)
        .acos()
        .to_degrees()
}

#[test]
fn recovers_a_rotated_anisotropy() {
    let (points, values) = field((40.0, 30.0, 20.0), (0.5, 0.25), 7);
    let found = variogram_volume(&points, &values, &params())
        .unwrap()
        .angles;
    println!("{found:?}");
    let major = angle_between((found.azimuth, found.dip), (40.0, 30.0));
    assert!(major < 6.0, "major axis off by {major:.1}°: {found:?}");
    assert!((found.rake - 20.0).abs() < 8.0, "{found:?}");
    assert!((found.semi / found.major - 0.5).abs() < 0.1, "{found:?}");
    assert!((found.minor / found.major - 0.25).abs() < 0.08, "{found:?}");
    assert!((found.major / 50.0 - 1.0).abs() < 0.3, "{found:?}");
}

#[test]
fn isotropic_field_has_ratios_near_one() {
    let (points, values) = field((0.0, 0.0, 0.0), (1.0, 1.0), 11);
    let found = variogram_volume(&points, &values, &params())
        .unwrap()
        .angles;
    println!("{found:?}");
    assert!(found.minor / found.major > 0.75, "{found:?}");
}

#[test]
fn same_result_on_one_and_eight_threads() {
    let (points, values) = field((40.0, 30.0, 20.0), (0.5, 0.25), 3);
    let run = |threads| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| variogram_volume(&points, &values, &params()).unwrap())
    };
    let (one, eight) = (run(1), run(8));
    assert_eq!(one.angles, eight.angles);
    assert_eq!(one.ranges, eight.ranges);
    assert_eq!(one.counts, eight.counts);
    let bits = |g: &[f64]| g.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&one.gammas), bits(&eight.gammas));
}
