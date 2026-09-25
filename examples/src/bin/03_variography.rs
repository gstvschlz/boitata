use ceres_examples::{fit_anisotropic, load, variable, write};
use variogram::LagBins;

fn main() {
    let (locs, v) = variable(&load("walker_sample.csv"), "V");
    let bins = LagBins {
        max_lag: 120.0,
        lag_width: 10.0,
    };
    let f = fit_anisotropic(&locs, &v, &bins);

    let p = &f.plane;
    let n = p.lags.len();
    let angle: Vec<f64> = p.angles.iter().flat_map(|a| vec![*a; n]).collect();
    let lag: Vec<f64> = p.angles.iter().flat_map(|_| p.lags.clone()).collect();
    let counts: Vec<f64> = p.counts.iter().map(|&c| c as f64).collect();
    write(
        "03_plane.csv",
        &[
            ("angle", &angle),
            ("lag", &lag),
            ("gamma", &p.gammas),
            ("count", &counts),
        ],
    );

    for (name, exp, azimuth) in [
        ("major", &f.major, f.azimuth),
        ("minor", &f.minor, f.azimuth + 90.0),
    ] {
        let counts: Vec<f64> = exp.counts.iter().map(|&c| c as f64).collect();
        let (x, y) = azimuth.to_radians().sin_cos();
        let model: Vec<f64> = exp
            .lags
            .iter()
            .map(|h| f.model.gamma_points(&(0.0, 0.0, 0.0), &(h * x, h * y, 0.0)))
            .collect();
        write(
            &format!("03_{name}.csv"),
            &[
                ("lag", &exp.lags),
                ("gamma", &exp.gammas),
                ("count", &counts),
                ("model", &model),
            ],
        );
    }

    let s = &f.model.structures[0];
    let ratio = f.model.anisotropy.as_ref().unwrap().angles.semi;
    let variance = {
        let m = v.iter().sum::<f64>() / v.len() as f64;
        v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64
    };
    write(
        "03_model.csv",
        &[
            ("azimuth", &[f.azimuth]),
            ("nugget", &[f.model.nugget]),
            ("sill", &[s.sill]),
            ("major", &[s.range]),
            ("minor", &[s.range * ratio]),
            ("variance", &[variance]),
        ],
    );
    println!(
        "azimuth {:.0}  nugget {:.0}  sill {:.0}  ranges {:.1} / {:.1} m  (variance {variance:.0})",
        f.azimuth,
        f.model.nugget,
        s.sill,
        s.range,
        s.range * ratio
    );
}
