//! Coordinate reference systems named by EPSG code: geographic 2D and UTM on
//! WGS 84, SIRGAS 2000, SAD69, NAD83, ETRS89 and GDA2020. Datums change by a
//! 7-parameter Helmert through WGS 84; heights are not changed. Transverse
//! Mercator uses the Krüger series to sixth order in n (Karney 2011).

use std::f64::consts::PI;

use crate::{Error, Result};

const SUPPORTED: &str = "EPSG geographic 2D and UTM codes on WGS 84 (4326, 32601-32660, \
    32701-32760), SIRGAS 2000 (4674, 31965-31985), SAD69 (4618, 29168-29172, 29187-29195), \
    NAD83 (4269, 26901-26923), ETRS89 (4258, 25828-25838) and GDA2020 (7844, 7846-7859)";

#[derive(Clone, Copy, Debug, PartialEq)]
struct Datum {
    a: f64,
    inv_f: f64,
    /// Position-vector Helmert to WGS 84: tx, ty, tz (m), rx, ry, rz
    /// (arc-seconds), scale (ppm).
    to_wgs84: [f64; 7],
}

const GRS80: (f64, f64) = (6_378_137.0, 298.257_222_101);

const WGS84: Datum = Datum {
    a: 6_378_137.0,
    inv_f: 298.257_223_563,
    to_wgs84: [0.0; 7],
};
/// EPSG:15894, SIRGAS 2000 to WGS 84 (1); also EPSG:1188 (NAD83), EPSG:1149
/// (ETRS89) and EPSG:8450 (GDA2020), all null.
const GRS80_NULL: Datum = Datum {
    a: GRS80.0,
    inv_f: GRS80.1,
    to_wgs84: [0.0; 7],
};
/// EPSG:15485, SAD69 to SIRGAS 2000 (1), then the null EPSG:15894.
const SAD69: Datum = Datum {
    a: 6_378_160.0,
    inv_f: 298.25,
    to_wgs84: [-67.35, 3.88, -38.22, 0.0, 0.0, 0.0, 0.0],
};

impl Datum {
    fn e2(&self) -> f64 {
        let f = 1.0 / self.inv_f;
        f * (2.0 - f)
    }

    fn geocentric(&self, lon: f64, lat: f64, h: f64) -> [f64; 3] {
        let e2 = self.e2();
        let n = self.a / (1.0 - e2 * lat.sin().powi(2)).sqrt();
        [
            (n + h) * lat.cos() * lon.cos(),
            (n + h) * lat.cos() * lon.sin(),
            (n * (1.0 - e2) + h) * lat.sin(),
        ]
    }

    fn geodetic(&self, [x, y, z]: [f64; 3]) -> [f64; 3] {
        let e2 = self.e2();
        let p = x.hypot(y);
        let mut lat = z.atan2(p * (1.0 - e2));
        let mut h = 0.0;
        for _ in 0..10 {
            let w = (1.0 - e2 * lat.sin().powi(2)).sqrt();
            h = p * lat.cos() + z * lat.sin() - self.a * w;
            let n = self.a / w;
            lat = z.atan2(p * (1.0 - e2 * n / (n + h)));
        }
        [y.atan2(x), lat, h]
    }
}

fn helmert([x, y, z]: [f64; 3], params: [f64; 7], sign: f64) -> [f64; 3] {
    let arcsec = sign * PI / 648_000.0;
    let [tx, ty, tz] = [0, 1, 2].map(|i| sign * params[i]);
    let [rx, ry, rz] = [3, 4, 5].map(|i| arcsec * params[i]);
    let m = 1.0 + sign * params[6] * 1e-6;
    [
        m * (x - rz * y + ry * z) + tx,
        m * (rz * x + y - rx * z) + ty,
        m * (-ry * x + rx * y + z) + tz,
    ]
}

#[derive(Clone, Copy, Debug)]
struct TransverseMercator {
    /// lat0, lon0 (degrees), k0, false easting, false northing.
    definition: [f64; 5],
    e: f64,
    radius: f64,
    alpha: [f64; 6],
    beta: [f64; 6],
    northing0: f64,
}

impl TransverseMercator {
    fn new(datum: Datum, definition: [f64; 5]) -> Self {
        let n = 1.0 / (2.0 * datum.inv_f - 1.0);
        let p = [1, 2, 3, 4, 5, 6].map(|i| n.powi(i));
        let series = |c: [[f64; 6]; 6]| c.map(|row| (0..6).map(|i| row[i] * p[i]).sum::<f64>());
        let mut tm = Self {
            definition,
            e: datum.e2().sqrt(),
            radius: datum.a / (1.0 + n) * (1.0 + p[1] / 4.0 + p[3] / 64.0 + p[5] / 256.0),
            alpha: series([
                [
                    1. / 2.,
                    -2. / 3.,
                    5. / 16.,
                    41. / 180.,
                    -127. / 288.,
                    7891. / 37800.,
                ],
                [
                    0.,
                    13. / 48.,
                    -3. / 5.,
                    557. / 1440.,
                    281. / 630.,
                    -1983433. / 1935360.,
                ],
                [
                    0.,
                    0.,
                    61. / 240.,
                    -103. / 140.,
                    15061. / 26880.,
                    167603. / 181440.,
                ],
                [
                    0.,
                    0.,
                    0.,
                    49561. / 161280.,
                    -179. / 168.,
                    6601661. / 7257600.,
                ],
                [0., 0., 0., 0., 34729. / 80640., -3418889. / 1995840.],
                [0., 0., 0., 0., 0., 212378941. / 319334400.],
            ]),
            beta: series([
                [
                    1. / 2.,
                    -2. / 3.,
                    37. / 96.,
                    -1. / 360.,
                    -81. / 512.,
                    96199. / 604800.,
                ],
                [
                    0.,
                    1. / 48.,
                    1. / 15.,
                    -437. / 1440.,
                    46. / 105.,
                    -1118711. / 3870720.,
                ],
                [
                    0.,
                    0.,
                    17. / 480.,
                    -37. / 840.,
                    -209. / 4480.,
                    5569. / 90720.,
                ],
                [
                    0.,
                    0.,
                    0.,
                    4397. / 161280.,
                    -11. / 504.,
                    -830251. / 7257600.,
                ],
                [0., 0., 0., 0., 4583. / 161280., -108847. / 3991680.],
                [0., 0., 0., 0., 0., 20648693. / 638668800.],
            ]),
            northing0: 0.0,
        };
        let [lat0, lon0, ..] = definition;
        tm.northing0 = tm.forward(lon0.to_radians(), lat0.to_radians())[1] - definition[4];
        tm
    }

    fn conformal(&self, tau: f64) -> f64 {
        let sigma = (self.e * (self.e * tau / tau.hypot(1.0)).atanh()).sinh();
        tau * sigma.hypot(1.0) - sigma * tau.hypot(1.0)
    }

    fn forward(&self, lon: f64, lat: f64) -> [f64; 2] {
        let [_, lon0, k0, fe, fn_] = self.definition;
        let l = (lon - lon0.to_radians() + PI).rem_euclid(2.0 * PI) - PI;
        let taup = self.conformal(lat.tan());
        let xip = taup.atan2(l.cos());
        let etap = (l.sin() / taup.hypot(l.cos())).asinh();
        let (mut xi, mut eta) = (xip, etap);
        for (j, a) in self.alpha.iter().enumerate() {
            let k = 2.0 * (j + 1) as f64;
            xi += a * (k * xip).sin() * (k * etap).cosh();
            eta += a * (k * xip).cos() * (k * etap).sinh();
        }
        let scale = k0 * self.radius;
        [fe + scale * eta, fn_ + scale * xi - self.northing0]
    }

    fn inverse(&self, x: f64, y: f64) -> [f64; 2] {
        let [_, lon0, k0, fe, fn_] = self.definition;
        let scale = k0 * self.radius;
        let xi = (y - fn_ + self.northing0) / scale;
        let eta = (x - fe) / scale;
        let (mut xip, mut etap) = (xi, eta);
        for (j, b) in self.beta.iter().enumerate() {
            let k = 2.0 * (j + 1) as f64;
            xip -= b * (k * xi).sin() * (k * eta).cosh();
            etap -= b * (k * xi).cos() * (k * eta).sinh();
        }
        let taup = xip.sin() / etap.sinh().hypot(xip.cos());
        let e2m = 1.0 - self.e * self.e;
        let mut tau = taup / e2m;
        for _ in 0..6 {
            let taui = self.conformal(tau);
            tau +=
                (taup - taui) / taui.hypot(1.0) * (1.0 + e2m * tau * tau) / (e2m * tau.hypot(1.0));
        }
        [lon0.to_radians() + etap.sinh().atan2(xip.cos()), tau.atan()]
    }
}

#[derive(Clone, Copy, Debug)]
struct Crs {
    code: u32,
    datum: Datum,
    projection: Option<TransverseMercator>,
}

fn utm(datum: Datum, zone: u32, south: bool) -> Option<TransverseMercator> {
    let lon0 = 6.0 * zone as f64 - 183.0;
    let fn_ = if south { 10_000_000.0 } else { 0.0 };
    Some(TransverseMercator::new(
        datum,
        [0.0, lon0, 0.9996, 500_000.0, fn_],
    ))
}

/// The EPSG code of `crs`: `EPSG:n`, or WKT ending with an EPSG authority.
fn code(crs: &str) -> Option<u32> {
    let s = crs.trim();
    if s.len() > 5 && s[..5].eq_ignore_ascii_case("EPSG:") {
        return s[5..].trim().parse().ok();
    }
    let rest = s[s.rfind("\"EPSG\"")? + 6..].trim_start_matches([',', ' ', '"']);
    let end = rest.find(|c: char| !c.is_ascii_digit())?;
    rest[end..]
        .trim_matches(|c: char| c == '"' || c == ']' || c == ')' || c.is_whitespace())
        .is_empty()
        .then(|| rest[..end].parse().ok())?
}

fn parse(crs: &str) -> Result<Crs> {
    let unsupported = || Error::Crs(format!("unsupported CRS {crs}; supported are {SUPPORTED}"));
    let code = code(crs).ok_or_else(unsupported)?;
    let (datum, projection) = match code {
        4326 => (WGS84, None),
        32601..=32660 => (WGS84, utm(WGS84, code - 32600, false)),
        32701..=32760 => (WGS84, utm(WGS84, code - 32700, true)),
        4674 | 4269 | 4258 | 7844 => (GRS80_NULL, None),
        31965..=31976 => (GRS80_NULL, utm(GRS80_NULL, code - 31954, false)),
        31977..=31985 => (GRS80_NULL, utm(GRS80_NULL, code - 31960, true)),
        26901..=26923 => (GRS80_NULL, utm(GRS80_NULL, code - 26900, false)),
        25828..=25838 => (GRS80_NULL, utm(GRS80_NULL, code - 25800, false)),
        7846..=7859 => (GRS80_NULL, utm(GRS80_NULL, code - 7800, true)),
        4618 => (SAD69, None),
        29168..=29172 => (SAD69, utm(SAD69, code - 29150, false)),
        29187..=29195 => (SAD69, utm(SAD69, code - 29170, true)),
        _ => return Err(unsupported()),
    };
    Ok(Crs {
        code,
        datum,
        projection,
    })
}

/// A change of coordinates between two supported CRSs. Geographic
/// coordinates are longitude, latitude in degrees; projected ones easting,
/// northing in metres; z is the height in metres and is kept.
#[derive(Clone, Copy, Debug)]
pub struct Transform {
    from: Crs,
    to: Crs,
}

impl Transform {
    pub fn new(from: &str, to: &str) -> Result<Self> {
        Ok(Self {
            from: parse(from)?,
            to: parse(to)?,
        })
    }

    /// `EPSG:n` of the target.
    pub fn target(&self) -> String {
        format!("EPSG:{}", self.to.code)
    }

    /// False when the coordinates stay as they are: same projection and a
    /// null datum change between them.
    pub fn moves(&self) -> bool {
        self.from.datum.to_wgs84 != self.to.datum.to_wgs84
            || self.from.projection.map(|p| p.definition)
                != self.to.projection.map(|p| p.definition)
    }

    pub fn is_geographic(&self) -> (bool, bool) {
        (self.from.projection.is_none(), self.to.projection.is_none())
    }

    pub fn apply(&self, [x, y, z]: [f64; 3]) -> Result<[f64; 3]> {
        if !self.moves() {
            return Ok([x, y, z]);
        }
        let [lon, lat] = match &self.from.projection {
            Some(tm) => tm.inverse(x, y),
            None if (-90.0..=90.0).contains(&y) => [x.to_radians(), y.to_radians()],
            None => return Err(Error::Crs(format!("latitude {y} is outside [-90, 90]"))),
        };
        let [lon, lat] = if self.from.datum.to_wgs84 == self.to.datum.to_wgs84 {
            [lon, lat]
        } else {
            let p = self.from.datum.geocentric(lon, lat, z);
            let p = helmert(
                helmert(p, self.from.datum.to_wgs84, 1.0),
                self.to.datum.to_wgs84,
                -1.0,
            );
            let [lon, lat, _] = self.to.datum.geodetic(p);
            [lon, lat]
        };
        Ok(match &self.to.projection {
            Some(tm) => {
                let [e, n] = tm.forward(lon, lat);
                [e, n, z]
            }
            None => [
                ((lon + PI).rem_euclid(2.0 * PI) - PI).to_degrees(),
                lat.to_degrees(),
                z,
            ],
        })
    }
}

/// Converts `coords` in `crs` and `unit` to `target`, returning the target
/// as `EPSG:n` and the length unit the coordinates then have: metres when
/// projected, none when geographic. Coordinates that do not move keep their
/// unit.
pub(crate) fn reproject(
    crs: Option<&str>,
    unit: Option<&str>,
    target: &str,
    coords: &mut [[f64; 3]],
) -> Result<(String, Option<String>)> {
    let crs =
        crs.ok_or_else(|| Error::Crs("declare the CRS of the coordinates to convert them".into()))?;
    let transform = Transform::new(crs, target)?;
    if !transform.moves() {
        return Ok((transform.target(), unit.map(str::to_string)));
    }
    let k = unit.map_or(Ok(1.0), |u| crate::units::conversion(u, "m"))?;
    let (geographic, to_geographic) = transform.is_geographic();
    let kxy = if geographic { 1.0 } else { k };
    for p in coords.iter_mut() {
        *p = transform.apply([p[0] * kxy, p[1] * kxy, p[2] * k])?;
    }
    Ok((transform.target(), (!to_geographic).then(|| "m".into())))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dms(d: f64, m: f64, s: f64) -> f64 {
        d + m / 60.0 + s / 3600.0
    }

    #[test]
    fn geocentric_matches_the_epsg_example() {
        // IOGP Guidance Note 7-2, geographic/geocentric conversions (EPSG:9602).
        let lat = dms(53.0, 48.0, 33.82).to_radians();
        let lon = dms(2.0, 7.0, 46.38).to_radians();
        let p = WGS84.geocentric(lon, lat, 73.0);
        let expected = [3_771_793.968, 140_253.342, 5_124_304.349];
        (0..3).for_each(|i| assert!((p[i] - expected[i]).abs() < 1e-3, "{p:?}"));
        let [lon2, lat2, h] = WGS84.geodetic(p);
        assert!(
            (lon2 - lon).abs() < 1e-12 && (lat2 - lat).abs() < 1e-12 && (h - 73.0).abs() < 1e-6
        );
    }

    #[test]
    fn helmert_matches_the_epsg_example() {
        // IOGP Guidance Note 7-2, position vector transformation (EPSG:9606),
        // WGS 72 to WGS 84 (EPSG:1238).
        let params = [0.0, 0.0, 4.5, 0.0, 0.0, 0.554, 0.219];
        let p = [3_657_660.66, 255_768.55, 5_201_382.11];
        let q = helmert(p, params, 1.0);
        let expected = [3_657_660.78, 255_778.43, 5_201_387.75];
        (0..3).for_each(|i| assert!((q[i] - expected[i]).abs() < 0.01, "{q:?}"));
        let back = helmert(q, params, -1.0);
        (0..3).for_each(|i| assert!((back[i] - p[i]).abs() < 1e-3));
    }

    #[test]
    fn transverse_mercator_matches_the_epsg_example() {
        // IOGP Guidance Note 7-2, Transverse Mercator (EPSG:9807), British
        // National Grid (EPSG:27700) on Airy 1830.
        let airy = Datum {
            a: 6_377_563.396,
            inv_f: 299.324_964_6,
            to_wgs84: [0.0; 7],
        };
        let tm =
            TransverseMercator::new(airy, [49.0, -2.0, 0.999_601_271_7, 400_000.0, -100_000.0]);
        let [e, n] = tm.forward(0.5f64.to_radians(), 50.5f64.to_radians());
        assert!(
            (e - 577_274.99).abs() < 0.01 && (n - 69_740.50).abs() < 0.01,
            "{e} {n}"
        );
        let [lon, lat] = tm.inverse(e, n);
        assert!((lon.to_degrees() - 0.5).abs() < 1e-10 && (lat.to_degrees() - 50.5).abs() < 1e-10);
    }

    #[test]
    fn utm_round_trips_below_a_tenth_of_a_millimetre() {
        let geo = Transform::new("EPSG:4326", "EPSG:32722").unwrap();
        let back = Transform::new("EPSG:32722", "EPSG:4326").unwrap();
        assert_eq!(
            geo.apply([-51.0, 0.0, 5.0]).unwrap(),
            [500_000.0, 10_000_000.0, 5.0]
        );
        for lat in (-80..=0).step_by(5) {
            for dlon in [-4.0, -3.0, -1.5, 0.0, 0.7, 3.0, 4.0] {
                let p = [-51.0 + dlon, lat as f64, 0.0];
                let xy = geo.apply(p).unwrap();
                let q = back.apply(xy).unwrap();
                let again = geo.apply(q).unwrap();
                assert!((again[0] - xy[0]).hypot(again[1] - xy[1]) < 1e-4, "{p:?}");
                assert!((q[0] - p[0]).abs() < 1e-9 && (q[1] - p[1]).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn sad69_to_sirgas_shifts_by_the_published_translation() {
        // EPSG:15485, SAD69 to SIRGAS 2000 (1).
        let t = Transform::new("EPSG:4618", "EPSG:4674").unwrap();
        let [lon, lat, _] = t.apply([-43.0, -20.0, 800.0]).unwrap();
        let a = SAD69.geocentric(-43f64.to_radians(), -20f64.to_radians(), 800.0);
        let b = GRS80_NULL.geocentric(lon.to_radians(), lat.to_radians(), 800.0);
        let shift = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        // The height moves with the shift and is kept, so only the horizontal
        // part is reproduced.
        let up = [
            lat.to_radians().cos() * lon.to_radians().cos(),
            lat.to_radians().cos() * lon.to_radians().sin(),
            lat.to_radians().sin(),
        ];
        let expected = [-67.35, 3.88, -38.22];
        let dh: f64 = (0..3).map(|i| (expected[i] - shift[i]) * up[i]).sum();
        (0..3).for_each(|i| {
            assert!(
                (shift[i] + dh * up[i] - expected[i]).abs() < 1e-3,
                "{shift:?}"
            )
        });
        let back = Transform::new("EPSG:4674", "EPSG:4618").unwrap();
        let q = back.apply([lon, lat, 800.0]).unwrap();
        assert!((q[0] + 43.0).abs() < 1e-9 && (q[1] + 20.0).abs() < 1e-9);
    }

    #[test]
    fn null_datum_changes_keep_the_coordinates() {
        let t = Transform::new("EPSG:31982", "epsg:32722").unwrap();
        assert!(!t.moves());
        assert_eq!(t.target(), "EPSG:32722");
        assert!(Transform::new("EPSG:31982", "EPSG:32723").unwrap().moves());
    }

    #[test]
    fn reads_codes_from_wkt_and_rejects_others() {
        let wkt = r#"PROJCS["SIRGAS 2000 / UTM zone 22S",GEOGCS["SIRGAS 2000",AUTHORITY["EPSG","4674"]],AUTHORITY["EPSG","31982"]]"#;
        assert_eq!(code(wkt), Some(31982));
        assert_eq!(
            code(r#"PROJCS["x",GEOGCS["y",AUTHORITY["EPSG","4674"]],UNIT["m",1]]"#),
            None
        );
        let error = Transform::new("EPSG:27700", "EPSG:4326")
            .unwrap_err()
            .to_string();
        assert!(error.contains("SIRGAS 2000"), "{error}");
    }
}
