use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Float64Type;
use arrow_array::{Array, ArrayRef, Float64Array, RecordBatch};
use arrow_cast::cast;
use arrow_schema::DataType;
use boitata_core::{BlockModel, Geometry, Layout};
use jpeg_encoder::{ColorType, Encoder, SamplingFactor};
use regex::Regex;
use zune_jpeg::JpegDecoder;
use zune_jpeg::zune_core::bytestream::ZCursor;
use zune_jpeg::zune_core::colorspace::ColorSpace;
use zune_jpeg::zune_core::options::DecoderOptions;

use crate::geotiff::{raster_geometry, raster_transform};
use crate::{Error, Result};

fn bad(message: impl Into<String>) -> Error {
    Error::Jpeg(message.into())
}

fn aux_path(path: &Path) -> PathBuf {
    let mut name = OsString::from(path.as_os_str());
    name.push(".aux.xml");
    name.into()
}

/// Reads a JPEG with a world file as a 2D BlockModel (nz = 1): one `Float64`
/// column `band_1` for grayscale, `band_1` to `band_3` (red, green, blue)
/// otherwise, values 0 to 255 and never null. The world file (`.jgw`, else
/// `.wld`) gives the pixel size, rotation and the center of the top-left
/// pixel; rotation terms become the azimuth. The `SRS` of a `<path>.aux.xml`
/// sidecar, when present, is the CRS.
pub fn read_jpeg(path: impl AsRef<Path>) -> Result<BlockModel> {
    let path = path.as_ref();
    let world = ["jgw", "wld"]
        .iter()
        .map(|e| path.with_extension(e))
        .find(|p| p.is_file())
        .ok_or_else(|| bad(format!("no .jgw or .wld world file next to {path:?}")))?;
    let terms = fs::read_to_string(&world)?
        .split_whitespace()
        .map(|t| t.parse::<f64>())
        .collect::<std::result::Result<Vec<_>, _>>()
        .ok()
        .filter(|t| t.len() == 6)
        .ok_or_else(|| bad("a world file holds six numbers"))?;

    let data = fs::read(path)?;
    let options = DecoderOptions::default()
        .set_max_width(usize::from(u16::MAX))
        .set_max_height(usize::from(u16::MAX));
    let mut header = JpegDecoder::new_with_options(ZCursor::new(&data), options);
    header.decode_headers().map_err(|e| bad(e.to_string()))?;
    let out = match header.input_colorspace() {
        Some(ColorSpace::Luma) => ColorSpace::Luma,
        _ => ColorSpace::RGB,
    };
    let mut decoder =
        JpegDecoder::new_with_options(ZCursor::new(&data), options.jpeg_set_out_colorspace(out));
    let pixels = &decoder.decode().map_err(|e| bad(e.to_string()))?;
    let (w, h) = header.dimensions().unwrap_or_default();
    let bands = out.num_components();

    let [a, d, b, e, c, f] = terms[..] else {
        unreachable!()
    };
    let (along, down) = ([a, d], [b, e]);
    let corner = [c - 0.5 * (a + b), f - 0.5 * (d + e)];
    let (origin, size, azimuth, north_up) = raster_geometry(along, down, corner, h)
        .ok_or_else(|| bad("skewed or mirrored rasters are not supported"))?;
    let geometry = Geometry {
        origin: [origin[0], origin[1], 0.0],
        size: [size[0], size[1], 1.0],
        count: [w, h, 1],
        rotation: [azimuth, 0.0, 0.0],
    };
    let columns = (0..bands).map(|band| {
        let cells = (0..h).flat_map(|j| {
            let r = if north_up { h - 1 - j } else { j };
            (0..w).map(move |c| f64::from(pixels[(r * w + c) * bands + band]))
        });
        let column: ArrayRef = Arc::new(Float64Array::from_iter_values(cells));
        (format!("band_{}", band + 1), column)
    });
    let mut model = BlockModel::regular(geometry, RecordBatch::try_from_iter(columns)?)?;
    let aux = aux_path(path);
    if aux.is_file() {
        let text = fs::read_to_string(aux)?;
        let srs = Regex::new(r"(?s)<SRS[^>]*>(.*?)</SRS>")?;
        model.crs = srs
            .captures(&text)
            .map(|m| unescape(m[1].trim()))
            .filter(|s| !s.is_empty());
    }
    Ok(model)
}

fn unescape(s: &str) -> String {
    s.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
        .replace("&amp;", "&")
}

/// Writes a 2D BlockModel (nz = 1) as a baseline JPEG with a `.jgw` world file:
/// one column as grayscale, three as red, green and blue. JPEG is lossy and
/// 8-bit, so every value must be a whole number from 0 to 255 and no cell may
/// be null; `quality` (1 to 100) trades size for fidelity and chroma is never
/// subsampled. A CRS goes to the `SRS` of a `<path>.aux.xml` sidecar, which is
/// removed when the model has none.
pub fn write_jpeg(path: impl AsRef<Path>, model: &BlockModel, quality: u8) -> Result<()> {
    let path = path.as_ref();
    let g = model.geometry();
    if g.count[2] != 1 {
        return Err(bad("only 2D models (nz = 1) can be written"));
    }
    if matches!(model.layout(), Layout::SubBlocked { .. }) {
        return Err(bad(
            "sub-blocked models cannot be written; regularize first",
        ));
    }
    if g.rotation[1] != 0.0 || g.rotation[2] != 0.0 {
        return Err(bad("only an azimuth rotation can be written"));
    }
    if !(1..=100).contains(&quality) {
        return Err(bad("quality must be from 1 to 100"));
    }
    let model = model.to_regular()?;
    let table = model.attributes();
    let color = match table.num_columns() {
        1 => ColorType::Luma,
        3 => ColorType::Rgb,
        n => return Err(bad(format!("JPEG holds 1 or 3 bands, not {n}"))),
    };
    let (Ok(w), Ok(h)) = (u16::try_from(g.count[0]), u16::try_from(g.count[1])) else {
        return Err(bad("a JPEG is at most 65535 pixels wide and high"));
    };
    let (nx, ny) = (g.count[0], g.count[1]);
    let mut bands = Vec::with_capacity(table.num_columns());
    for (f, column) in table.schema().fields().iter().zip(table.columns()) {
        if !(f.data_type().is_numeric() || f.data_type() == &DataType::Boolean) {
            return Err(bad(format!("column `{}` is not numeric", f.name())));
        }
        let column = cast(column, &DataType::Float64)?;
        let column = column.as_primitive::<Float64Type>();
        if column.null_count() > 0 {
            return Err(bad(format!(
                "column `{}` has nulls, which JPEG cannot store",
                f.name()
            )));
        }
        let values = column.values();
        if values
            .iter()
            .any(|&v| !(0.0..=255.0).contains(&v) || v.fract() != 0.0)
        {
            return Err(bad(format!(
                "column `{}` must hold whole numbers from 0 to 255",
                f.name()
            )));
        }
        bands.push(values.clone());
    }
    let pixels: Vec<u8> = (0..ny)
        .rev()
        .flat_map(|j| (0..nx).map(move |i| j * nx + i))
        .flat_map(|row| bands.iter().map(move |b| b[row] as u8))
        .collect();

    let mut encoder = Encoder::new_file(path, quality).map_err(|e| bad(e.to_string()))?;
    encoder.set_sampling_factor(SamplingFactor::R_4_4_4);
    encoder
        .encode(&pixels, w, h, color)
        .map_err(|e| bad(e.to_string()))?;

    let (along, down, top) = raster_transform(g);
    let center = [0, 1].map(|a| top[a] + 0.5 * (along[a] + down[a]));
    let world = [along[0], along[1], down[0], down[1], center[0], center[1]]
        .map(|v| format!("{v}\n"))
        .concat();
    fs::write(path.with_extension("jgw"), world)?;
    let aux = aux_path(path);
    match &model.crs {
        Some(crs) => {
            let crs = crs
                .replace('&', "&amp;")
                .replace('<', "&lt;")
                .replace('>', "&gt;");
            fs::write(
                aux,
                format!("<PAMDataset>\n  <SRS>{crs}</SRS>\n</PAMDataset>\n"),
            )?;
        }
        None if aux.is_file() => fs::remove_file(aux)?,
        None => {}
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("boitata-jpeg-{name}"));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn model(rotation: f64, bands: usize) -> BlockModel {
        let geometry = Geometry {
            origin: [500_000.5, 7_000_000.25, 0.0],
            size: [2.5, 4.0, 1.0],
            count: [40, 24, 1],
            rotation: [rotation, 0.0, 0.0],
        };
        let n = 40 * 24;
        let columns = (0..bands).map(|b| {
            let v = (0..n).map(|i| {
                let (x, y) = ((i % 40) as f64, (i / 40) as f64);
                (127.5 + 100.0 * ((x + 7.0 * b as f64) / 9.0).sin() * (y / 7.0).cos()).round()
            });
            let column: ArrayRef = Arc::new(Float64Array::from_iter_values(v));
            (format!("band_{}", b + 1), column)
        });
        BlockModel::regular(geometry, RecordBatch::try_from_iter(columns).unwrap()).unwrap()
    }

    #[test]
    fn round_trip_keeps_geometry_crs_and_values_within_tolerance() {
        let path = dir("round-trip").join("grid.jpg");
        for (rotation, bands, crs) in [
            (0.0, 1, Some("EPSG:31982")),
            (30.0, 3, Some("PROJCS[\"local <mine> grid\"]")),
            (215.0, 3, None),
        ] {
            let mut m = model(rotation, bands);
            m.crs = crs.map(Into::into);
            write_jpeg(&path, &m, 100).unwrap();
            let back = read_jpeg(&path).unwrap();
            let (a, b) = (m.geometry(), back.geometry());
            assert_eq!(a.count, b.count);
            for k in 0..3 {
                assert!((a.origin[k] - b.origin[k]).abs() < 1e-6, "{a:?} {b:?}");
                assert!((a.size[k] - b.size[k]).abs() < 1e-9, "{a:?} {b:?}");
                assert!((a.rotation[k] - b.rotation[k]).abs() < 1e-9, "{a:?} {b:?}");
            }
            assert_eq!(back.crs.as_deref(), crs);
            assert_eq!(back.attributes().num_columns(), bands);
            for (x, y) in m
                .attributes()
                .columns()
                .iter()
                .zip(back.attributes().columns())
            {
                let (x, y) = (
                    x.as_primitive::<Float64Type>(),
                    y.as_primitive::<Float64Type>(),
                );
                let worst = x
                    .values()
                    .iter()
                    .zip(y.values())
                    .map(|(x, y)| (x - y).abs())
                    .fold(0.0, f64::max);
                assert!(worst <= 3.0, "max error {worst}");
            }
        }
    }

    #[test]
    fn reads_world_file_with_rows_flipped_and_rejects_bad_models() {
        let path = dir("errors").join("grid.jpg");
        let m = model(0.0, 1);
        write_jpeg(&path, &m, 90).unwrap();
        fs::remove_file(path.with_extension("jgw")).unwrap();
        assert!(read_jpeg(&path).is_err());
        fs::write(path.with_extension("wld"), "2.5\n0\n0\n4\n100\n200\n").unwrap();
        let back = read_jpeg(&path).unwrap();
        let g = back.geometry();
        assert_eq!(g.origin, [98.75, 198.0, 0.0]);
        assert_eq!(g.rotation, [0.0; 3]);
        let first = back.attributes().column(0).as_primitive::<Float64Type>();
        let top = m.attributes().column(0).as_primitive::<Float64Type>();
        assert!((first.value(0) - top.value(40 * 23)).abs() <= 8.0);
        fs::remove_file(path.with_extension("wld")).unwrap();

        let keep: Vec<bool> = (0..960).map(|i| i % 2 == 0).collect();
        let masked = m.mask(&keep.into()).unwrap();
        assert!(write_jpeg(&path, &masked, 90).is_err());
        let two = model(0.0, 2);
        assert!(write_jpeg(&path, &two, 90).is_err());
        let table = RecordBatch::try_from_iter([(
            "v",
            Arc::new(Float64Array::from(vec![0.5; 960])) as ArrayRef,
        )])
        .unwrap();
        let fractional = BlockModel::regular(*m.geometry(), table).unwrap();
        assert!(write_jpeg(&path, &fractional, 90).is_err());
        assert!(write_jpeg(&path, &m, 0).is_err());
    }
}
