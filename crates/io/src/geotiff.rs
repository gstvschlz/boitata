use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Float64Type;
use arrow_array::{Array, ArrayRef, Float32Array, Float64Array, RecordBatch};
use arrow_cast::cast;
use arrow_schema::DataType;
use ceres_core::{BlockModel, Geometry, Layout};
use regex::Regex;
use tiff::TiffError;
use tiff::decoder::{ChunkType, Decoder, DecodingResult, Limits};
use tiff::encoder::colortype::{ColorType, Gray32Float, Gray64Float};
use tiff::encoder::{Compression, DeflateLevel, TiffEncoder, TiffValue};
use tiff::tags::{ExtraSamples, Tag};

use crate::{Error, Result};

const GDAL_METADATA: Tag = Tag::Unknown(42112);

fn bad(message: impl Into<String>) -> Error {
    Error::GeoTiff(message.into())
}

fn tiff_error(e: TiffError) -> Error {
    match e {
        TiffError::IoError(e) => Error::Io(e),
        e => bad(e.to_string()),
    }
}

/// Reads a GeoTIFF as a 2D BlockModel (nz = 1), one column per band.
///
/// Stripped or tiled, chunky or planar, integer or float samples. Float32
/// bands stay `Float32`, the rest become `Float64`. Pixels equal to `nodata`
/// (the file's `GDAL_NODATA` by default) and NaN are null. Georeferencing comes
/// from ModelPixelScale + ModelTiepoint or from ModelTransformation, rotation
/// included, with pixel-is-point rasters shifted so cell centers fall on the
/// tie points. An EPSG code in the GeoKeys becomes `EPSG:<code>`, otherwise the
/// citation is the CRS. Band names come from GDAL's band descriptions.
pub fn read_geotiff(path: impl AsRef<Path>, nodata: Option<f64>) -> Result<BlockModel> {
    let mut tif = Decoder::new(BufReader::new(File::open(path)?))
        .map_err(tiff_error)?
        .with_limits(Limits::unlimited());
    let (w, h) = tif.dimensions().map_err(tiff_error)?;
    let (w, h) = (w as usize, h as usize);
    let bands = tif
        .find_tag_unsigned::<usize>(Tag::SamplesPerPixel)
        .map_err(tiff_error)?
        .unwrap_or(1);
    let f64s = |tif: &mut Decoder<_>, tag| -> Result<Option<Vec<f64>>> {
        tif.find_tag(tag)
            .and_then(|v| v.map(|v| v.into_f64_vec()).transpose())
            .map_err(tiff_error)
    };
    let scale = f64s(&mut tif, Tag::ModelPixelScaleTag)?;
    let tiepoint = f64s(&mut tif, Tag::ModelTiepointTag)?;
    let transform = f64s(&mut tif, Tag::ModelTransformationTag)?;
    let keys = tif
        .find_tag(Tag::GeoKeyDirectoryTag)
        .and_then(|v| v.map(|v| v.into_u16_vec()).transpose())
        .map_err(tiff_error)?
        .unwrap_or_default();
    let ascii = tif
        .find_tag(Tag::GeoAsciiParamsTag)
        .and_then(|v| v.map(|v| v.into_string()).transpose())
        .map_err(tiff_error)?
        .unwrap_or_default();
    let text = |tif: &mut Decoder<_>, tag| {
        tif.find_tag(tag)
            .ok()
            .flatten()
            .and_then(|v| v.into_string().ok())
    };
    let nodata = match nodata {
        Some(v) => Some(v),
        None => text(&mut tif, Tag::GdalNodata)
            .map(|t| {
                let t = t.trim_end_matches('\0').trim();
                t.parse::<f64>()
                    .map_err(|_| bad(format!("GDAL_NODATA `{t}` is not a number")))
            })
            .transpose()?,
    };
    let metadata = text(&mut tif, GDAL_METADATA).unwrap_or_default();

    let (cw, ch) = tif.chunk_dimensions();
    let (cw, ch) = (cw as usize, ch as usize);
    let across = w.div_ceil(cw);
    let per_plane = across * h.div_ceil(ch);
    let chunks = match tif.get_chunk_type() {
        ChunkType::Strip => tif.strip_count(),
        ChunkType::Tile => tif.tile_count(),
    }
    .map_err(tiff_error)? as usize;
    let per_pixel = match chunks {
        n if n == per_plane => bands,
        n if n == per_plane * bands => 1,
        _ => return Err(bad("unexpected number of strips or tiles")),
    };
    let mut values = vec![0.0; w * h * bands];
    let mut single = false;
    for k in 0..chunks {
        let (plane, at) = (k / per_plane, k % per_plane);
        let (x0, y0) = (at % across * cw, at / across * ch);
        let chunk = tif.read_chunk(k as u32).map_err(tiff_error)?;
        let dw = tif.chunk_data_dimensions(k as u32).0 as usize;
        let chunk: Vec<f64> = match chunk {
            DecodingResult::U8(v) => v.into_iter().map(f64::from).collect(),
            DecodingResult::U16(v) => v.into_iter().map(f64::from).collect(),
            DecodingResult::U32(v) => v.into_iter().map(f64::from).collect(),
            DecodingResult::U64(v) => v.into_iter().map(|x| x as f64).collect(),
            DecodingResult::I8(v) => v.into_iter().map(f64::from).collect(),
            DecodingResult::I16(v) => v.into_iter().map(f64::from).collect(),
            DecodingResult::I32(v) => v.into_iter().map(f64::from).collect(),
            DecodingResult::I64(v) => v.into_iter().map(|x| x as f64).collect(),
            DecodingResult::F16(v) => v.into_iter().map(|x| x.to_f64()).collect(),
            DecodingResult::F32(v) => {
                single = true;
                v.into_iter().map(f64::from).collect()
            }
            DecodingResult::F64(v) => v,
        };
        for (r, row) in chunk.chunks_exact(dw * per_pixel).enumerate() {
            let y = y0 + r;
            for (c, pixel) in row.chunks_exact(per_pixel).enumerate() {
                let x = x0 + c;
                if y < h && x < w {
                    let at = (y * w + x) * bands + plane;
                    values[at..at + per_pixel].copy_from_slice(pixel);
                }
            }
        }
    }
    let nodata = nodata.map(|v| if single { v as f32 as f64 } else { v });

    let key = |id: u16| {
        keys.get(4..)?
            .chunks_exact(4)
            .find(|k| k[0] == id)
            .map(|k| (k[1], k[2] as usize, k[3]))
    };
    let point = matches!(key(1025), Some((0, _, 2)));
    let s = if point { 0.5 } else { 0.0 };
    let (along, down, tie, z0, dz) = match (&transform, &scale, &tiepoint) {
        (Some(m), ..) if m.len() >= 12 => (
            [m[0], m[4]],
            [m[1], m[5]],
            [0.0, 0.0, m[3], m[7]],
            m[11],
            m[10],
        ),
        (_, Some(sc), Some(t)) if sc.len() >= 2 && t.len() >= 6 => (
            [sc[0], 0.0],
            [0.0, -sc[1]],
            [t[0], t[1], t[3], t[4]],
            t[5],
            sc.get(2).copied().unwrap_or(0.0),
        ),
        _ => {
            return Err(bad(
                "no ModelTransformation or ModelPixelScale + ModelTiepoint",
            ));
        }
    };
    let world = |c: f64, r: f64| {
        let (c, r) = (c - s - tie[0], r - s - tie[1]);
        [0, 1].map(|a| tie[2 + a] + c * along[a] + r * down[a])
    };
    let (sx, sy) = (along[0].hypot(along[1]), down[0].hypot(down[1]));
    let ux = [along[0] / sx, along[1] / sx];
    let uy = [-ux[1], ux[0]];
    let dot = |d: [f64; 2]| (uy[0] * d[0] + uy[1] * d[1]) / sy;
    let north_up = if dot(down) < -1.0 + 1e-9 {
        true
    } else if dot(down) > 1.0 - 1e-9 {
        false
    } else {
        return Err(bad("skewed or mirrored rasters are not supported"));
    };
    let origin = world(0.0, if north_up { h as f64 } else { 0.0 });
    let geometry = Geometry {
        origin: [origin[0], origin[1], z0],
        size: [sx, sy, if dz > 0.0 { dz } else { 1.0 }],
        count: [w, h, 1],
        rotation: [
            uy[0].atan2(uy[1]).to_degrees().rem_euclid(360.0) + 0.0,
            0.0,
            0.0,
        ],
    };

    let names = band_names(&metadata, bands);
    let mut columns: Vec<(String, ArrayRef)> = Vec::with_capacity(bands);
    for (b, name) in names.into_iter().enumerate() {
        let cells = (0..h).flat_map(|j| {
            let r = if north_up { h - 1 - j } else { j };
            (0..w).map(move |c| (r * w + c) * bands + b)
        });
        let cells = cells.map(|at| {
            let v = values[at];
            (!v.is_nan() && Some(v) != nodata).then_some(v)
        });
        let column: ArrayRef = if single {
            Arc::new(Float32Array::from_iter(cells.map(|v| v.map(|v| v as f32))))
        } else {
            Arc::new(Float64Array::from_iter(cells))
        };
        columns.push((name, column));
    }
    let mut model = BlockModel::regular(geometry, RecordBatch::try_from_iter(columns)?)?;
    model.crs = crs(&key, &ascii);
    Ok(model)
}

fn crs(key: &dyn Fn(u16) -> Option<(u16, usize, u16)>, ascii: &str) -> Option<String> {
    for id in [3072, 2048] {
        if let Some((0, _, code @ 1..32767)) = key(id) {
            return Some(format!("EPSG:{code}"));
        }
    }
    [1026, 3073, 2049].into_iter().find_map(|id| {
        let (34737, count, offset) = key(id)? else {
            return None;
        };
        let text = ascii.get(offset as usize..offset as usize + count)?;
        let text = text.trim_end_matches(['|', '\0']).trim();
        (!text.is_empty()).then(|| text.to_string())
    })
}

fn band_names(metadata: &str, bands: usize) -> Vec<String> {
    let mut names: Vec<String> = (1..=bands).map(|b| format!("band_{b}")).collect();
    let item = Regex::new(r"<Item([^>]*)>([^<]*)</Item>").expect("valid regex");
    let sample = Regex::new(r#"sample="(\d+)""#).expect("valid regex");
    for m in item.captures_iter(metadata) {
        if !m[1].contains(r#"name="DESCRIPTION""#) {
            continue;
        }
        let Some(b) = sample
            .captures(&m[1])
            .and_then(|s| s[1].parse::<usize>().ok())
        else {
            continue;
        };
        if let Some(name) = names.get_mut(b).filter(|_| !m[2].is_empty()) {
            *name = m[2]
                .replace("&lt;", "<")
                .replace("&gt;", ">")
                .replace("&quot;", "\"")
                .replace("&apos;", "'")
                .replace("&amp;", "&");
        }
    }
    names
}

/// Writes a 2D BlockModel (nz = 1) as a deflate-compressed GeoTIFF, one band
/// per column: `Float32` when every column is, `Float64` otherwise. Nulls
/// become `nodata`, stored in `GDAL_NODATA`; a value equal to it is an error.
/// Unrotated grids use ModelPixelScale + ModelTiepoint, rotated ones
/// ModelTransformation. A masked model is written filled with `nodata`. An
/// `EPSG:<code>` CRS goes to the GeoKeys, any other CRS to the citation.
pub fn write_geotiff(path: impl AsRef<Path>, model: &BlockModel, nodata: f64) -> Result<()> {
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
    let model = model.to_regular()?;
    let table = model.attributes();
    if table.num_columns() == 0 {
        return Err(bad("the model has no columns to write"));
    }
    let schema = table.schema();
    for f in schema.fields() {
        if !(f.data_type().is_numeric() || f.data_type() == &DataType::Boolean) {
            return Err(bad(format!("column `{}` is not numeric", f.name())));
        }
    }
    let single = schema
        .fields()
        .iter()
        .all(|f| f.data_type() == &DataType::Float32);
    let (Ok(w), Ok(h)) = (u32::try_from(g.count[0]), u32::try_from(g.count[1])) else {
        return Err(bad("the grid is too large for a TIFF"));
    };
    let (nx, ny) = (g.count[0], g.count[1]);
    let raster = |column: &dyn Array| -> Result<Vec<Option<f64>>> {
        let column = cast(column, &DataType::Float64)?;
        let column = column.as_primitive::<Float64Type>();
        Ok((0..ny)
            .rev()
            .flat_map(|j| (0..nx).map(move |i| j * nx + i))
            .map(|row| column.is_valid(row).then(|| column.value(row)))
            .collect())
    };
    let bands = table
        .columns()
        .iter()
        .map(|c| raster(c.as_ref()))
        .collect::<Result<Vec<_>>>()?;

    let nd = if single { nodata as f32 as f64 } else { nodata };
    let mut data = Vec::with_capacity(nx * ny * bands.len());
    for p in 0..nx * ny {
        for (b, band) in bands.iter().enumerate() {
            data.push(match band[p] {
                Some(v) if v == nd || (single && v as f32 as f64 == nd) => {
                    return Err(bad(format!(
                        "column `{}` holds the nodata value {nodata}; choose another",
                        schema.field(b).name()
                    )));
                }
                Some(v) => v,
                None => nd,
            });
        }
    }

    let file = BufWriter::new(File::create(path)?);
    let mut tif = TiffEncoder::new(file)
        .map_err(tiff_error)?
        .with_compression(Compression::Deflate(DeflateLevel::default()));
    let tags = Tags::new(&model, nodata, single);
    if single {
        let data: Vec<f32> = data.iter().map(|&v| v as f32).collect();
        encode::<_, Gray32Float>(&mut tif, w, h, bands.len(), &data, &tags)
    } else {
        encode::<_, Gray64Float>(&mut tif, w, h, bands.len(), &data, &tags)
    }
}

struct Tags {
    keys: Vec<u16>,
    ascii: Option<String>,
    scale: Vec<f64>,
    tiepoint: Vec<f64>,
    transform: Vec<f64>,
    nodata: String,
    metadata: String,
}

impl Tags {
    fn new(model: &BlockModel, nodata: f64, single: bool) -> Self {
        let g = model.geometry();
        let mut keys = vec![(1025u16, 0u16, 1u16, 1u16)];
        let mut ascii = None;
        if let Some(crs) = &model.crs {
            let code = crs
                .trim()
                .get(..5)
                .filter(|p| p.eq_ignore_ascii_case("EPSG:"))
                .and_then(|_| crs.trim()[5..].trim().parse::<u16>().ok())
                .filter(|c| (1..32767).contains(c));
            match code {
                Some(code) if (4000..5000).contains(&code) => {
                    keys.extend([(1024, 0, 1, 2), (2048, 0, 1, code)])
                }
                Some(code) => keys.extend([(1024, 0, 1, 1), (3072, 0, 1, code)]),
                None => {
                    let text = format!("{}|", crs.replace('|', "/"));
                    keys.push((1026, 34737, text.len() as u16, 0));
                    ascii = Some(text);
                }
            }
        }
        keys.sort();
        let mut directory = vec![1, 1, 0, keys.len() as u16];
        directory.extend(keys.into_iter().flat_map(|k| [k.0, k.1, k.2, k.3]));

        let ny = g.count[1] as f64;
        let (scale, tiepoint, transform) = if g.rotation[0].rem_euclid(360.0) == 0.0 {
            (
                vec![g.size[0], g.size[1], g.size[2]],
                vec![
                    0.0,
                    0.0,
                    0.0,
                    g.origin[0],
                    g.origin[1] + ny * g.size[1],
                    g.origin[2],
                ],
                vec![],
            )
        } else {
            let (s, c) = g.rotation[0].to_radians().sin_cos();
            let (ux, uy) = ([c, -s], [s, c]);
            let top = [0, 1].map(|a| g.origin[a] + ny * g.size[1] * uy[a]);
            #[rustfmt::skip]
            let m = vec![
                g.size[0] * ux[0], -g.size[1] * uy[0], 0.0, top[0],
                g.size[0] * ux[1], -g.size[1] * uy[1], 0.0, top[1],
                0.0, 0.0, g.size[2], g.origin[2],
                0.0, 0.0, 0.0, 1.0,
            ];
            (vec![], vec![], m)
        };

        let nodata = match nodata {
            v if v.is_nan() => "nan".to_string(),
            v if single => format!("{}", v as f32),
            v => format!("{v}"),
        };
        let escape = |s: &str| {
            s.replace('&', "&amp;")
                .replace('<', "&lt;")
                .replace('>', "&gt;")
                .replace('"', "&quot;")
        };
        let items: String = model
            .attributes()
            .schema()
            .fields()
            .iter()
            .enumerate()
            .map(|(b, f)| {
                format!(
                    "  <Item name=\"DESCRIPTION\" sample=\"{b}\" role=\"description\">{}</Item>\n",
                    escape(f.name())
                )
            })
            .collect();
        Tags {
            keys: directory,
            ascii,
            scale,
            tiepoint,
            transform,
            nodata,
            metadata: format!("<GDALMetadata>\n{items}</GDALMetadata>\n"),
        }
    }
}

fn encode<W, C>(
    tif: &mut TiffEncoder<W>,
    w: u32,
    h: u32,
    bands: usize,
    data: &[C::Inner],
    tags: &Tags,
) -> Result<()>
where
    W: std::io::Write + std::io::Seek,
    C: ColorType,
    [C::Inner]: TiffValue,
{
    let mut run = || -> tiff::TiffResult<()> {
        let mut image = tif.new_image::<C>(w, h)?;
        if bands > 1 {
            image.extra_samples(&vec![ExtraSamples::Unspecified; bands - 1])?;
        }
        let dir = image.encoder();
        dir.write_tag(Tag::GeoKeyDirectoryTag, tags.keys.as_slice())?;
        if let Some(ascii) = &tags.ascii {
            dir.write_tag(Tag::GeoAsciiParamsTag, ascii.as_str())?;
        }
        if tags.transform.is_empty() {
            dir.write_tag(Tag::ModelPixelScaleTag, tags.scale.as_slice())?;
            dir.write_tag(Tag::ModelTiepointTag, tags.tiepoint.as_slice())?;
        } else {
            dir.write_tag(Tag::ModelTransformationTag, tags.transform.as_slice())?;
        }
        dir.write_tag(Tag::GdalNodata, tags.nodata.as_str())?;
        dir.write_tag(GDAL_METADATA, tags.metadata.as_str())?;
        image.write_data(data)
    };
    run().map_err(tiff_error)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tiff::encoder::compression::{CompressionAlgorithm, Deflate};

    fn dir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("ceres-geotiff-{name}"));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn model(rotation: f64, single: bool) -> BlockModel {
        let geometry = Geometry {
            origin: [500_000.5, 7_000_000.25, 12.0],
            size: [2.5, 4.0, 1.0],
            count: [3, 2, 1],
            rotation: [rotation, 0.0, 0.0],
        };
        let au = [
            Some(0.1),
            None,
            Some(-3.5),
            Some(7.25),
            Some(1e-9),
            Some(2.0),
        ];
        let au: ArrayRef = if single {
            Arc::new(Float32Array::from_iter(au.map(|v| v.map(|v| v as f32))))
        } else {
            Arc::new(Float64Array::from_iter(au))
        };
        let cu: ArrayRef = Arc::new(Float32Array::from(vec![
            None,
            Some(1.5),
            Some(2.0),
            None,
            Some(0.0),
            Some(-1.0),
        ]));
        let table = RecordBatch::try_from_iter([("au <ppm>", au), ("cu", cu)]).unwrap();
        let mut model = BlockModel::regular(geometry, table).unwrap();
        model.crs = Some("EPSG:31982".into());
        model
    }

    #[test]
    fn round_trip_keeps_geometry_values_nulls_and_crs() {
        let path = dir("round-trip").join("grid.tif");
        for (rotation, single, crs) in [
            (0.0, false, "EPSG:31982"),
            (0.0, true, "EPSG:4674"),
            (30.0, false, "local mine grid"),
            (215.0, true, "EPSG:31982"),
        ] {
            let mut m = model(rotation, single);
            m.crs = Some(crs.into());
            write_geotiff(&path, &m, -9999.0).unwrap();
            let back = read_geotiff(&path, None).unwrap();
            let (a, b) = (m.geometry(), back.geometry());
            assert_eq!(a.count, b.count);
            for k in 0..3 {
                assert!((a.origin[k] - b.origin[k]).abs() < 1e-6, "{a:?} {b:?}");
                assert!((a.size[k] - b.size[k]).abs() < 1e-9, "{a:?} {b:?}");
                assert!((a.rotation[k] - b.rotation[k]).abs() < 1e-9, "{a:?} {b:?}");
            }
            if rotation == 0.0 {
                assert_eq!(a, b);
            }
            let expected = if single {
                m.attributes().clone()
            } else {
                let cu = cast(m.attributes().column(1), &DataType::Float64).unwrap();
                RecordBatch::try_from_iter([
                    ("au <ppm>", m.attributes().column(0).clone()),
                    ("cu", cu),
                ])
                .unwrap()
            };
            assert_eq!(back.attributes(), &expected);
            assert_eq!(back.crs.as_deref(), Some(crs));
        }
    }

    #[test]
    fn writes_masked_and_rejects_others() {
        let path = dir("errors").join("grid.tif");
        let m = model(0.0, false);
        let masked = m
            .mask(&vec![true, false, true, true, false, false].into())
            .unwrap();
        write_geotiff(&path, &masked, -9999.0).unwrap();
        let back = read_geotiff(&path, None).unwrap();
        assert_eq!(back.attributes().column(0).null_count(), 3);
        assert!(write_geotiff(&path, &m, 2.0).is_err());

        let mut g = *m.geometry();
        g.count[2] = 2;
        g.count[1] = 1;
        let deep = BlockModel::regular(g, m.attributes().clone()).unwrap();
        assert!(write_geotiff(&path, &deep, -9999.0).is_err());
    }

    #[test]
    fn reads_compressed_planar_tiles_pixel_is_point() {
        let (w, h, tile) = (20u32, 18u32, 16u32);
        let pixel =
            |c: u32, r: u32, b: u32| (c as i16 + 100 * r as i16) * if b == 0 { 1 } else { -1 };
        let mut offsets = vec![];
        let mut counts = vec![];
        let path = dir("tiled").join("tiled.tif");
        let file = File::create(&path).unwrap();
        let mut tif = TiffEncoder::new(file).unwrap();
        let mut d = tif.image_directory().unwrap();
        for b in 0..2 {
            for tr in 0..h.div_ceil(tile) {
                for tc in 0..w.div_ceil(tile) {
                    let mut raw = vec![];
                    for r in tr * tile..(tr + 1) * tile {
                        for c in tc * tile..(tc + 1) * tile {
                            let v = if c < w && r < h { pixel(c, r, b) } else { 0 };
                            let v = if (c, r) == (3, 4) { -32768 } else { v };
                            raw.extend(v.to_le_bytes());
                        }
                    }
                    let mut packed = vec![];
                    Deflate::default().write_to(&mut packed, &raw).unwrap();
                    offsets.push(d.write_data(packed.as_slice()).unwrap() as u32);
                    counts.push(packed.len() as u32);
                }
            }
        }
        d.write_tag(Tag::PlanarConfiguration, 2u16).unwrap();
        d.write_tag(Tag::ImageWidth, w).unwrap();
        d.write_tag(Tag::ImageLength, h).unwrap();
        d.write_tag(Tag::BitsPerSample, &[16u16, 16][..]).unwrap();
        d.write_tag(Tag::Compression, 8u16).unwrap();
        d.write_tag(Tag::PhotometricInterpretation, 1u16).unwrap();
        d.write_tag(Tag::SamplesPerPixel, 2u16).unwrap();
        d.write_tag(Tag::TileWidth, tile).unwrap();
        d.write_tag(Tag::TileLength, tile).unwrap();
        d.write_tag(Tag::TileOffsets, offsets.as_slice()).unwrap();
        d.write_tag(Tag::TileByteCounts, counts.as_slice()).unwrap();
        d.write_tag(Tag::SampleFormat, &[2u16, 2][..]).unwrap();
        d.write_tag(Tag::ExtraSamples, 0u16).unwrap();
        let keys: &[u16] = &[1, 1, 0, 2, 1025, 0, 1, 2, 3072, 0, 1, 32722];
        d.write_tag(Tag::GeoKeyDirectoryTag, keys).unwrap();
        d.write_tag(Tag::ModelPixelScaleTag, &[10.0, 5.0, 0.0][..])
            .unwrap();
        d.write_tag(
            Tag::ModelTiepointTag,
            &[0.0, 0.0, 0.0, 1000.0, 2000.0, 0.0][..],
        )
        .unwrap();
        d.write_tag(Tag::GdalNodata, "-32768").unwrap();
        d.finish().unwrap();
        drop(tif);

        let m = read_geotiff(&path, None).unwrap();
        let g = m.geometry();
        assert_eq!(g.count, [20, 18, 1]);
        assert_eq!(g.size, [10.0, 5.0, 1.0]);
        assert_eq!(g.rotation, [0.0; 3]);
        assert_eq!(m.crs.as_deref(), Some("EPSG:32722"));
        let top_left = g.index([0, 17, 0]);
        assert_eq!(g.centroid(top_left), [1000.0, 2000.0, 0.5]);
        let bands = m.attributes();
        assert_eq!(bands.schema().field(1).name(), "band_2");
        let (b1, b2) = (
            bands.column(0).as_primitive::<Float64Type>(),
            bands.column(1).as_primitive::<Float64Type>(),
        );
        for r in 0..h {
            for c in 0..w {
                let row = g.index([c as usize, (h - 1 - r) as usize, 0]) as usize;
                if (c, r) == (3, 4) {
                    assert!(b1.is_null(row) && b2.is_null(row));
                } else {
                    assert_eq!(b1.value(row), f64::from(pixel(c, r, 0)));
                    assert_eq!(b2.value(row), f64::from(pixel(c, r, 1)));
                }
            }
        }
    }
}
