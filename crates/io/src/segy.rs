//! SEG-Y post-stack cubes as 3D block models.
//!
//! ```text
//! 3200 bytes   textual header (EBCDIC or ASCII)
//!  400 bytes   binary header: sample interval, samples per trace, sample format, ...
//! n x 3200     extended textual headers
//! traces       each a 240-byte header (inline, crossline, CDP x/y, ...) and its samples
//! ```

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::Arc;

use arrow_array::{ArrayRef, Float32Array, RecordBatch};
use ceres_core::{BlockModel, Geometry};

use crate::{Error, Result};

const TEXT_HEADER: u64 = 3200;
const BINARY_HEADER: usize = 400;
const TRACE_HEADER: usize = 240;

fn bad(message: impl Into<String>) -> Error {
    Error::Segy(message.into())
}

/// What [`read_segy`] names the column and where the trace headers keep the
/// line numbers and coordinates (1-based bytes of 4-byte integers).
#[derive(Debug, Clone, PartialEq)]
pub struct SegyOptions {
    pub column: String,
    pub inline_byte: usize,
    pub crossline_byte: usize,
    pub x_byte: usize,
    pub y_byte: usize,
    /// Sample value read as null.
    pub nodata: Option<f64>,
}

impl Default for SegyOptions {
    fn default() -> Self {
        Self {
            column: "amplitude".into(),
            inline_byte: 189,
            crossline_byte: 193,
            x_byte: 181,
            y_byte: 185,
            nodata: None,
        }
    }
}

/// Reads a post-stack SEG-Y cube as a regular 3D BlockModel with one
/// `Float32` column.
///
/// Formats 1 (IBM float), 2, 3, 5, 6, 8, 10, 11 and 16 in either byte order,
/// with the revision 2 extended sample count and first-trace offset. x indexes
/// inlines, y crosslines: the axes run from the smallest to the largest line
/// number in steps of the gcd of their gaps, and positions without a trace are
/// null. Cell sizes, origin and azimuth are a least-squares fit of the CDP
/// coordinates (scaled by trace byte 71) to the line indices; the azimuth is
/// that of the crossline axis, and a left-handed survey reverses the inlines
/// so the grid stays right-handed. z is minus the sample time (ms) or depth:
/// the delay of the first trace plus the sample interval divided by 1000 per
/// sample, so the first sample is the top cell. NaN samples are null.
pub fn read_segy(path: impl AsRef<Path>, options: &SegyOptions) -> Result<BlockModel> {
    for (name, byte) in [
        ("inline_byte", options.inline_byte),
        ("crossline_byte", options.crossline_byte),
        ("x_byte", options.x_byte),
        ("y_byte", options.y_byte),
    ] {
        if !(1..=TRACE_HEADER - 3).contains(&byte) {
            return Err(bad(format!(
                "{name} is {byte}; a 4-byte number starts at a byte from 1 to {} of the \
                 240-byte trace header",
                TRACE_HEADER - 3
            )));
        }
    }
    let file = File::open(path)?;
    let file_size = file.metadata()?.len();
    let mut reader = BufReader::new(file);
    if file_size < TEXT_HEADER + BINARY_HEADER as u64 {
        return Err(bad(format!(
            "the file has {file_size} bytes, fewer than the 3600 of the SEG-Y headers"
        )));
    }
    let mut binary = [0u8; BINARY_HEADER];
    reader.seek(SeekFrom::Start(TEXT_HEADER))?;
    reader.read_exact(&mut binary)?;
    let bin = |byte: usize, len: usize| &binary[byte - 3201..byte - 3201 + len];

    let order = byte_order(bin(3297, 4), bin(3225, 2)).ok_or_else(|| {
        bad(
            "the sample format code (byte 3225) is invalid in both byte orders; this is \
             not a SEG-Y file or its binary header is damaged",
        )
    })?;
    let format = order.u16(bin(3225, 2));
    let sample_size = sample_size(format).ok_or_else(|| {
        bad(format!(
            "sample format {format} is not supported; formats 1, 2, 3, 5, 6, 8, 10, 11 and 16 are"
        ))
    })?;

    let revision = bin(3501, 1)[0];
    let short_count = usize::from(order.u16(bin(3221, 2)));
    let long_count = order.i32(bin(3269, 4));
    let n_samples = if long_count > 0 && (short_count == 0 || revision >= 2) {
        long_count as usize
    } else {
        short_count
    };
    let n_extended = order.i16(bin(3505, 2));
    if n_extended < 0 {
        return Err(bad(
            "a variable number of extended textual headers is not supported",
        ));
    }
    let revision_first = if revision >= 2 {
        order.u64(bin(3521, 8))
    } else {
        0
    };
    let first_trace = if revision_first > 0 {
        revision_first
    } else {
        TEXT_HEADER + BINARY_HEADER as u64 + TEXT_HEADER * n_extended as u64
    };
    let trace_size = (TRACE_HEADER + n_samples * sample_size) as u64;
    let data_size = file_size.saturating_sub(first_trace);
    if n_samples == 0 || data_size == 0 || !data_size.is_multiple_of(trace_size) {
        return Err(bad(format!(
            "the {data_size} bytes after the headers are not a whole number of traces of \
             {n_samples} samples of {sample_size} bytes; the file is truncated or its binary \
             header is wrong"
        )));
    }
    let n_traces = usize::try_from(data_size / trace_size)
        .map_err(|_| bad("the file holds too many traces"))?;

    let mut header = [0u8; TRACE_HEADER];
    let mut first_header = [0u8; TRACE_HEADER];
    let mut traces = Vec::with_capacity(n_traces);
    for t in 0..n_traces {
        reader.seek(SeekFrom::Start(first_trace + t as u64 * trace_size))?;
        reader.read_exact(&mut header)?;
        if t == 0 {
            first_header = header;
        }
        let field = |byte: usize| order.i32(&header[byte - 1..byte + 3]);
        let scalar = order.i16(&header[70..72]);
        let xy = [options.x_byte, options.y_byte].map(|b| scale(f64::from(field(b)), scalar));
        traces.push((
            field(options.inline_byte),
            field(options.crossline_byte),
            xy,
        ));
    }
    let tr = |byte: usize| &first_header[byte - 1..byte + 1];
    let interval = match order.u16(bin(3217, 2)) {
        0 => order.u16(tr(117)),
        dt => dt,
    };
    if interval == 0 {
        return Err(bad(
            "the sample interval is 0 in the binary and trace headers",
        ));
    }
    let dz = f64::from(interval) / 1000.0;
    let delay = scale(f64::from(order.i16(tr(109))), order.i16(tr(215)));

    let (i_first, nx, i_step) = axis(traces.iter().map(|t| t.0));
    let (j_first, ny, j_step) = axis(traces.iter().map(|t| t.1));
    let nz = n_samples;
    let too_large = || {
        bad(format!(
            "the line numbers span a {nx} x {ny} x {nz} cube, which does not fit in memory; \
             check inline_byte and crossline_byte"
        ))
    };
    let n_positions = nx.checked_mul(ny).ok_or_else(too_large)?;
    let n_values = n_positions.checked_mul(nz).ok_or_else(too_large)?;
    let mut values: Vec<f32> = Vec::new();
    values
        .try_reserve_exact(n_values)
        .map_err(|_| too_large())?;
    let mut trace_at = vec![usize::MAX; n_positions];
    let mut placed = Vec::with_capacity(n_traces);
    for (t, &(il, xl, xy)) in traces.iter().enumerate() {
        let i = ((i64::from(il) - i_first) / i_step) as usize;
        let j = ((i64::from(xl) - j_first) / j_step) as usize;
        let cell = &mut trace_at[i + nx * j];
        if *cell != usize::MAX {
            return Err(bad(format!(
                "traces {} and {} are both at inline {il}, crossline {xl}; only post-stack \
                 cubes with one trace per position are read (set inline_byte and \
                 crossline_byte if the line numbers are elsewhere)",
                *cell + 1,
                t + 1
            )));
        }
        *cell = t;
        placed.push((i as f64, j as f64, xy));
    }

    let (mut p0, mut a, b) = fit(&placed)?;
    let flip = a[0] * b[1] - a[1] * b[0] < 0.0;
    if flip {
        p0 = [0, 1].map(|c| p0[c] + (nx - 1) as f64 * a[c]);
        a = a.map(|v| -v);
    }
    let azimuth = b[0].atan2(b[1]).to_degrees().rem_euclid(360.0) + 0.0;
    let geometry = Geometry {
        origin: [
            p0[0] - 0.5 * (a[0] + b[0]),
            p0[1] - 0.5 * (a[1] + b[1]),
            -(delay + (nz - 1) as f64 * dz) - 0.5 * dz,
        ],
        size: [a[0].hypot(a[1]), b[0].hypot(b[1]), dz],
        count: [nx, ny, nz],
        rotation: [if 360.0 - azimuth < 1e-9 { 0.0 } else { azimuth }, 0.0, 0.0],
    };

    values.resize(n_values, f32::NAN);
    let mut bytes = vec![0u8; nz * sample_size];
    for (t, (il, xl, _)) in traces.iter().enumerate() {
        let i = ((i64::from(*il) - i_first) / i_step) as usize;
        let j = ((i64::from(*xl) - j_first) / j_step) as usize;
        let i = if flip { nx - 1 - i } else { i };
        reader.seek(SeekFrom::Start(
            first_trace + t as u64 * trace_size + TRACE_HEADER as u64,
        ))?;
        reader.read_exact(&mut bytes)?;
        for (k, sample) in bytes.chunks_exact(sample_size).enumerate() {
            values[i + nx * (j + ny * (nz - 1 - k))] = order.sample(format, sample);
        }
    }
    let nodata = options.nodata.map(|v| v as f32);
    let column: ArrayRef = Arc::new(Float32Array::from_iter(
        values
            .into_iter()
            .map(|v| (!v.is_nan() && Some(v) != nodata).then_some(v)),
    ));
    let table = RecordBatch::try_from_iter([(options.column.as_str(), column)])?;
    Ok(BlockModel::regular(geometry, table)?)
}

/// A header number with a SEG-Y scalar: positive multiplies, negative divides.
fn scale(v: f64, scalar: i16) -> f64 {
    match scalar {
        0 => v,
        s if s < 0 => v / -f64::from(s),
        s => v * f64::from(s),
    }
}

/// First line number, number of lines and step of one axis: from the smallest
/// to the largest of `lines`, in steps of the gcd of their gaps.
fn axis(lines: impl Iterator<Item = i32>) -> (i64, usize, i64) {
    let mut sorted: Vec<i64> = lines.map(i64::from).collect();
    sorted.sort_unstable();
    sorted.dedup();
    let (min, max) = (sorted[0], sorted[sorted.len() - 1]);
    let step = sorted.windows(2).fold(0, |g, w| gcd(g, w[1] - w[0])).max(1);
    (min, ((max - min) / step + 1) as usize, step)
}

fn gcd(a: i64, b: i64) -> i64 {
    if b == 0 { a.abs() } else { gcd(b, a % b) }
}

/// Least-squares `p = p0 + i a + j b` over the traces. An axis with a single
/// line gets the other's spacing, at right angles, keeping the frame
/// right-handed; a single trace gets unit cells.
fn fit(traces: &[(f64, f64, [f64; 2])]) -> Result<([f64; 2], [f64; 2], [f64; 2])> {
    let n = traces.len() as f64;
    let mi = traces.iter().map(|t| t.0).sum::<f64>() / n;
    let mj = traces.iter().map(|t| t.1).sum::<f64>() / n;
    let mp = [0, 1].map(|c| traces.iter().map(|t| t.2[c]).sum::<f64>() / n);
    let (mut sii, mut sij, mut sjj, mut sip, mut sjp) = (0.0, 0.0, 0.0, [0.0; 2], [0.0; 2]);
    for &(i, j, p) in traces {
        let (i, j) = (i - mi, j - mj);
        sii += i * i;
        sij += i * j;
        sjj += j * j;
        for c in 0..2 {
            sip[c] += i * (p[c] - mp[c]);
            sjp[c] += j * (p[c] - mp[c]);
        }
    }
    let (a, b) = match (sii > 0.0, sjj > 0.0) {
        (true, true) => {
            let det = sii * sjj - sij * sij;
            if det <= 1e-9 * sii * sjj {
                return Err(bad("the traces lie on one diagonal line of the survey"));
            }
            (
                [0, 1].map(|c| (sjj * sip[c] - sij * sjp[c]) / det),
                [0, 1].map(|c| (sii * sjp[c] - sij * sip[c]) / det),
            )
        }
        (true, false) => {
            let a = sip.map(|v| v / sii);
            (a, [-a[1], a[0]])
        }
        (false, true) => {
            let b = sjp.map(|v| v / sjj);
            ([b[1], -b[0]], b)
        }
        (false, false) => ([1.0, 0.0], [0.0, 1.0]),
    };
    let (la, lb) = (a[0].hypot(a[1]), b[0].hypot(b[1]));
    if la == 0.0 || lb == 0.0 {
        return Err(bad(
            "the CDP coordinates do not change from line to line; check x_byte and y_byte",
        ));
    }
    if (a[0] * b[0] + a[1] * b[1]).abs() > 0.05 * la * lb {
        return Err(bad("the inline and crossline axes are not at right angles"));
    }
    let p0 = [0, 1].map(|c| mp[c] - mi * a[c] - mj * b[c]);
    Ok((p0, a, b))
}

fn sample_size(format: u16) -> Option<usize> {
    match format {
        8 | 16 => Some(1),
        3 | 11 => Some(2),
        1 | 2 | 5 | 10 => Some(4),
        6 => Some(8),
        _ => None,
    }
}

/// From the revision 2 byte-order constant (bytes 3297-3300), or else the
/// order that gives a valid sample format code.
fn byte_order(constant: &[u8], format: &[u8]) -> Option<Order> {
    match constant {
        [1, 2, 3, 4] => return Some(Order::Big),
        [4, 3, 2, 1] => return Some(Order::Little),
        _ => {}
    }
    [Order::Big, Order::Little]
        .into_iter()
        .find(|order| sample_size(order.u16(format)).is_some())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Order {
    Big,
    Little,
}

impl Order {
    fn bytes<const N: usize>(self, b: &[u8]) -> [u8; N] {
        let mut out: [u8; N] = b[..N].try_into().expect("a field of N bytes");
        if self == Order::Little {
            out.reverse();
        }
        out
    }
    fn i16(self, b: &[u8]) -> i16 {
        i16::from_be_bytes(self.bytes(b))
    }
    fn u16(self, b: &[u8]) -> u16 {
        u16::from_be_bytes(self.bytes(b))
    }
    fn i32(self, b: &[u8]) -> i32 {
        i32::from_be_bytes(self.bytes(b))
    }
    fn u64(self, b: &[u8]) -> u64 {
        u64::from_be_bytes(self.bytes(b))
    }

    fn sample(self, format: u16, b: &[u8]) -> f32 {
        match format {
            1 => ibm_to_f32(u32::from_be_bytes(self.bytes(b))),
            2 => self.i32(b) as f32,
            3 => f32::from(self.i16(b)),
            5 => f32::from_be_bytes(self.bytes(b)),
            6 => f64::from_be_bytes(self.bytes(b)) as f32,
            8 => f32::from(b[0] as i8),
            10 => u32::from_be_bytes(self.bytes(b)) as f32,
            11 => f32::from(self.u16(b)),
            16 => f32::from(b[0]),
            _ => unreachable!("sample_size admits only the formats above"),
        }
    }
}

/// IBM System/360 single float: sign, 7-bit base-16 exponent biased by 64,
/// 24-bit fraction.
fn ibm_to_f32(bits: u32) -> f32 {
    let sign = if bits >> 31 == 0 { 1.0 } else { -1.0 };
    let exponent = ((bits >> 24) & 0x7f) as i32 - 64;
    let fraction = f64::from(bits & 0x00ff_ffff) / f64::from(1u32 << 24);
    (sign * fraction * 16f64.powi(exponent)) as f32
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use arrow_array::Array;
    use arrow_array::cast::AsArray;
    use arrow_array::types::Float32Type;

    use super::*;

    fn temp(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("ceres-segy-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    /// One trace per `positions` entry, sample k of trace t being `value(t, k)`,
    /// CDP coordinates from `xy`, stored with coordinate scalar -100.
    struct Survey {
        format: u16,
        order: Order,
        n_samples: usize,
        n_extended: i16,
        interval_us: u16,
        delay_ms: i16,
        positions: Vec<(i32, i32)>,
        xy: fn(i32, i32) -> [f64; 2],
    }

    impl Survey {
        fn new(positions: Vec<(i32, i32)>) -> Self {
            Self {
                format: 5,
                order: Order::Big,
                n_samples: 4,
                n_extended: 0,
                interval_us: 4000,
                delay_ms: 0,
                positions,
                xy: |il, xl| [1000.0 + 25.0 * f64::from(il), 2000.0 + 12.5 * f64::from(xl)],
            }
        }

        fn value(t: usize, k: usize) -> f32 {
            (t * 10 + k) as f32
        }

        fn put(&self, out: &mut [u8], at: usize, bytes: &[u8]) {
            let mut bytes = bytes.to_vec();
            if self.order == Order::Little {
                bytes.reverse();
            }
            out[at..at + bytes.len()].copy_from_slice(&bytes);
        }

        fn write(&self, name: &str) -> PathBuf {
            let mut data = vec![b' '; 3200];
            data.resize(3600, 0);
            self.put(&mut data, 3216, &self.interval_us.to_be_bytes());
            self.put(&mut data, 3220, &(self.n_samples as u16).to_be_bytes());
            self.put(&mut data, 3224, &self.format.to_be_bytes());
            self.put(&mut data, 3504, &self.n_extended.to_be_bytes());
            data.resize(data.len() + 3200 * self.n_extended as usize, b' ');
            for (t, &(il, xl)) in self.positions.iter().enumerate() {
                let mut header = vec![0u8; 240];
                let [x, y] = (self.xy)(il, xl).map(|v| (v * 100.0).round() as i32);
                self.put(&mut header, 70, &(-100i16).to_be_bytes());
                self.put(&mut header, 108, &self.delay_ms.to_be_bytes());
                self.put(&mut header, 180, &x.to_be_bytes());
                self.put(&mut header, 184, &y.to_be_bytes());
                self.put(&mut header, 188, &il.to_be_bytes());
                self.put(&mut header, 192, &xl.to_be_bytes());
                data.extend(header);
                for k in 0..self.n_samples {
                    let v = Self::value(t, k);
                    let be: Vec<u8> = match self.format {
                        1 => f32_to_ibm(v).to_be_bytes().to_vec(),
                        2 => (v as i32).to_be_bytes().to_vec(),
                        3 => (v as i16).to_be_bytes().to_vec(),
                        5 => v.to_be_bytes().to_vec(),
                        6 => f64::from(v).to_be_bytes().to_vec(),
                        8 => vec![v as i8 as u8],
                        10 => (v as u32).to_be_bytes().to_vec(),
                        11 => (v as u16).to_be_bytes().to_vec(),
                        16 => vec![v as u8],
                        _ => unreachable!(),
                    };
                    let at = data.len();
                    data.resize(at + be.len(), 0);
                    self.put(&mut data, at, &be);
                }
            }
            let path = temp(name);
            std::fs::write(&path, data).unwrap();
            path
        }
    }

    /// Exact for the small integers of the tests.
    fn f32_to_ibm(v: f32) -> u32 {
        if v == 0.0 {
            return 0;
        }
        let sign = u32::from(v < 0.0) << 31;
        let (mut fraction, mut exponent) = (f64::from(v.abs()), 64u32);
        while fraction >= 1.0 {
            fraction /= 16.0;
            exponent += 1;
        }
        while fraction < 1.0 / 16.0 {
            fraction *= 16.0;
            exponent -= 1;
        }
        sign | (exponent << 24) | (fraction * f64::from(1u32 << 24)) as u32
    }

    /// Inline-sorted positions, crossline fastest.
    fn grid(ils: &[i32], xls: &[i32]) -> Vec<(i32, i32)> {
        ils.iter()
            .flat_map(|&il| xls.iter().map(move |&xl| (il, xl)))
            .collect()
    }

    fn read(path: &Path) -> BlockModel {
        read_segy(path, &SegyOptions::default()).unwrap()
    }

    /// Sample `k` (0 = first) of the trace at inline index `i`, crossline `j`.
    fn at(m: &BlockModel, i: usize, j: usize, k: usize) -> Option<f32> {
        let g = m.geometry();
        let row = g.index([i, j, g.count[2] - 1 - k]) as usize;
        let c = m.attributes().column(0).as_primitive::<Float32Type>();
        c.is_valid(row).then(|| c.value(row))
    }

    fn message(e: Error) -> String {
        e.to_string()
    }

    #[test]
    fn ibm_floats_convert_to_ieee() {
        assert_eq!(ibm_to_f32(0xC276_A000), -118.625);
        assert_eq!(ibm_to_f32(0x4110_0000), 1.0);
        assert_eq!(ibm_to_f32(0x4264_0000), 100.0);
        assert_eq!(ibm_to_f32(0x3F40_0000), 0.015625);
        assert_eq!(ibm_to_f32(0), 0.0);
    }

    #[test]
    fn reads_every_format_in_both_byte_orders() {
        for format in [1, 2, 3, 5, 6, 8, 10, 11, 16] {
            for order in [Order::Big, Order::Little] {
                let mut survey = Survey::new(grid(&[1, 2, 3], &[10, 11]));
                survey.format = format;
                survey.order = order;
                let m = read(&survey.write(&format!("format-{format}-{order:?}.sgy")));
                assert_eq!(m.geometry().count, [3, 2, 4]);
                let c = m.attributes().column(0);
                assert_eq!(c.data_type(), &arrow_schema::DataType::Float32);
                assert_eq!(at(&m, 2, 1, 3), Some(Survey::value(5, 3)), "{format}");
                assert_eq!(at(&m, 1, 0, 2), Some(Survey::value(2, 2)), "{format}");
            }
        }
    }

    #[test]
    fn geometry_from_line_numbers_coordinates_and_times() {
        let mut survey = Survey::new(grid(&[100, 102, 104], &[7, 8]));
        survey.delay_ms = 20;
        survey.interval_us = 2000;
        survey.n_extended = 2;
        let m = read(&survey.write("axes.sgy"));
        let g = m.geometry();
        assert_eq!(g.count, [3, 2, 4]);
        assert_eq!(m.attributes().schema().field(0).name(), "amplitude");
        for (got, want) in g.size.iter().zip([50.0, 12.5, 2.0]) {
            assert!((got - want).abs() < 1e-9, "{g:?}");
        }
        assert_eq!(g.rotation, [0.0; 3]);
        let top = g.centroid(g.index([0, 0, 3]));
        let want = [1000.0 + 2500.0, 2000.0 + 87.5, -20.0];
        for c in 0..3 {
            assert!((top[c] - want[c]).abs() < 1e-9, "{top:?}");
        }
        assert!((g.centroid(0)[2] + 26.0).abs() < 1e-9);
    }

    #[test]
    fn rotated_and_left_handed_surveys() {
        let mut survey = Survey::new(grid(&[1, 2, 3], &[1, 2, 3, 4]));
        survey.xy = |il, xl| {
            let (s, c) = 30f64.to_radians().sin_cos();
            let (i, j) = (f64::from(il), f64::from(xl));
            [
                5e5 + 25.0 * i * c + 12.5 * j * s,
                7e6 - 25.0 * i * s + 12.5 * j * c,
            ]
        };
        let right = read(&survey.write("right.sgy"));
        survey.xy = |il, xl| {
            let (s, c) = 30f64.to_radians().sin_cos();
            let (i, j) = (f64::from(il), f64::from(xl));
            [
                5e5 - 25.0 * i * c + 12.5 * j * s,
                7e6 + 25.0 * i * s + 12.5 * j * c,
            ]
        };
        let left = read(&survey.write("left.sgy"));
        for m in [&right, &left] {
            let g = m.geometry();
            assert!((g.rotation[0] - 30.0).abs() < 1e-2, "{g:?}");
            assert!((g.size[0] - 25.0).abs() < 1e-2 && (g.size[1] - 12.5).abs() < 1e-2);
        }
        assert_eq!(at(&right, 0, 0, 0), Some(Survey::value(0, 0)));
        assert_eq!(at(&left, 2, 0, 0), Some(Survey::value(0, 0)));
        let g = left.geometry();
        let first = g.centroid(g.index([2, 0, 3]));
        let want = (survey.xy)(1, 1);
        assert!((first[0] - want[0]).abs() < 0.01 && (first[1] - want[1]).abs() < 0.01);
    }

    #[test]
    fn crossline_sorted_file_lands_on_the_same_cells() {
        let inline_sorted = read(&Survey::new(grid(&[1, 2, 3], &[5, 6])).write("il.sgy"));
        let positions = [5, 6]
            .iter()
            .flat_map(|&xl| [1, 2, 3].map(|il| (il, xl)))
            .collect();
        let crossline_sorted = read(&Survey::new(positions).write("xl.sgy"));
        assert_eq!(at(&inline_sorted, 1, 1, 0), Some(Survey::value(3, 0)));
        assert_eq!(at(&crossline_sorted, 1, 1, 0), Some(Survey::value(4, 0)));
        assert_eq!(inline_sorted.geometry(), crossline_sorted.geometry());
    }

    #[test]
    fn missing_traces_and_nodata_are_null() {
        let survey = Survey::new(vec![(1, 10), (1, 11), (2, 10), (4, 10), (4, 11)]);
        let path = survey.write("missing.sgy");
        let m = read(&path);
        assert_eq!(m.geometry().count, [4, 2, 4]);
        assert_eq!(at(&m, 3, 0, 1), Some(Survey::value(3, 1)));
        assert_eq!(at(&m, 1, 1, 0), None);
        assert_eq!(at(&m, 2, 0, 3), None);
        let options = SegyOptions {
            column: "vp".into(),
            nodata: Some(11.0),
            ..SegyOptions::default()
        };
        let m = read_segy(&path, &options).unwrap();
        assert_eq!(m.attributes().schema().field(0).name(), "vp");
        assert_eq!(at(&m, 0, 1, 1), None);
        assert_eq!(at(&m, 0, 1, 0), Some(10.0));
    }

    #[test]
    fn two_traces_at_one_position_are_an_error() {
        let survey = Survey::new(vec![(1, 1), (1, 1), (1, 2), (1, 2)]);
        let err = read_segy(survey.write("prestack.sgy"), &SegyOptions::default()).unwrap_err();
        assert!(message(err).contains("post-stack"));
    }

    #[test]
    fn truncated_file_is_an_error() {
        let path = Survey::new(grid(&[1, 2], &[1, 2])).write("truncated.sgy");
        let data = std::fs::read(&path).unwrap();
        std::fs::write(&path, &data[..data.len() - 3]).unwrap();
        let err = read_segy(&path, &SegyOptions::default()).unwrap_err();
        assert!(message(err).contains("truncated"));
    }

    fn with_format(path: &Path, code: [u8; 2], constant: bool) {
        let mut data = std::fs::read(path).unwrap();
        data[3224..3226].copy_from_slice(&code);
        if constant {
            data[3296..3300].copy_from_slice(&[1, 2, 3, 4]);
        }
        std::fs::write(path, data).unwrap();
    }

    #[test]
    fn unsupported_format_is_named() {
        let path = Survey::new(grid(&[1], &[1])).write("format-4.sgy");
        with_format(&path, [0, 4], true);
        let err = read_segy(&path, &SegyOptions::default()).unwrap_err();
        assert!(message(err).contains("format 4 is not supported"));
    }

    #[test]
    fn invalid_format_in_both_byte_orders_is_not_segy() {
        let path = Survey::new(grid(&[1], &[1])).write("format-bad.sgy");
        with_format(&path, [4, 4], false);
        let err = read_segy(&path, &SegyOptions::default()).unwrap_err();
        assert!(message(err).contains("not a SEG-Y"));
    }

    #[test]
    fn not_segy_is_an_error() {
        let path = temp("not.sgy");
        std::fs::write(&path, "3 2 1\n1\nfacies\n0\n").unwrap();
        let err = read_segy(&path, &SegyOptions::default()).unwrap_err();
        assert!(message(err).contains("SEG-Y"));
        let options = SegyOptions {
            x_byte: 239,
            ..SegyOptions::default()
        };
        assert!(read_segy(&path, &options).is_err());
    }
}
