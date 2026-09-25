//! Visible-face extraction: the skin of a block model instead of every block.
//!
//! Drawing a block model as one cube per block costs `O(n³)` geometry for an
//! `n³` model, which is what makes a real mine-scale model unrenderable — a
//! 500×500×200 model is 50M cells, 600M triangles. Almost none of it is
//! visible: only the faces on the boundary of the *rendered set* are, and
//! those grow with surface area, `O(n²)`. The same 50M-cell model has a skin
//! of roughly 900k faces.
//!
//! A face is emitted when the cell across it is not rendered — outside the
//! model, absent from a sparse (ore-only) export, filtered out by the active
//! slab, or null in the attribute being coloured by. That last part is what
//! makes a section cheap *and* correct: extracting the shell of the blocks
//! inside a slab produces the cut faces, where clipping a hollow shell on the
//! GPU would show it hollow.
//!
//! Blocks arrive as axis-aligned boxes with no topology (a CSV block model is
//! just rows of centroid + size), so the sweep first infers the lattice they
//! sit on — the minimum block size per axis, anchored at the model's minimum
//! corner. A uniform model maps one block to one cell; a sub-blocked model
//! maps a parent block to the span of cells it covers. Models whose blocks
//! don't land on any such lattice are rejected rather than silently
//! mis-rendered.

use rayon::prelude::*;

use crate::error::{BlockModelError, Result};

/// One block as the sweep needs it: an axis-aligned box. Deliberately not
/// `miningio::Block` — this crate stays free of the I/O layer (see `solid.rs`,
/// which keeps its own `BlockSolid` for the same reason).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShellBlock {
    pub centroid: [f64; 3],
    pub size: [f64; 3],
}

/// A plane-bounded slab: every block whose centroid lies within
/// `thickness / 2` of the plane is kept. Matches `pointInSlab` in
/// `src/lib/slicing.ts`, so a section looks the same whether it was cut here
/// or client-side.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Slab {
    pub origin: [f64; 3],
    pub normal: [f64; 3],
    pub thickness: f64,
}

impl Slab {
    fn contains(&self, centroid: &[f64; 3]) -> bool {
        let d = (centroid[0] - self.origin[0]) * self.normal[0]
            + (centroid[1] - self.origin[1]) * self.normal[1]
            + (centroid[2] - self.origin[2]) * self.normal[2];
        d.abs() <= self.thickness / 2.0
    }
}

/// How the block grid is rotated relative to the world axes.
///
/// A block's `size` is measured along the grid's own axes, and the sweep below
/// works on an axis-aligned lattice, so a rotated model has to be swept in grid
/// space and its faces rotated back out. Without this, a rotation of even one
/// degree puts nearly every block off the lattice — the tolerance is a fraction
/// of a cell — and the model can't be shelled at all.
///
/// Stored as the grid's own axis directions in world space, which is the shape
/// `miningio::GridOrientation` already carries. Pure rotation: the grid's
/// position is absorbed by the lattice's own minimum corner, so no origin is
/// needed here.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Orientation {
    u: [f64; 3],
    v: [f64; 3],
    w: [f64; 3],
}

impl Default for Orientation {
    fn default() -> Self {
        Self::IDENTITY
    }
}

fn dot(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

impl Orientation {
    /// The world axes — an unrotated grid.
    pub const IDENTITY: Orientation = Orientation {
        u: [1.0, 0.0, 0.0],
        v: [0.0, 1.0, 0.0],
        w: [0.0, 0.0, 1.0],
    };

    /// A grid rotated about +Z by `radians`, measured from world +X towards +Y.
    /// This is the mining case: a bearing, with benches still level.
    pub fn from_azimuth(radians: f64) -> Self {
        let (sin, cos) = radians.sin_cos();
        Orientation {
            u: [cos, sin, 0.0],
            v: [-sin, cos, 0.0],
            w: [0.0, 0.0, 1.0],
        }
    }

    /// A grid whose axes are `u`/`v`/`w` in world space. The axes must be
    /// orthonormal and right-handed — anything else would not be a rotation,
    /// and shearing or scaling the lattice is not something the sweep can
    /// represent.
    pub fn from_axes(u: [f64; 3], v: [f64; 3], w: [f64; 3]) -> Result<Self> {
        let invalid = |what: &str| {
            Err(BlockModelError::InvalidGridParams(format!(
                "block model's grid orientation is not a rotation ({what})"
            )))
        };
        for axis in [&u, &v, &w] {
            if !axis.iter().all(|c| c.is_finite()) {
                return invalid("non-finite axis");
            }
            if (dot(axis, axis) - 1.0).abs() > 1e-6 {
                return invalid("axis is not unit length");
            }
        }
        if dot(&u, &v).abs() > 1e-6 || dot(&u, &w).abs() > 1e-6 || dot(&v, &w).abs() > 1e-6 {
            return invalid("axes are not perpendicular");
        }
        // u × v must be w, or the frame is left-handed and every emitted face
        // would be wound inside-out.
        let cross = [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ];
        if dot(&cross, &w) < 0.0 {
            return invalid("axes are left-handed");
        }
        Ok(Orientation { u, v, w })
    }

    pub fn is_identity(&self) -> bool {
        *self == Self::IDENTITY
    }

    /// Rotation about +Z this orientation represents, radians. Reporting only.
    pub fn azimuth(&self) -> f64 {
        self.u[1].atan2(self.u[0])
    }

    /// World point -> grid space.
    fn to_local(self, p: [f64; 3]) -> [f64; 3] {
        [dot(&self.u, &p), dot(&self.v, &p), dot(&self.w, &p)]
    }

    /// Grid space -> world point.
    fn to_world(self, p: [f64; 3]) -> [f64; 3] {
        [
            self.u[0] * p[0] + self.v[0] * p[1] + self.w[0] * p[2],
            self.u[1] * p[0] + self.v[1] * p[1] + self.w[1] * p[2],
            self.u[2] * p[0] + self.v[2] * p[1] + self.w[2] * p[2],
        ]
    }

    /// The block as the sweep sees it: centroid in grid space, size unchanged
    /// (a block's size is already measured along the grid's own axes).
    fn localize(&self, block: &ShellBlock) -> ShellBlock {
        ShellBlock {
            centroid: self.to_local(block.centroid),
            size: block.size,
        }
    }
}

/// Which blocks take part in the shell. Everything filtered out here counts as
/// empty space, so its neighbours grow a face — that is how a slab produces
/// cut faces and a null-valued region produces a hole.
#[derive(Debug, Clone, Default)]
pub struct ShellFilter {
    /// The live cuts. A block is kept when it lies inside **any** of them, so a
    /// plan level and a vertical section shell *together* — the union the
    /// viewport draws — rather than intersecting down to where they cross. An
    /// empty list keeps every block.
    pub slabs: Vec<Slab>,
    /// Drop blocks whose colour-by value is NaN (a null grade), so they render
    /// as absent rather than as a hole-coloured cube.
    pub drop_nan_values: bool,
    /// A live selection filter (#349): only these block rows are kept, ANDed
    /// with the cuts above. `None` keeps every block.
    pub keep: Option<BlockSubset>,
}

/// A subset of block rows as a bitmask, indexed by the model's own row order —
/// the same order `extract_shell`'s iterator yields blocks in. One bit per
/// block, so a multi-million-row selection costs len/8 bytes rather than a
/// hash set.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BlockSubset {
    words: Vec<u64>,
}

impl BlockSubset {
    pub fn from_indices(indices: impl IntoIterator<Item = usize>, len: usize) -> Self {
        let mut words = vec![0u64; len.div_ceil(64)];
        for i in indices {
            if i < len {
                words[i / 64] |= 1 << (i % 64);
            }
        }
        Self { words }
    }

    pub fn contains(&self, index: usize) -> bool {
        self.words
            .get(index / 64)
            .is_some_and(|w| w & (1 << (index % 64)) != 0)
    }
}

/// Ceilings that turn a pathological model into an error instead of an
/// out-of-memory kill. Both are generous for real models and small enough to
/// stay inside a browser's GPU budget.
#[derive(Debug, Clone, Copy)]
pub struct ShellLimits {
    /// Cells in the inferred lattice. Guards a sparse model whose bounding box
    /// dwarfs its block count (the lattice is one bit per cell).
    pub max_cells: usize,
    /// Faces in the emitted skin.
    pub max_faces: usize,
}

impl Default for ShellLimits {
    fn default() -> Self {
        // 400M cells is 50 MB of occupancy bits; 4M faces is 16M triangles,
        // which a mid-range GPU draws comfortably.
        Self {
            max_cells: 400_000_000,
            max_faces: 4_000_000,
        }
    }
}

/// The extracted skin: a triangle mesh in the same shape the viewport's mesh
/// payload already takes, plus one value per vertex when the caller asked for
/// an attribute (each face carries its own block's value, so the value is
/// repeated across that face's four corners).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ShellMesh {
    /// Flat xyz, 3 per vertex.
    pub vertices: Vec<f64>,
    /// Flat triangle indices, 3 per triangle.
    pub triangles: Vec<u32>,
    /// One value per vertex, or empty when no attribute was requested.
    pub values: Vec<f64>,
    /// Blocks that passed the filter — what the shell was extracted from.
    pub blocks_kept: usize,
}

impl ShellMesh {
    pub fn face_count(&self) -> usize {
        self.triangles.len() / 6
    }
}

/// The lattice the blocks were found to sit on: the smallest cell size per
/// axis, anchored at the model's minimum corner.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Lattice {
    min: [f64; 3],
    cell: [f64; 3],
    counts: [usize; 3],
}

impl Lattice {
    fn cell_total(&self) -> usize {
        self.counts[0] * self.counts[1] * self.counts[2]
    }

    fn index(&self, i: usize, j: usize, k: usize) -> usize {
        (k * self.counts[1] + j) * self.counts[0] + i
    }

    /// Node position: the lattice corner at integer coordinates.
    fn node(&self, i: usize, j: usize, k: usize) -> [f64; 3] {
        [
            self.min[0] + i as f64 * self.cell[0],
            self.min[1] + j as f64 * self.cell[1],
            self.min[2] + k as f64 * self.cell[2],
        ]
    }
}

/// The half-open cell span `[lo, hi)` a block covers on each axis.
#[derive(Debug, Clone, Copy)]
struct Span {
    lo: [usize; 3],
    hi: [usize; 3],
}

/// How far off the lattice a block corner may sit before the model is judged
/// irregular — a fraction of the cell, so it scales with block size.
const ALIGN_TOLERANCE: f64 = 0.02;

fn infer_lattice<I>(blocks: I, limits: &ShellLimits) -> Result<Lattice>
where
    I: Iterator<Item = ShellBlock>,
{
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    let mut cell = [f64::INFINITY; 3];
    let mut count = 0usize;

    for block in blocks {
        count += 1;
        for axis in 0..3 {
            let half = block.size[axis] / 2.0;
            if !block.size[axis].is_finite() || block.size[axis] <= 0.0 {
                return Err(BlockModelError::InvalidGridParams(
                    "block model has a block with a non-positive size".to_string(),
                ));
            }
            min[axis] = min[axis].min(block.centroid[axis] - half);
            max[axis] = max[axis].max(block.centroid[axis] + half);
            cell[axis] = cell[axis].min(block.size[axis]);
        }
    }

    if count == 0 {
        return Err(BlockModelError::InvalidGridParams(
            "block model is empty".to_string(),
        ));
    }

    let mut counts = [0usize; 3];
    for axis in 0..3 {
        let extent = max[axis] - min[axis];
        // Round rather than ceil: an exact grid divides evenly, and rounding
        // absorbs the float error that accumulates across a 1e6-magnitude
        // mine grid. The max(1) keeps a single-layer model (a bench plan,
        // say) one cell deep instead of zero.
        counts[axis] = ((extent / cell[axis]).round() as usize).max(1);
    }

    let total = counts[0]
        .checked_mul(counts[1])
        .and_then(|n| n.checked_mul(counts[2]))
        .ok_or_else(|| {
            BlockModelError::InvalidGridParams(
                "block model's lattice is too large to index".to_string(),
            )
        })?;
    if total > limits.max_cells {
        return Err(BlockModelError::InvalidGridParams(format!(
            "block model spans {total} lattice cells, over the {} cell limit — \
             its blocks are far smaller than its extent",
            limits.max_cells
        )));
    }

    Ok(Lattice { min, cell, counts })
}

/// The cells a block covers, or `None` when it doesn't sit on the lattice.
fn span_of(lattice: &Lattice, block: &ShellBlock) -> Option<Span> {
    let mut lo = [0usize; 3];
    let mut hi = [0usize; 3];
    for axis in 0..3 {
        let start = (block.centroid[axis] - block.size[axis] / 2.0 - lattice.min[axis])
            / lattice.cell[axis];
        let width = block.size[axis] / lattice.cell[axis];
        let start_round = start.round();
        let width_round = width.round();
        if (start - start_round).abs() > ALIGN_TOLERANCE
            || (width - width_round).abs() > ALIGN_TOLERANCE
            || start_round < 0.0
            || width_round < 1.0
        {
            return None;
        }
        lo[axis] = start_round as usize;
        hi[axis] = (start_round + width_round) as usize;
        if hi[axis] > lattice.counts[axis] {
            hi[axis] = lattice.counts[axis];
        }
        if lo[axis] >= hi[axis] {
            return None;
        }
    }
    Some(Span { lo, hi })
}

/// A one-bit-per-cell occupancy map. A block model at mine scale has tens of
/// millions of cells; a `Vec<bool>` would be eight times this.
struct Occupancy(Vec<u64>);

impl Occupancy {
    fn new(cells: usize) -> Self {
        Occupancy(vec![0u64; cells.div_ceil(64)])
    }

    fn set(&mut self, index: usize) {
        self.0[index / 64] |= 1u64 << (index % 64);
    }

    fn get(&self, index: usize) -> bool {
        self.0[index / 64] & (1u64 << (index % 64)) != 0
    }
}

/// Whether every cell across `block`'s face on `axis` (`positive` side) is
/// occupied. If any isn't, that face is on the skin and must be drawn.
fn face_is_buried(
    lattice: &Lattice,
    occupancy: &Occupancy,
    span: &Span,
    axis: usize,
    positive: bool,
) -> bool {
    let neighbour = if positive {
        if span.hi[axis] >= lattice.counts[axis] {
            return false; // model boundary
        }
        span.hi[axis]
    } else {
        if span.lo[axis] == 0 {
            return false;
        }
        span.lo[axis] - 1
    };

    // Sweep the other two axes across the whole face — a sub-blocked parent
    // touches several neighbours, and one gap is enough to expose it.
    let (a, b) = match axis {
        0 => (1, 2),
        1 => (0, 2),
        _ => (0, 1),
    };
    for x in span.lo[a]..span.hi[a] {
        for y in span.lo[b]..span.hi[b] {
            let mut cell = [0usize; 3];
            cell[axis] = neighbour;
            cell[a] = x;
            cell[b] = y;
            if !occupancy.get(lattice.index(cell[0], cell[1], cell[2])) {
                return false;
            }
        }
    }
    true
}

/// Appends one quad, wound counter-clockwise seen from outside so the normal
/// points away from the block.
fn push_face(mesh: &mut ShellMesh, corners: [[f64; 3]; 4], value: Option<f64>) {
    let base = (mesh.vertices.len() / 3) as u32;
    for corner in corners {
        mesh.vertices.extend_from_slice(&corner);
        if let Some(v) = value {
            mesh.values.push(v);
        }
    }
    mesh.triangles
        .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
}

/// Emits the exposed face of `span` on `axis`/`positive`.
#[allow(clippy::too_many_arguments)]
fn emit_face(
    mesh: &mut ShellMesh,
    lattice: &Lattice,
    orientation: &Orientation,
    span: &Span,
    axis: usize,
    positive: bool,
    value: Option<f64>,
) {
    let plane = if positive {
        span.hi[axis]
    } else {
        span.lo[axis]
    };
    // The two in-face axes, ordered so (a, b, axis) is right-handed.
    let (a, b) = match axis {
        0 => (1, 2),
        1 => (2, 0),
        _ => (0, 1),
    };

    // The lattice is in grid space; the mesh is in world space. A pure
    // rotation preserves winding, so the normals still point outwards.
    let corner = |ia: usize, ib: usize| -> [f64; 3] {
        let mut node = [0usize; 3];
        node[axis] = plane;
        node[a] = ia;
        node[b] = ib;
        orientation.to_world(lattice.node(node[0], node[1], node[2]))
    };

    let (a0, a1) = (span.lo[a], span.hi[a]);
    let (b0, b1) = (span.lo[b], span.hi[b]);
    let quad = [
        corner(a0, b0),
        corner(a1, b0),
        corner(a1, b1),
        corner(a0, b1),
    ];
    // Reversed on the negative side: the same winding seen from the far side
    // of the block would point its normal inwards.
    if positive {
        push_face(mesh, quad, value);
    } else {
        push_face(mesh, [quad[0], quad[3], quad[2], quad[1]], value);
    }
}

/// Extracts the visible skin of a block model.
///
/// `blocks` is called once per pass (three passes: infer the lattice, mark
/// occupancy, emit faces) so a model stored implicitly — a regular grid that
/// computes its blocks on demand — never has to be materialized.
/// `values`, when given, is one value per block in the same order the
/// iterator yields them.
pub fn extract_shell<F, I>(
    blocks: F,
    values: Option<&[f64]>,
    filter: &ShellFilter,
    limits: &ShellLimits,
    orientation: &Orientation,
) -> Result<ShellMesh>
where
    F: Fn() -> I,
    I: Iterator<Item = ShellBlock>,
{
    let keeps = |index: usize, block: &ShellBlock| -> bool {
        if let Some(keep) = &filter.keep
            && !keep.contains(index)
        {
            return false;
        }
        if !filter.slabs.is_empty()
            && !filter
                .slabs
                .iter()
                .any(|slab| slab.contains(&block.centroid))
        {
            return false;
        }
        if filter.drop_nan_values
            && let Some(v) = values.and_then(|v| v.get(index))
            && v.is_nan()
        {
            return false;
        }
        true
    };

    // The lattice covers the whole model, not just the kept blocks: a slab
    // must not shift the grid the blocks are indexed against. It is built in
    // GRID space — for a rotated model the world-space centroids sit on no
    // axis-aligned lattice at all (see `Orientation`).
    let lattice = infer_lattice(blocks().map(|b| orientation.localize(&b)), limits)?;

    let mut occupancy = Occupancy::new(lattice.cell_total());
    let mut irregular = 0usize;
    let mut kept = 0usize;
    for (index, block) in blocks().enumerate() {
        if !keeps(index, &block) {
            continue;
        }
        match span_of(&lattice, &orientation.localize(&block)) {
            Some(span) => {
                kept += 1;
                for i in span.lo[0]..span.hi[0] {
                    for j in span.lo[1]..span.hi[1] {
                        for k in span.lo[2]..span.hi[2] {
                            occupancy.set(lattice.index(i, j, k));
                        }
                    }
                }
            }
            None => irregular += 1,
        }
    }

    // A handful of stray blocks is a data quirk; a model that is mostly
    // off-lattice isn't a block model the shell can describe, and drawing a
    // shell of the rest would silently hide data.
    if irregular > 0 && irregular * 20 > kept + irregular {
        return Err(BlockModelError::InvalidGridParams(format!(
            "{irregular} of {} blocks don't sit on a regular lattice; \
             this model can't be drawn as a shell",
            kept + irregular
        )));
    }

    let mut mesh = ShellMesh {
        blocks_kept: kept,
        ..Default::default()
    };

    // Face emission is the expensive pass — per block it localizes, spans, and
    // tests six faces against the occupancy bitmap. It parallelises because
    // `push_face` never dedups against earlier vertices: every face appends
    // four fresh ones and indexes them from the running base. So a chunk can
    // build a mesh from base 0 and be folded in afterwards by offsetting its
    // triangle indices.
    //
    // Per the workspace manifest, the map is parallel and the fold is
    // sequential: `par_chunks(..).collect()` is an *indexed* rayon iterator, so
    // the parts come back in chunk order and the concatenation below is
    // bit-for-bit identical to the sequential emission whatever the thread
    // count. Never fold in completion order.
    let mut buffer: Vec<(usize, ShellBlock)> = Vec::with_capacity(EMIT_BUFFER);
    let mut source = blocks().enumerate();
    loop {
        buffer.clear();
        buffer.extend(source.by_ref().take(EMIT_BUFFER));
        if buffer.is_empty() {
            break;
        }

        let parts: Vec<ShellMesh> = buffer
            .par_chunks(EMIT_CHUNK)
            .map(|chunk| {
                let mut local = ShellMesh::default();
                for (index, block) in chunk {
                    if !keeps(*index, block) {
                        continue;
                    }
                    let Some(span) = span_of(&lattice, &orientation.localize(block)) else {
                        continue;
                    };
                    let value = values.and_then(|v| v.get(*index)).copied();
                    for axis in 0..3 {
                        for positive in [false, true] {
                            if face_is_buried(&lattice, &occupancy, &span, axis, positive) {
                                continue;
                            }
                            emit_face(
                                &mut local,
                                &lattice,
                                orientation,
                                &span,
                                axis,
                                positive,
                                value,
                            );
                        }
                    }
                }
                local
            })
            .collect();

        for part in parts {
            append_mesh(&mut mesh, part);
        }

        // Checked once per buffer rather than per face. The sequential version
        // stopped at the first face past the limit; this one may emit up to a
        // buffer's worth beyond it before returning. Same error, same
        // threshold, bounded extra work.
        if mesh.face_count() > limits.max_faces {
            return Err(BlockModelError::InvalidGridParams(format!(
                "block model's shell exceeds {} faces — narrow it with a \
                 slab or an attribute cutoff",
                limits.max_faces
            )));
        }
    }

    Ok(mesh)
}

/// Blocks drained from the iterator before each parallel round. Bounds peak
/// memory to `EMIT_BUFFER * size_of::<ShellBlock>()` (~3 MB) rather than
/// materializing the whole model, which at 100M blocks would be ~4.8 GB.
const EMIT_BUFFER: usize = 65_536;

/// Blocks per rayon task. Small enough to keep every core fed on a buffer,
/// large enough that per-task mesh allocation is not the dominant cost.
const EMIT_CHUNK: usize = 1_024;

/// Concatenates `src` onto `dst`, rebasing its triangle indices. Mirrors
/// [`push_face`]'s layout: vertices are flat xyz, values are one per vertex,
/// and triangles index vertices.
fn append_mesh(dst: &mut ShellMesh, src: ShellMesh) {
    let base = (dst.vertices.len() / 3) as u32;
    dst.vertices.extend_from_slice(&src.vertices);
    dst.values.extend_from_slice(&src.values);
    dst.triangles
        .extend(src.triangles.iter().map(|index| index + base));
}

/// Blocks sampled when guessing a model's rotation. A grid gives itself away
/// in a few thousand centroids; sampling keeps detection independent of model
/// size (this runs on models with tens of millions of blocks).
const ORIENTATION_SAMPLE: usize = 4096;

/// Minimum alignment score to accept a guessed rotation.
///
/// Tied to [`ALIGN_TOLERANCE`]: if every block sits within that tolerance of a
/// lattice node, the phase spread below is at most TAU * 0.02, and the mean
/// resultant of a spread that tight is ~0.997. Demanding 0.999 therefore means
/// "tighter than the sweep itself requires" — which is the point, since an
/// angle the sweep would go on to reject is no use to anyone.
const ORIENTATION_MIN_SCORE: f64 = 0.999;

/// How tightly the sampled blocks land on a lattice of `cell` along one axis,
/// ignoring where the lattice starts.
///
/// Scores the blocks LOW CORNERS, not their centroids, because that is what
/// `span_of` goes on to test — and because centroids do not work: in a
/// sub-blocked model a parent centroid and its childrens centroids sit at
/// different offsets within the cell, so a measure of "do these agree" sees
/// two populations and reports no alignment for a model that is perfectly
/// regular. Low corners land on a node for parents and children alike.
///
/// Every low corner on an axis is an integer number of cells from every other,
/// so `low / cell` has the *same* fractional part throughout, whatever the
/// grids origin. Reading that fraction as an angle and taking the mean
/// resultant length turns "do these share a fractional part" into a number in
/// [0, 1]: 1 when they agree exactly, ~0 when spread evenly. Being origin-free
/// is the point — the score needs no lattice, which is what makes scanning
/// angles cheap.
#[allow(clippy::neg_cmp_op_on_partial_ord)]
fn lattice_fit(low_corners: &[[f64; 3]], cell: [f64; 3], axis: usize) -> f64 {
    if low_corners.is_empty() || !(cell[axis] > 0.0) {
        return 0.0;
    }
    let (mut sin_sum, mut cos_sum) = (0.0, 0.0);
    for corner in low_corners {
        let phase = std::f64::consts::TAU * (corner[axis] / cell[axis]).fract();
        let (s, c) = phase.sin_cos();
        sin_sum += s;
        cos_sum += c;
    }
    let n = low_corners.len() as f64;
    ((sin_sum / n).powi(2) + (cos_sum / n).powi(2)).sqrt()
}

/// Score a candidate azimuth: both horizontal axes have to land on the lattice,
/// so the weaker one is the score. Z is untouched by an azimuth.
fn azimuth_score(sample: &[ShellBlock], cell: [f64; 3], radians: f64) -> f64 {
    let orientation = Orientation::from_azimuth(radians);
    let low: Vec<[f64; 3]> = sample
        .iter()
        .map(|b| {
            let c = orientation.to_local(b.centroid);
            [
                c[0] - b.size[0] / 2.0,
                c[1] - b.size[1] / 2.0,
                c[2] - b.size[2] / 2.0,
            ]
        })
        .collect();
    lattice_fit(&low, cell, 0).min(lattice_fit(&low, cell, 1))
}

/// Guesses the rotation of a model that doesn't declare one.
///
/// CSV import produces an explicit list of blocks with world-space centroids
/// and no orientation, so a rotated model arrives looking like a cloud that
/// sits on no lattice — and the sweep rejects it outright. This recovers the
/// rotation from the centroids themselves.
///
/// Only an azimuth (rotation about Z) is guessed. It is overwhelmingly the
/// mining case — benches stay level — and a dipping grid has two more free
/// parameters, which is a much weaker thing to infer from geometry alone. A
/// model that really is tilted still needs its orientation declared.
///
/// Returns `None` when the model is already axis-aligned, or when no angle
/// makes its blocks sit on a lattice — in both cases the caller should sweep
/// with `Orientation::IDENTITY` and let the existing off-lattice error speak.
pub fn infer_orientation<F, I>(blocks: F) -> Option<Orientation>
where
    F: Fn() -> I,
    I: Iterator<Item = ShellBlock>,
{
    // Stride-sample rather than taking a prefix: blocks usually arrive in
    // i/j/k order, so the first few thousand are one corner of the model and
    // a corner is exactly where a grid looks least like itself.
    let total = blocks().count();
    if total == 0 {
        return None;
    }
    let stride = total.div_ceil(ORIENTATION_SAMPLE).max(1);
    let sample: Vec<ShellBlock> = blocks().step_by(stride).collect();

    // Sizes are measured along the grid's own axes, so they are the same
    // whatever the rotation — the cell can be taken before knowing it.
    let mut cell = [f64::INFINITY; 3];
    for block in &sample {
        for axis in 0..3 {
            if block.size[axis] > 0.0 && block.size[axis].is_finite() {
                cell[axis] = cell[axis].min(block.size[axis]);
            }
        }
    }
    if !cell.iter().all(|c| c.is_finite()) {
        return None;
    }

    // Already aligned: don't perturb a model that needs no rotation. Checked
    // before the scan, not as a shortcut inside it — a small model rotated by
    // a fraction of a degree still scores respectably at zero, so this has to
    // be the same strict threshold the winner is held to.
    let upright = azimuth_score(&sample, cell, 0.0);
    if upright >= ORIENTATION_MIN_SCORE {
        return None;
    }

    // A quarter turn maps a grid onto itself, so the search space is [0, 90).
    const COARSE_STEPS: usize = 360; // 0.25 degrees
    let quarter = std::f64::consts::FRAC_PI_2;
    let mut best = (0.0f64, f64::NEG_INFINITY);
    for step in 0..COARSE_STEPS {
        let radians = quarter * step as f64 / COARSE_STEPS as f64;
        let score = azimuth_score(&sample, cell, radians);
        if score > best.1 {
            best = (radians, score);
        }
    }

    // Refine inside the winning coarse bucket, twice, for ~0.0001 degrees.
    let mut window = quarter / COARSE_STEPS as f64;
    for _ in 0..2 {
        let (centre, _) = best;
        for step in -10..=10 {
            let radians = centre + window * step as f64 / 10.0;
            let score = azimuth_score(&sample, cell, radians);
            if score > best.1 {
                best = (radians, score);
            }
        }
        window /= 10.0;
    }

    (best.1 >= ORIENTATION_MIN_SCORE).then(|| Orientation::from_azimuth(best.0))
}

/// Face count of a model's skin without building it — used to decide whether
/// a shell is worth returning before paying to extract one.
pub fn estimate_shell_faces<F, I>(
    blocks: F,
    limits: &ShellLimits,
    orientation: &Orientation,
) -> Result<usize>
where
    F: Fn() -> I,
    I: Iterator<Item = ShellBlock>,
{
    let mesh = extract_shell(blocks, None, &ShellFilter::default(), limits, orientation)?;
    Ok(mesh.face_count())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn cube_grid(nx: usize, ny: usize, nz: usize, size: f64) -> Vec<ShellBlock> {
        let mut blocks = Vec::new();
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    blocks.push(ShellBlock {
                        centroid: [
                            (i as f64 + 0.5) * size,
                            (j as f64 + 0.5) * size,
                            (k as f64 + 0.5) * size,
                        ],
                        size: [size, size, size],
                    });
                }
            }
        }
        blocks
    }

    fn shell_of(blocks: &[ShellBlock]) -> ShellMesh {
        extract_shell(
            || blocks.iter().copied(),
            None,
            &ShellFilter::default(),
            &ShellLimits::default(),
            &Orientation::IDENTITY,
        )
        .expect("shell")
    }

    #[test]
    fn single_block_has_six_faces() {
        let mesh = shell_of(&cube_grid(1, 1, 1, 10.0));
        assert_eq!(mesh.face_count(), 6);
        assert_eq!(mesh.vertices.len(), 6 * 4 * 3);
        assert_eq!(mesh.triangles.len(), 6 * 6);
    }

    // The whole point: interior faces never reach the GPU. A 10³ model is
    // 1000 blocks (6000 cube faces) but only 600 skin faces.
    #[test]
    fn interior_faces_are_dropped() {
        let mesh = shell_of(&cube_grid(10, 10, 10, 5.0));
        assert_eq!(mesh.face_count(), 6 * 10 * 10);
    }

    // Face emission drains the block iterator into fixed-size buffers and
    // emits each in parallel chunks, so a model has to exceed EMIT_BUFFER
    // before the concatenation path runs at all. Every model in the tests
    // above fits in one buffer; this one spans several, and its skin is known
    // analytically, so a mis-rebased chunk shows up as a wrong face count.
    #[test]
    fn a_model_spanning_several_buffers_shells_to_its_analytic_skin() {
        let n = 64;
        let blocks = cube_grid(n, n, n, 1.0);
        assert!(
            blocks.len() > EMIT_BUFFER,
            "{} blocks does not exercise the multi-buffer path",
            blocks.len()
        );

        let mesh = shell_of(&blocks);
        assert_eq!(mesh.face_count(), 6 * n * n);
        assert_eq!(mesh.vertices.len(), mesh.face_count() * 4 * 3);
    }

    // The concatenation rebases each chunk's triangle indices against the
    // running vertex count. If that offset were dropped or applied twice, the
    // indices would still look plausible while pointing at another chunk's
    // vertices — so check every one of them lands in range.
    #[test]
    fn concatenated_chunks_keep_their_triangle_indices_in_range() {
        let blocks = cube_grid(48, 48, 48, 1.0);
        let mesh = shell_of(&blocks);
        let vertex_count = (mesh.vertices.len() / 3) as u32;

        assert!(
            mesh.triangles.iter().all(|&i| i < vertex_count),
            "a triangle indexes past the {vertex_count} vertices emitted"
        );
        // Each face contributes four vertices used by exactly two triangles,
        // so every vertex must be referenced.
        let used: HashSet<u32> = mesh.triangles.iter().copied().collect();
        assert_eq!(used.len(), vertex_count as usize);
    }

    // The workspace manifest requires parallel results to be reproducible
    // whatever the thread count. `par_chunks(..).collect()` is indexed, so the
    // fold order is fixed — this pins that property against a future edit that
    // reaches for an unordered reduction.
    #[test]
    fn repeated_extraction_is_bit_for_bit_identical() {
        let blocks = cube_grid(40, 40, 40, 1.0);
        let values: Vec<f64> = (0..blocks.len()).map(|i| i as f64 * 0.5).collect();
        let run = || {
            extract_shell(
                || blocks.iter().copied(),
                Some(&values),
                &ShellFilter::default(),
                &ShellLimits::default(),
                &Orientation::IDENTITY,
            )
            .expect("shell")
        };

        assert_eq!(run(), run());
    }

    #[test]
    fn shell_covers_the_model_bounds() {
        let mesh = shell_of(&cube_grid(4, 3, 2, 2.0));
        let xs: Vec<f64> = mesh.vertices.iter().step_by(3).copied().collect();
        let min = xs.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!((min - 0.0).abs() < 1e-9, "min x {min}");
        assert!((max - 8.0).abs() < 1e-9, "max x {max}");
    }

    // A hole in the middle of a solid model exposes the six faces around it.
    #[test]
    fn a_missing_block_exposes_its_neighbours() {
        let mut blocks = cube_grid(3, 3, 3, 1.0);
        let centre = blocks
            .iter()
            .position(|b| b.centroid == [1.5, 1.5, 1.5])
            .expect("centre block");
        blocks.remove(centre);
        let mesh = shell_of(&blocks);
        assert_eq!(mesh.face_count(), 6 * 9 + 6);
    }

    #[test]
    fn slab_keeps_only_the_blocks_it_cuts() {
        let blocks = cube_grid(5, 5, 5, 10.0);
        let filter = ShellFilter {
            slabs: vec![Slab {
                origin: [0.0, 0.0, 25.0],
                normal: [0.0, 0.0, 1.0],
                thickness: 10.0,
            }],
            ..Default::default()
        };
        let mesh = extract_shell(
            || blocks.iter().copied(),
            None,
            &filter,
            &ShellLimits::default(),
            &Orientation::IDENTITY,
        )
        .expect("shell");
        // One 5×5 layer survives: 25 blocks, whose skin is the two 5×5 cut
        // faces plus the 4×5 sides.
        assert_eq!(mesh.blocks_kept, 25);
        assert_eq!(mesh.face_count(), 2 * 25 + 4 * 5);
    }

    // Multislices: several cuts live together keep the UNION of what each one
    // does, so a level and a vertical section cross on screen. Intersecting
    // them instead would leave only the row where the two overlap, which is
    // what a shell clipped by both planes on one mapper would have shown.
    #[test]
    fn crossing_slabs_keep_the_union_of_both() {
        let blocks = cube_grid(5, 5, 5, 10.0);
        let level = Slab {
            origin: [0.0, 0.0, 25.0],
            normal: [0.0, 0.0, 1.0],
            thickness: 10.0,
        };
        let section = Slab {
            origin: [25.0, 0.0, 0.0],
            normal: [1.0, 0.0, 0.0],
            thickness: 10.0,
        };
        let kept = |slabs: Vec<Slab>| {
            extract_shell(
                || blocks.iter().copied(),
                None,
                &ShellFilter {
                    slabs,
                    ..Default::default()
                },
                &ShellLimits::default(),
                &Orientation::IDENTITY,
            )
            .expect("shell")
            .blocks_kept
        };

        // 25 blocks each, sharing the 5 where the two planes cross.
        assert_eq!(kept(vec![level]), 25);
        assert_eq!(kept(vec![section]), 25);
        assert_eq!(kept(vec![level, section]), 45);
    }

    #[test]
    fn no_slabs_keeps_every_block() {
        let blocks = cube_grid(3, 3, 3, 1.0);
        let mesh = extract_shell(
            || blocks.iter().copied(),
            None,
            &ShellFilter::default(),
            &ShellLimits::default(),
            &Orientation::IDENTITY,
        )
        .expect("shell");
        assert_eq!(mesh.blocks_kept, 27);
    }

    // A section shows the cut face — that is the whole reason a slab
    // re-extracts rather than clipping the full shell on the GPU.
    #[test]
    fn slab_shell_includes_the_cut_faces() {
        let blocks = cube_grid(3, 3, 3, 1.0);
        let filter = ShellFilter {
            slabs: vec![Slab {
                origin: [1.5, 0.0, 0.0],
                normal: [1.0, 0.0, 0.0],
                thickness: 1.0,
            }],
            ..Default::default()
        };
        let mesh = extract_shell(
            || blocks.iter().copied(),
            None,
            &filter,
            &ShellLimits::default(),
            &Orientation::IDENTITY,
        )
        .expect("shell");
        let normals_at_cut = mesh
            .vertices
            .chunks(3)
            .filter(|v| (v[0] - 1.0).abs() < 1e-9 || (v[0] - 2.0).abs() < 1e-9)
            .count();
        assert!(normals_at_cut > 0, "expected vertices on the cut planes");
        assert_eq!(mesh.blocks_kept, 9);
    }

    #[test]
    fn null_values_drop_out_when_asked() {
        let blocks = cube_grid(3, 1, 1, 1.0);
        let values = [1.0, f64::NAN, 3.0];
        let filter = ShellFilter {
            drop_nan_values: true,
            ..Default::default()
        };
        let mesh = extract_shell(
            || blocks.iter().copied(),
            Some(&values),
            &filter,
            &ShellLimits::default(),
            &Orientation::IDENTITY,
        )
        .expect("shell");
        // Two isolated blocks, six faces each — the gap between them exposes
        // the faces that would otherwise be buried.
        assert_eq!(mesh.blocks_kept, 2);
        assert_eq!(mesh.face_count(), 12);
    }

    // A live selection (#349) narrows the shell to the kept rows, and the
    // boundary of the kept set grows real faces like a slab cut does.
    #[test]
    fn keep_mask_narrows_the_shell_to_the_selected_rows() {
        let blocks = cube_grid(3, 1, 1, 1.0);
        let filter = ShellFilter {
            keep: Some(BlockSubset::from_indices([0usize, 2], blocks.len())),
            ..Default::default()
        };
        let mesh = extract_shell(
            || blocks.iter().copied(),
            None,
            &filter,
            &ShellLimits::default(),
            &Orientation::IDENTITY,
        )
        .expect("shell");
        // Two isolated blocks, six faces each — the dropped middle block
        // exposes the faces it used to bury.
        assert_eq!(mesh.blocks_kept, 2);
        assert_eq!(mesh.face_count(), 12);
    }

    #[test]
    fn keep_mask_intersects_with_slabs() {
        let blocks = cube_grid(3, 3, 3, 1.0);
        let filter = ShellFilter {
            slabs: vec![Slab {
                origin: [1.5, 0.0, 0.0],
                normal: [1.0, 0.0, 0.0],
                thickness: 1.0,
            }],
            keep: Some(BlockSubset::from_indices(0..3usize, blocks.len())),
            ..Default::default()
        };
        let mesh = extract_shell(
            || blocks.iter().copied(),
            None,
            &filter,
            &ShellLimits::default(),
            &Orientation::IDENTITY,
        )
        .expect("shell");
        // The slab keeps the 9 middle-x blocks; of the first three rows only
        // one is in that slice — the two filters AND together.
        assert_eq!(mesh.blocks_kept, 1);
    }

    #[test]
    fn values_ride_along_per_vertex() {
        let blocks = cube_grid(2, 1, 1, 1.0);
        let values = [7.0, 9.0];
        let mesh = extract_shell(
            || blocks.iter().copied(),
            Some(&values),
            &ShellFilter::default(),
            &ShellLimits::default(),
            &Orientation::IDENTITY,
        )
        .expect("shell");
        assert_eq!(mesh.values.len(), mesh.vertices.len() / 3);
        let distinct: HashSet<u64> = mesh.values.iter().map(|v| v.to_bits()).collect();
        assert_eq!(distinct.len(), 2);
        // Ten faces (the shared face between the two is interior), five each.
        assert_eq!(mesh.face_count(), 10);
    }

    // Sub-blocked models: a parent block spans several lattice cells, and the
    // lattice is set by the smallest block. Burial is per-face and
    // conservative — the sub-block's face against the parent is hidden and
    // goes, while the parent's face stays because the sub-block covers only a
    // quarter of it. Over-drawing a partly-covered parent face is the safe
    // direction: dropping it would punch a visible hole in the model.
    #[test]
    fn sub_blocked_parents_span_several_cells() {
        let blocks = vec![
            ShellBlock {
                centroid: [1.0, 1.0, 1.0],
                size: [2.0, 2.0, 2.0],
            },
            ShellBlock {
                centroid: [2.5, 0.5, 0.5],
                size: [1.0, 1.0, 1.0],
            },
        ];
        let mesh = shell_of(&blocks);
        assert_eq!(mesh.blocks_kept, 2);
        assert_eq!(mesh.face_count(), 6 + 5);
    }

    #[test]
    fn off_lattice_models_are_rejected() {
        let blocks = [
            ShellBlock {
                centroid: [0.5, 0.5, 0.5],
                size: [1.0, 1.0, 1.0],
            },
            ShellBlock {
                centroid: [1.83, 0.5, 0.5],
                size: [1.0, 1.0, 1.0],
            },
        ];
        let err = extract_shell(
            || blocks.iter().copied(),
            None,
            &ShellFilter::default(),
            &ShellLimits::default(),
            &Orientation::IDENTITY,
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("regular lattice"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn a_too_fine_lattice_is_refused_before_allocating() {
        let blocks = [
            ShellBlock {
                centroid: [0.0, 0.0, 0.0],
                size: [0.001, 0.001, 0.001],
            },
            ShellBlock {
                centroid: [10_000.0, 10_000.0, 10_000.0],
                size: [10.0, 10.0, 10.0],
            },
        ];
        let err = extract_shell(
            || blocks.iter().copied(),
            None,
            &ShellFilter::default(),
            &ShellLimits::default(),
            &Orientation::IDENTITY,
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("lattice cells") || err.to_string().contains("too large"),
            "unexpected error: {err}"
        );
    }

    /// `cube_grid` rotated about Z by `degrees`, as a CSV-imported rotated
    /// model arrives: world-space centroids, sizes still along the grid axes.
    fn rotated_grid(nx: usize, ny: usize, nz: usize, size: f64, degrees: f64) -> Vec<ShellBlock> {
        let orientation = Orientation::from_azimuth(degrees.to_radians());
        cube_grid(nx, ny, nz, size)
            .into_iter()
            .map(|b| ShellBlock {
                centroid: orientation.to_world(b.centroid),
                size: b.size,
            })
            .collect()
    }

    #[test]
    fn a_rotated_grid_shells_to_the_same_skin_as_an_upright_one() {
        let upright = shell_of(&cube_grid(4, 5, 3, 10.0));
        for degrees in [1.0f64, 15.0, 30.0, 45.0, 73.5] {
            let blocks = rotated_grid(4, 5, 3, 10.0, degrees);
            let mesh = extract_shell(
                || blocks.iter().copied(),
                None,
                &ShellFilter::default(),
                &ShellLimits::default(),
                &Orientation::from_azimuth(degrees.to_radians()),
            )
            .unwrap_or_else(|e| panic!("{degrees} deg should shell: {e}"));
            assert_eq!(
                mesh.face_count(),
                upright.face_count(),
                "rotating a model must not change how many faces its skin has ({degrees} deg)"
            );
            assert_eq!(mesh.blocks_kept, upright.blocks_kept);
        }
    }

    #[test]
    fn a_rotated_grid_without_its_orientation_is_still_rejected() {
        // The regression this whole thing exists for: swept as if upright, a
        // rotation of one degree already puts most blocks off the lattice.
        let blocks = rotated_grid(4, 5, 3, 10.0, 1.0);
        let err = extract_shell(
            || blocks.iter().copied(),
            None,
            &ShellFilter::default(),
            &ShellLimits::default(),
            &Orientation::IDENTITY,
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("regular lattice"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn rotated_faces_come_back_in_world_space() {
        // One block, rotated 90 degrees: its skin must sit where the block is,
        // not where an unrotated sweep would have put it.
        let degrees = 90.0;
        let blocks = rotated_grid(1, 1, 1, 10.0, degrees);
        let mesh = extract_shell(
            || blocks.iter().copied(),
            None,
            &ShellFilter::default(),
            &ShellLimits::default(),
            &Orientation::from_azimuth(degrees.to_radians()),
        )
        .expect("single rotated block should shell");

        let centre = blocks[0].centroid;
        for vertex in mesh.vertices.chunks_exact(3) {
            for axis in 0..3 {
                assert!(
                    (vertex[axis] - centre[axis]).abs() <= 5.0 + 1e-6,
                    "vertex {vertex:?} is not on the block at {centre:?}"
                );
            }
        }
    }

    #[test]
    fn infer_orientation_recovers_the_angle_it_was_rotated_by() {
        for degrees in [1.0f64, 7.25, 15.0, 30.0, 45.0, 62.5, 89.0] {
            let blocks = rotated_grid(6, 7, 3, 10.0, degrees);
            let found = infer_orientation(|| blocks.iter().copied())
                .unwrap_or_else(|| panic!("{degrees} deg should be detected"));
            let recovered = found.azimuth().to_degrees();
            assert!(
                (recovered - degrees).abs() < 0.05,
                "expected ~{degrees} deg, inferred {recovered} deg"
            );

            // And the angle it found has to be one the sweep can actually use.
            extract_shell(
                || blocks.iter().copied(),
                None,
                &ShellFilter::default(),
                &ShellLimits::default(),
                &found,
            )
            .unwrap_or_else(|e| panic!("inferred orientation should shell at {degrees}: {e}"));
        }
    }

    #[test]
    fn infer_orientation_leaves_an_upright_model_alone() {
        let blocks = cube_grid(6, 6, 3, 10.0);
        assert!(infer_orientation(|| blocks.iter().copied()).is_none());
    }

    #[test]
    fn infer_orientation_declines_a_model_that_is_not_a_grid() {
        // Jittered centroids sit on no lattice at any angle; guessing one would
        // be worse than reporting that the model cannot be shelled.
        let mut blocks = cube_grid(6, 6, 3, 10.0);
        for (i, block) in blocks.iter_mut().enumerate() {
            let jitter = ((i * 37) % 100) as f64 / 100.0 * 4.0;
            block.centroid[0] += jitter;
            block.centroid[1] -= jitter;
        }
        assert!(infer_orientation(|| blocks.iter().copied()).is_none());
    }

    #[test]
    fn infer_orientation_handles_a_sub_blocked_rotated_model() {
        // Sub-blocks are smaller than their parents; the finest size is the
        // cell, and both still land on it.
        let degrees = 22.5f64;
        let orientation = Orientation::from_azimuth(degrees.to_radians());
        let mut blocks = Vec::new();
        for i in 0..8 {
            for j in 0..8 {
                let parent = [i as f64 * 10.0 + 5.0, j as f64 * 10.0 + 5.0, 5.0];
                if (i + j) % 2 == 0 {
                    blocks.push(ShellBlock {
                        centroid: orientation.to_world(parent),
                        size: [10.0, 10.0, 10.0],
                    });
                } else {
                    for (dx, dy) in [(-2.5, -2.5), (2.5, -2.5), (-2.5, 2.5), (2.5, 2.5)] {
                        blocks.push(ShellBlock {
                            centroid: orientation.to_world([
                                parent[0] + dx,
                                parent[1] + dy,
                                parent[2],
                            ]),
                            size: [5.0, 5.0, 10.0],
                        });
                    }
                }
            }
        }
        let found =
            infer_orientation(|| blocks.iter().copied()).expect("sub-blocked model should detect");
        assert!((found.azimuth().to_degrees() - degrees).abs() < 0.05);
    }

    #[test]
    fn from_axes_rejects_a_frame_that_is_not_a_rotation() {
        // Not unit length.
        assert!(Orientation::from_axes([2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]).is_err());
        // Not perpendicular.
        let d = 1.0 / 2f64.sqrt();
        assert!(Orientation::from_axes([1.0, 0.0, 0.0], [d, d, 0.0], [0.0, 0.0, 1.0]).is_err());
        // Left-handed.
        assert!(
            Orientation::from_axes([1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]).is_err()
        );
        // A genuine rotation.
        assert!(Orientation::from_axes([0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]).is_ok());
    }

    #[test]
    fn a_slab_cuts_a_rotated_model_in_world_space() {
        // The slab is a world-space plane; only the sweep is in grid space.
        let degrees = 30.0f64;
        let orientation = Orientation::from_azimuth(degrees.to_radians());
        let blocks = rotated_grid(6, 6, 3, 10.0, degrees);
        let whole = extract_shell(
            || blocks.iter().copied(),
            None,
            &ShellFilter::default(),
            &ShellLimits::default(),
            &orientation,
        )
        .expect("whole model should shell");

        let cut = extract_shell(
            || blocks.iter().copied(),
            None,
            &ShellFilter {
                slabs: vec![Slab {
                    origin: blocks[0].centroid,
                    normal: [0.0, 0.0, 1.0],
                    thickness: 10.0,
                }],
                drop_nan_values: false,
                keep: None,
            },
            &ShellLimits::default(),
            &orientation,
        )
        .expect("cut model should shell");

        assert!(cut.blocks_kept > 0, "the slab kept nothing");
        assert!(
            cut.blocks_kept < whole.blocks_kept,
            "the slab kept everything ({} of {})",
            cut.blocks_kept,
            whole.blocks_kept
        );
    }

    #[test]
    fn face_budget_is_enforced() {
        let blocks = cube_grid(4, 4, 4, 1.0);
        let err = extract_shell(
            || blocks.iter().copied(),
            None,
            &ShellFilter::default(),
            &ShellLimits {
                max_faces: 10,
                ..ShellLimits::default()
            },
            &Orientation::IDENTITY,
        )
        .unwrap_err();
        assert!(err.to_string().contains("faces"), "unexpected error: {err}");
    }

    #[test]
    fn empty_models_are_an_error_not_a_panic() {
        let blocks: Vec<ShellBlock> = Vec::new();
        assert!(
            extract_shell(
                || blocks.iter().copied(),
                None,
                &ShellFilter::default(),
                &ShellLimits::default(),
                &Orientation::IDENTITY,
            )
            .is_err()
        );
    }

    // Winding: every face's normal points away from the model, so a solid
    // model reads as solid under back-face culling.
    #[test]
    fn faces_wind_outward() {
        let mesh = shell_of(&cube_grid(1, 1, 1, 2.0));
        let centre = [1.0, 1.0, 1.0];
        for face in 0..mesh.face_count() {
            let v = |n: usize| {
                let idx = mesh.triangles[face * 6 + n] as usize;
                [
                    mesh.vertices[idx * 3],
                    mesh.vertices[idx * 3 + 1],
                    mesh.vertices[idx * 3 + 2],
                ]
            };
            let (a, b, c) = (v(0), v(1), v(2));
            let e1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let e2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
            let normal = [
                e1[1] * e2[2] - e1[2] * e2[1],
                e1[2] * e2[0] - e1[0] * e2[2],
                e1[0] * e2[1] - e1[1] * e2[0],
            ];
            let outward = [a[0] - centre[0], a[1] - centre[1], a[2] - centre[2]];
            let dot = normal[0] * outward[0] + normal[1] * outward[1] + normal[2] * outward[2];
            assert!(dot > 0.0, "face {face} winds inward (dot {dot})");
        }
    }
}
