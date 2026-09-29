//! Neighbours and simple-kriging weights of nodes. They depend only on the
//! locations of the nodes and data, so they are found once and applied to
//! the residuals of every realization conditioned from the same data.

use estimation::Sample;
use estimation::krige::{Kind, krige};
use estimation::search::SearchTree;
use variogram::Variogram;

use crate::error::{Result, SimError};

/// Datum `datum` with simple-kriging weight `weight`, its grade capped at
/// `cap` by a clamping high-grade restriction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Neighbor {
    pub datum: u32,
    pub weight: f64,
    pub cap: Option<f64>,
}

/// The neighbours of each of a run of nodes.
pub(crate) struct Conditioning {
    offsets: Vec<usize>,
    neighbors: Vec<Neighbor>,
}

impl Conditioning {
    /// The data `tree` selects for each of `targets`, of domain `domains[i]`
    /// when given, with their weights in the simple kriging of `vg`; `data`
    /// are the samples the tree was built from.
    pub(crate) fn new(
        targets: &[(f64, f64, f64)],
        domains: Option<&[u32]>,
        tree: &SearchTree,
        data: &[Sample],
        vg: &Variogram,
    ) -> Result<Self> {
        let mut offsets = Vec::with_capacity(targets.len() + 1);
        offsets.push(0);
        let mut neighbors = Vec::new();
        for (i, g) in targets.iter().enumerate() {
            let chosen = tree
                .neighbors_in(g, domains.map(|d| d[i]))
                .unwrap_or_default();
            if !chosen.is_empty() {
                let near: Vec<Sample> = chosen
                    .iter()
                    .map(|&k| Sample::new(data[k].loc, 0.0))
                    .collect();
                let weights = krige(Kind::Simple { mean: 0.0 }, g, &near, vg)
                    .map_err(|e| SimError::Estimation(e.to_string()))?
                    .weights;
                neighbors.extend(chosen.iter().zip(weights).map(|(&k, weight)| Neighbor {
                    datum: k as u32,
                    weight,
                    cap: tree.cap(g, None, k),
                }));
            }
            offsets.push(neighbors.len());
        }
        Ok(Self { offsets, neighbors })
    }

    pub(crate) fn of(&self, i: usize) -> &[Neighbor] {
        &self.neighbors[self.offsets[i]..self.offsets[i + 1]]
    }

    /// The kriged residual of node `i` in each realization of a batch, in
    /// `out`, summed in neighbour order as [`krige`] sums it;
    /// `residual(n, row)` writes neighbour `n`'s residual in each realization
    /// into `row` (as long as `out`). False, leaving `out`, when the node has
    /// no neighbours.
    pub(crate) fn krige_batch(
        &self,
        i: usize,
        residual: impl Fn(&Neighbor, &mut [f64]),
        row: &mut [f64],
        out: &mut [f64],
    ) -> bool {
        let near = self.of(i);
        if near.is_empty() {
            return false;
        }
        out.fill(-0.0);
        for n in near {
            residual(n, row);
            for (o, r) in out.iter_mut().zip(row.iter()) {
                *o += n.weight * r;
            }
        }
        for o in out.iter_mut() {
            *o += 0.0;
        }
        true
    }
}
