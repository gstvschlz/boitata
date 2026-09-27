use std::collections::{BTreeMap, BTreeSet, HashMap};

use serde::{Deserialize, Serialize};

use crate::{Error, Result};

/// Named categories with integer codes: a code is the index of its name.
/// `mapping` sends raw labels to names, many to one; unlisted labels fall
/// into `other`, always the last code, when there is one.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "Parts")]
pub struct Categories {
    names: Vec<String>,
    colors: Option<Vec<String>>,
    mapping: BTreeMap<String, String>,
    other: Option<String>,
}

#[derive(Deserialize)]
struct Parts {
    names: Vec<String>,
    #[serde(default)]
    colors: Option<Vec<String>>,
    #[serde(default)]
    mapping: BTreeMap<String, String>,
    #[serde(default)]
    other: Option<String>,
}

impl TryFrom<Parts> for Categories {
    type Error = Error;
    fn try_from(p: Parts) -> Result<Self> {
        Self::new(p.names, p.colors, p.mapping, p.other)
    }
}

fn invalid(message: impl Into<String>) -> Error {
    Error::Categories(message.into())
}

fn check_weights(n: usize, weights: Option<&[f64]>) -> Result<()> {
    match weights {
        Some(w) if w.len() != n => Err(Error::Length {
            expected: n,
            found: w.len(),
        }),
        Some(w) if w.iter().any(|w| !(w.is_finite() && *w >= 0.0)) => {
            Err(invalid("weights must be finite and >= 0"))
        }
        _ => Ok(()),
    }
}

impl Categories {
    /// `other`, when not listed, is appended as the last name.
    pub fn new(
        mut names: Vec<String>,
        colors: Option<Vec<String>>,
        mapping: BTreeMap<String, String>,
        other: Option<String>,
    ) -> Result<Self> {
        if let Some(o) = &other
            && !names.contains(o)
        {
            names.push(o.clone());
        }
        if names.is_empty() {
            return Err(invalid("need at least one name"));
        }
        if names.iter().collect::<BTreeSet<_>>().len() < names.len() {
            return Err(invalid("names must be distinct"));
        }
        if other.as_ref().is_some_and(|o| names.last() != Some(o)) {
            return Err(invalid("other must be the last name"));
        }
        if colors.as_ref().is_some_and(|c| c.len() != names.len()) {
            return Err(invalid(format!(
                "need {} colors, one per name",
                names.len()
            )));
        }
        for (label, name) in &mapping {
            if names.contains(label) {
                return Err(invalid(format!("mapping key {label} is a name")));
            }
            if !names.contains(name) {
                return Err(invalid(format!(
                    "mapping sends {label} to {name}, not a name"
                )));
            }
        }
        Ok(Self {
            names,
            colors,
            mapping,
            other,
        })
    }

    /// Names in code order: `mapping` applied, then the distinct names
    /// sorted, as integers when all are. Names whose share of `weights`
    /// is below `min_share` are lumped into `other`, which exists only
    /// then. Independent of row order.
    pub fn from_labels(
        labels: &[Option<&str>],
        weights: Option<&[f64]>,
        min_share: f64,
        mapping: BTreeMap<String, String>,
        other: &str,
    ) -> Result<Self> {
        if !(0.0..=1.0).contains(&min_share) {
            return Err(invalid("min_share must be in [0, 1]"));
        }
        check_weights(labels.len(), weights)?;
        let mut found: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
        for (i, l) in labels.iter().enumerate() {
            if let Some(l) = *l {
                let name = mapping.get(l).map_or(l, String::as_str);
                found
                    .entry(name)
                    .or_default()
                    .push(weights.map_or(1.0, |w| w[i]));
            }
        }
        let sums: BTreeMap<&str, f64> = found
            .into_iter()
            .map(|(n, mut w)| {
                w.sort_by(f64::total_cmp);
                (n, w.iter().sum())
            })
            .collect();
        let total: f64 = sums.values().sum();
        if total.is_nan() || total <= 0.0 {
            return Err(invalid("no labels with positive weight"));
        }
        let lumped: BTreeSet<&str> = sums
            .iter()
            .filter(|&(_, s)| s / total < min_share)
            .map(|(&n, _)| n)
            .collect();
        let other = (!lumped.is_empty()).then(|| other.to_string());
        let mut names: Vec<String> = sums
            .keys()
            .filter(|&&n| !lumped.contains(n) && Some(n) != other.as_deref())
            .map(|n| n.to_string())
            .collect();
        if names.iter().all(|n| n.parse::<i64>().is_ok()) {
            names.sort_by_key(|n| (n.parse::<i64>().expect("integer"), n.clone()));
        }
        names.extend(other.clone());
        let mut kept = BTreeMap::new();
        for (label, name) in &mapping {
            let name = match &other {
                Some(o) if lumped.contains(name.as_str()) => o,
                _ => name,
            };
            if names.contains(name) {
                kept.insert(label.clone(), name.clone());
            }
        }
        if let Some(o) = &other {
            for n in lumped.into_iter().filter(|n| n != o) {
                kept.insert(n.to_string(), o.clone());
            }
        }
        Self::new(names, None, kept, other)
    }

    pub fn names(&self) -> &[String] {
        &self.names
    }

    pub fn colors(&self) -> Option<&[String]> {
        self.colors.as_deref()
    }

    pub fn mapping(&self) -> &BTreeMap<String, String> {
        &self.mapping
    }

    pub fn other(&self) -> Option<&str> {
        self.other.as_deref()
    }

    /// Code of each label; an unlisted one goes to `other`, else is an error.
    pub fn encode(&self, labels: &[Option<&str>]) -> Result<Vec<Option<u32>>> {
        let code: HashMap<&str, u32> = self
            .names
            .iter()
            .enumerate()
            .map(|(c, n)| (n.as_str(), c as u32))
            .collect();
        let other = self.other.as_ref().map(|_| self.names.len() as u32 - 1);
        let mut unknown: BTreeMap<&str, usize> = BTreeMap::new();
        let codes = labels
            .iter()
            .map(|l| {
                let l = (*l)?;
                let name = self.mapping.get(l).map_or(l, String::as_str);
                let c = code.get(name).copied().or(other);
                if c.is_none() {
                    *unknown.entry(l).or_default() += 1;
                }
                c
            })
            .collect();
        if unknown.is_empty() {
            return Ok(codes);
        }
        let list = |items: Vec<String>| {
            let more = items.len().saturating_sub(5);
            let mut s = items[..items.len() - more].join(", ");
            if more > 0 {
                s += &format!(" and {more} more");
            }
            s
        };
        let found = unknown
            .iter()
            .map(|(l, &n)| format!("{l} ({n} row{})", if n == 1 { "" } else { "s" }))
            .collect();
        let known = self
            .names
            .iter()
            .chain(self.mapping.keys())
            .cloned()
            .collect();
        let count = match unknown.len() {
            1 => "1 label is not a category".to_string(),
            n => format!("{n} labels are not categories"),
        };
        Err(Error::UnknownLabels(format!(
            "{count}: {}; known labels are {}",
            list(found),
            list(known)
        )))
    }

    pub fn decode(&self, codes: &[Option<u32>]) -> Result<Vec<Option<&str>>> {
        codes
            .iter()
            .map(|c| {
                c.map(|c| {
                    self.names
                        .get(c as usize)
                        .map(String::as_str)
                        .ok_or_else(|| invalid(format!("code {c} is not below {}", self.len())))
                })
                .transpose()
            })
            .collect()
    }

    /// `names` merged into `into`: a listed name, else `other` when there
    /// is none yet, else a new name before `other`. The merged names
    /// become mapping entries.
    pub fn lump(&self, names: &[&str], into: &str) -> Result<Self> {
        for n in names {
            if !self.names.iter().any(|m| m == n) {
                return Err(invalid(format!("{n} is not a name")));
            }
            if *n == into || Some(*n) == self.other() {
                return Err(invalid(format!("cannot lump {n} into {into}")));
            }
        }
        let keep = |n: &String| !names.contains(&n.as_str());
        let mut kept: Vec<String> = self.names.iter().filter(|n| keep(n)).cloned().collect();
        let mut colors = self.colors.as_ref().map(|c| {
            c.iter()
                .zip(&self.names)
                .filter(|(_, n)| keep(n))
                .map(|(c, _)| c.clone())
                .collect::<Vec<_>>()
        });
        let mut other = self.other.clone();
        if !kept.iter().any(|n| n == into) {
            let at = kept.len() - usize::from(other.is_some());
            kept.insert(at, into.to_string());
            if let Some(c) = &mut colors {
                c.insert(at, "0.6".into());
            }
            other.get_or_insert_with(|| into.to_string());
        }
        let mut mapping = self.mapping.clone();
        for v in mapping.values_mut().filter(|v| !keep(v)) {
            *v = into.to_string();
        }
        mapping.extend(names.iter().map(|n| (n.to_string(), into.to_string())));
        Self::new(kept, colors, mapping, other)
    }

    /// Weighted proportion of each code, nulls skipped.
    pub fn shares(&self, codes: &[Option<u32>], weights: Option<&[f64]>) -> Result<Vec<f64>> {
        check_weights(codes.len(), weights)?;
        self.decode(codes)?;
        let mut sums = vec![0.0; self.len()];
        for (i, c) in codes.iter().enumerate() {
            if let Some(c) = c {
                sums[*c as usize] += weights.map_or(1.0, |w| w[i]);
            }
        }
        let total: f64 = sums.iter().sum();
        if total <= 0.0 {
            return Err(invalid("no codes with positive weight"));
        }
        Ok(sums.into_iter().map(|s| s / total).collect())
    }

    pub fn len(&self) -> usize {
        self.names.len()
    }

    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn some(labels: &[String]) -> Vec<Option<&str>> {
        labels.iter().map(|l| Some(l.as_str())).collect()
    }

    fn labels() -> impl Strategy<Value = (Vec<String>, Vec<f64>)> {
        prop::collection::vec(("[a-e]|[0-9]{1,2}", 0.0..10.0f64), 1..60)
            .prop_map(|v| v.into_iter().unzip())
    }

    proptest! {
        #[test]
        fn round_trip((raw, _) in labels(), cut in 0usize..6) {
            let mut names: Vec<String> = raw.iter().cloned().collect::<BTreeSet<_>>().into_iter().collect();
            let lumped: Vec<String> = names.split_off(cut.min(names.len()));
            let c = Categories::new(names.clone(), None, BTreeMap::new(), Some("other".into())).unwrap();
            let mut with_null = some(&raw);
            with_null.push(None);
            let back = c.decode(&c.encode(&with_null).unwrap()).unwrap();
            for (l, b) in with_null.iter().zip(&back) {
                let expected = l.map(|l| if lumped.iter().any(|m| m == l) { "other" } else { l });
                prop_assert_eq!(*b, expected);
            }
        }

        #[test]
        fn from_labels_ignores_order((raw, w) in labels(), min_share in 0.0..0.3f64, seed: u64) {
            let a = Categories::from_labels(&some(&raw), Some(&w), min_share, BTreeMap::new(), "other");
            let mut rows: Vec<usize> = (0..raw.len()).collect();
            rows.sort_by_key(|&i| (i as u64).wrapping_mul(seed | 1).rotate_left(17));
            let raw2: Vec<String> = rows.iter().map(|&i| raw[i].clone()).collect();
            let w2: Vec<f64> = rows.iter().map(|&i| w[i]).collect();
            let b = Categories::from_labels(&some(&raw2), Some(&w2), min_share, BTreeMap::new(), "other");
            match (a, b) {
                (Ok(a), Ok(b)) => {
                    prop_assert_eq!(serde_json::to_string(&a).unwrap(), serde_json::to_string(&b).unwrap());
                    prop_assert_eq!(&a, &b);
                    if let Some(o) = a.other() {
                        prop_assert_eq!(a.names().last().map(String::as_str), Some(o));
                    }
                    let plain = &a.names()[..a.len() - usize::from(a.other().is_some())];
                    if plain.iter().all(|n| n.parse::<i64>().is_ok()) {
                        prop_assert!(plain.windows(2).all(|p| p[0].parse::<i64>().unwrap() <= p[1].parse::<i64>().unwrap()));
                    }
                }
                (a, b) => prop_assert_eq!(a.is_err(), b.is_err()),
            }
        }

        #[test]
        fn shares_add_up((raw, w) in labels(), min_share in 0.0..0.3f64) {
            prop_assume!(w.iter().sum::<f64>() > 0.0);
            let full = Categories::from_labels(&some(&raw), Some(&w), 0.0, BTreeMap::new(), "other").unwrap();
            let full_shares = full.shares(&full.encode(&some(&raw)).unwrap(), Some(&w)).unwrap();
            let c = Categories::from_labels(&some(&raw), Some(&w), min_share, BTreeMap::new(), "other").unwrap();
            let shares = c.shares(&c.encode(&some(&raw)).unwrap(), Some(&w)).unwrap();
            prop_assert!((shares.iter().sum::<f64>() - 1.0).abs() < 1e-9);
            let lumped: Vec<usize> = full.names().iter().enumerate()
                .filter(|(_, n)| !c.names().contains(n)).map(|(i, _)| i).collect();
            if let Some(o) = c.other() {
                let own = full.names().iter().position(|n| n == o).map_or(0.0, |i| full_shares[i]);
                let summed: f64 = lumped.iter().map(|&i| full_shares[i]).sum::<f64>() + own;
                prop_assert!((shares[c.len() - 1] - summed).abs() < 1e-9);
            }
            for (n, s) in c.names().iter().zip(&shares) {
                if Some(n.as_str()) != c.other() {
                    prop_assert!(*s >= min_share - 1e-12);
                }
            }
            let names: Vec<&str> = lumped.iter().map(|&i| full.names()[i].as_str()).collect();
            if !names.is_empty() && full.len() > names.len() {
                let merged = full.lump(&names, "other").unwrap();
                let by_hand: Vec<Option<&str>> = raw.iter()
                    .map(|l| Some(if names.contains(&l.as_str()) { "other" } else { l.as_str() })).collect();
                prop_assert_eq!(merged.encode(&some(&raw)).unwrap(), merged.encode(&by_hand).unwrap());
            }
        }

        #[test]
        fn json_round_trip((raw, w) in labels(), min_share in 0.0..0.3f64) {
            if let Ok(c) = Categories::from_labels(&some(&raw), Some(&w), min_share, BTreeMap::new(), "other") {
                let back: Categories = serde_json::from_str(&serde_json::to_string(&c).unwrap()).unwrap();
                prop_assert_eq!(back, c);
            }
        }
    }

    #[test]
    fn errors() {
        let s = |v: &[&str]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>();
        let none = BTreeMap::new;
        assert!(Categories::new(vec![], None, none(), None).is_err());
        assert!(Categories::new(s(&["a", "a"]), None, none(), None).is_err());
        assert!(Categories::new(s(&["o", "a"]), None, none(), Some("o".into())).is_err());
        assert!(Categories::new(s(&["a"]), Some(s(&["r", "g"])), none(), None).is_err());
        let to_b = BTreeMap::from([("x".to_string(), "b".to_string())]);
        assert!(Categories::new(s(&["a"]), None, to_b, None).is_err());
        let c = Categories::new(s(&["a", "b"]), None, none(), None).unwrap();
        let err = c.encode(&[Some("a"), Some("z"), Some("y")]).unwrap_err();
        assert_eq!(
            err.to_string(),
            "2 labels are not categories: y (1 row), z (1 row); known labels are a, b"
        );
        assert!(c.decode(&[Some(2)]).is_err());
        assert!(c.shares(&[Some(0)], Some(&[-1.0])).is_err());
        assert!(c.lump(&["z"], "other").is_err());
        assert!(Categories::from_labels(&[None], None, 0.0, none(), "other").is_err());
        assert!(
            serde_json::from_str::<Categories>(
                r#"{"names":[],"colors":null,"mapping":{},"other":null}"#
            )
            .is_err()
        );
    }

    #[test]
    fn integers_sort_numerically_and_lump_goes_last() {
        let raw = [
            "10", "9", "10", "100", "9", "10", "10", "9", "10", "9", "10",
        ];
        let raw: Vec<_> = raw.iter().map(|l| Some(*l)).collect();
        let c = Categories::from_labels(&raw, None, 0.1, BTreeMap::new(), "other").unwrap();
        assert_eq!(c.names(), ["9", "10", "other"]);
        assert_eq!(c.mapping()["100"], "other");
        let m = c.lump(&["9"], "10").unwrap();
        assert_eq!(m.names(), ["10", "other"]);
        assert_eq!(m.encode(&raw).unwrap()[1], Some(0));
    }
}
