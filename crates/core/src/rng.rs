//! Seed derivation shared by every seeded algorithm.

/// The SplitMix64 finalizer: a bijection on `u64` that scatters nearby
/// inputs across the whole range.
pub fn splitmix(z: u64) -> u64 {
    let z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    let z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Seed of realization (or stream) `k` under the user seed `seed`.
///
/// Stable and portable: pure integer arithmetic, identical on every platform
/// and release. Nearby seeds share no realization, and realization `k` does
/// not depend on how many are drawn. Nest it for streams within a
/// realization: `realization_seed(realization_seed(seed, k), j)`.
pub fn realization_seed(seed: u64, k: u64) -> u64 {
    splitmix(splitmix(seed) ^ k)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_values() {
        assert_eq!(splitmix(0), 0xE220_A839_7B1D_CDAF);
        assert_eq!(realization_seed(0, 0), splitmix(splitmix(0)));
    }

    #[test]
    fn nearby_seeds_share_no_realization() {
        let seeds: std::collections::HashSet<u64> = (0..100)
            .flat_map(|s| (0..1000).map(move |k| realization_seed(s, k)))
            .collect();
        assert_eq!(seeds.len(), 100_000);
    }
}
