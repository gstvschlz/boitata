*[realization]: One simulated map that honors the samples and reproduces the histogram and variogram of the data.
*[realizations]: Simulated maps, each honoring the samples and reproducing the histogram and variogram of the data.
*[normal scores]: Values replaced by the standard Gaussian values with the same cumulative probability.
*[back-transform]: The inverse of a transform, returning simulated scores to the original units.
*[random path]: The random order in which sequential simulation visits the grid nodes.
*[SGS]: Sequential Gaussian simulation: nodes are visited in random order, kriged, drawn from the Gaussian that kriging gives, and added to the data.
*[E-type]: The mean of the realizations at each node, a smooth estimate comparable to kriging.
*[seed]: The number that starts the random-number generator, so the same seed gives the same realizations.
*[cross-validation]: Re-estimating samples from the other samples and comparing the estimates with the actual values.
*[kriging efficiency]: Share of the block variance that an estimate resolves: 1 for a perfectly known block, 0 or less for one known no better than the mean.
*[grade-tonnage curve]: Tonnage and mean grade above each cutoff.
