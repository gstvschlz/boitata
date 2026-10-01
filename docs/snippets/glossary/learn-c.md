*[realization]: one simulated map that honors the samples and reproduces the histogram and variogram of the data.
*[realizations]: simulated maps, each honoring the samples and reproducing the histogram and variogram of the data.
*[normal scores]: values replaced by the standard gaussian values with the same cumulative probability.
*[back-transform]: the inverse of a transform, returning simulated scores to the original units.
*[random path]: the random order in which sequential simulation visits the grid nodes.
*[SGS]: sequential gaussian simulation: it visits the nodes in random order, krigs each one, draws from the gaussian that kriging gives and adds the draw to the data.
*[E-type]: the mean of the realizations at each node, a smooth estimate comparable to kriging.
*[seed]: the number that starts the random-number generator, so the same seed gives the same realizations.
*[cross-validation]: re-estimating each sample from the other samples and comparing the estimates with the actual values.
*[kriging efficiency]: share of the block variance an estimate resolves: 1 for a perfectly known block, 0 or less for one known no better than the mean.
*[grade-tonnage curve]: tonnage and mean grade above each cutoff.
