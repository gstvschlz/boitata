*[lag]: the separation vector between two samples, a distance and a direction.
*[h-scatterplot]: a plot of the value at one end of each pair of samples a lag h apart against the value at the other end.
*[experimental variogram]: half the mean squared difference between samples, computed for pairs grouped by lag.
*[lag tolerance]: how far a pair's distance may differ from a lag center and still count toward that lag; half the lag spacing.
*[angular tolerance]: the half-angle of the cone around a direction within which pairs count toward a directional variogram.
*[variogram map]: the experimental variogram computed for every direction and lag, drawn as an image.
*[anisotropy]: continuity that depends on direction, with a longer range along some directions than others.
*[anisotropy ellipse]: the ellipse, or ellipsoid in 3D, whose axes are the variogram ranges in the principal directions.
*[azimuth]: a horizontal direction in degrees, measured clockwise from north.
*[dip]: the angle in degrees below the horizontal; positive points down.
*[positive definite]: property of a variogram model that guarantees a positive variance for any weighted combination of samples.
*[inverse distance]: an estimator that weights each sample by a power of the inverse of its distance to the target.
*[ordinary kriging]: kriging with an unknown mean that is constant near the target; its weights sum to one.
*[kriging variance]: the error variance kriging predicts at a target, from the sample layout and the variogram alone.
*[screen effect]: the loss of weight by a sample hidden behind a closer sample in the same direction.
*[declustering effect]: the sharing of weight among clustered samples, which kriging treats almost as one.
*[block kriging]: kriging of the average value over a block instead of the value at a point.
*[slope of regression]: the slope of the true values regressed on the estimates; 1 means no conditional bias.
*[conditional bias]: a systematic error that depends on the estimate itself, with high estimates too high and low ones too low.
