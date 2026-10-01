*[lag]: The separation vector between two samples, a distance and a direction.
*[h-scatterplot]: A plot of the value at one end of each pair of samples a lag h apart against the value at the other end.
*[experimental variogram]: Half the mean squared difference between samples, computed for pairs grouped by lag.
*[lag tolerance]: How far a pair's distance may differ from a lag center and still count toward that lag; half the lag spacing.
*[angular tolerance]: The half-angle of the cone around a direction within which pairs count toward a directional variogram.
*[variogram map]: The experimental variogram computed for every direction and lag, drawn as an image.
*[anisotropy]: Continuity that depends on direction, with a longer range along some directions than others.
*[anisotropy ellipse]: The ellipse, or ellipsoid in 3D, whose axes are the variogram ranges in the principal directions.
*[azimuth]: A horizontal direction in degrees, measured clockwise from north.
*[dip]: The angle in degrees below the horizontal; positive points down.
*[positive definite]: Property of a variogram model that guarantees a positive variance for any weighted combination of samples.
*[inverse distance]: An estimator that weights each sample by a power of the inverse of its distance to the target.
*[ordinary kriging]: Kriging with an unknown mean that is constant near the target; its weights sum to one.
*[kriging variance]: The error variance kriging predicts at a target, from the sample layout and the variogram alone.
*[screen effect]: The loss of weight by a sample hidden behind a closer sample in the same direction.
*[declustering effect]: The sharing of weight among clustered samples, which kriging treats almost as one.
*[block kriging]: Kriging of the average value over a block rather than the value at a point.
*[slope of regression]: The slope of the true values regressed on the estimates; 1 means no conditional bias.
*[conditional bias]: A systematic error that depends on the estimate itself, with high estimates too high and low ones too low.
