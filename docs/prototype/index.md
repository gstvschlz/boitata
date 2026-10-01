# Themes and components

Pick a theme in the header selector. Every theme keeps the logo and the palette; they differ in type, measure,
spacing, rules and how the teaching components look. This page uses every component once, so you can judge a theme
on one screen and in both light and dark mode.

| Theme | Intent |
| --- | --- |
| Refined baseline | The current site: sans body, Archivo headings, Material defaults. |
| Editorial serif | A book page: serif body, wide leading, a 68-character measure, small-caps headings. |
| Swiss grid | Grotesk type, numbered sections, heavy rules, flat blocks, everything flush left. |
| Tufte | A narrow text column with notes in the margin; figures run wider than the text. |
| Textbook | A chapter band on the title, boxed definitions and key ideas, prominent figure numbers. |
| Field notebook | Graph paper, handwritten titles and captions, notes taped to the page. |
| Blueprint | Navy drafting sheet in both modes, thin grid lines, mono labels, flame accents. |
| Lab terminal | Monospaced headings, dense spacing, bracketed notes. |
| Magazine | Large display headings, open whitespace, card layouts, pull-quote key ideas. |
| Minimal paper | Hairline rules, a quiet sidebar and chrome that shows itself on hover. |
| Strata | Sections banded like a core log; notes carry colored strata on the left edge. |
| High contrast | 18 px text and up, AAA contrast, underlined links, thick focus rings. |

## How samples relate with distance

!!! learn "What you'll learn"
    - Read a variogram: nugget, sill and range.
    - Set a search ellipse from the variogram.
    - Tell ordinary kriging from simpler estimators.

    **Prerequisites:** [samples and support](../learn/01-samples-and-support/learn_01.md).

Two samples a meter apart tend to carry similar grades; two samples a kilometer apart do not. The variogram measures
how fast that similarity decays. Plot half the mean squared difference between pairs of samples against the distance
that separates them, and you get a curve that starts at the nugget, climbs, and flattens at the sill once the pairs
lie farther apart than the range.

<figure class="bt-figure">
--8<-- "svg/demo-variogram.svg"
<figcaption><b>Figure 1.</b> A variogram. Pairs of close samples differ little; past the range, pairs differ as much as any two samples picked at random.</figcaption>
</figure>

!!! key "Key idea"
    The variogram turns "nearby samples look alike" into a number for every distance. Kriging reads its weights from
    that curve.

### Reading the curve

- The **nugget** holds measurement error and variation shorter than the closest sample spacing.
- The **sill** equals the variance of the data when the domain is stationary.
- The **range** tells you how far a sample still informs its neighbors.

!!! pitfall "Pitfall"
    A variogram computed with a lag shorter than the sample spacing has few pairs in its first points. Those points
    scatter, and a nugget fitted to them can land anywhere. Start the lag at the typical spacing.

??? math "The math"
    For \(N(h)\) pairs of samples separated by the lag \(h\), the experimental variogram is

    \[ \gamma(h) = \frac{1}{2N(h)} \sum_{i=1}^{N(h)} \left( z(x_i) - z(x_i + h) \right)^2 \]

    A spherical model with nugget \(c_0\), sill \(c_0 + c\) and range \(a\) reaches its sill at \(h = a\).

## From variogram to estimate

!!! step "Step 1 — Compute and fit the variogram"
    Compute experimental variograms in several directions and fit one model to all of them:

    ```python
    import boitata as bt
    import numpy as np

    samples = bt.datasets.walker_lake()
    azimuths = np.arange(0, 180, 22.5)
    directional = [bt.experimental_variogram(samples, "V", 10.0, 120.0, azimuth=a) for a in azimuths]
    model = bt.Variogram.fit_directional(directional, [(a, 0) for a in azimuths], ["spherical", "spherical"])
    ```

!!! step "Step 2 — Choose the search"
    A `Search` sets which samples inform each target. Give it the major radius, the orientation and the ratio of the
    minor to the major axis:

    ```python
    search = bt.Search(radius=120, rotation=(30, 0, 0), ratios=(0.5, 0.5), max_samples=24, min_samples=4)
    ```

<figure class="bt-figure">
--8<-- "svg/proto-search.svg"
<figcaption><b>Figure 2.</b> A search ellipse with a 120 m major radius at azimuth 30° and a minor radius half as long. Samples outside it, even close ones across the short axis, do not inform the target.</figcaption>
</figure>

!!! step "Step 3 — Estimate"
    Estimators follow `fit` then `predict`, so they drop into the same pipelines as the transforms:

    ```python
    grid = bt.BlockModel(origin=(0.5, 0.5), size=(5, 5), count=(52, 60))
    ok = bt.OrdinaryKriging(model, search).fit(samples, "V")
    estimate, variance = ok.predict(grid, return_variance=True)
    ```

### Support changes the histogram

Block values average many points, so they spread less than the samples. The mean stays put. A resource model
reports blocks, so check its histogram against the spread expected at block support.

<figure class="bt-figure">
--8<-- "svg/proto-support.svg"
<figcaption><b>Figure 3.</b> One simulated grade field read at two supports. The 10 m block averages keep the mean of the 1 m points and lose the long upper tail.</figcaption>
</figure>

??? tryit "Try it"
    Double the minor radius in Step 2 by setting `ratios=(1.0, 1.0)`. Which samples in Figure 2 join the search, and
    how does the estimate map change?

    ??? answer "Answer"
        The ellipse becomes a 120 m circle and takes in the samples across the short axis. The estimates lose part
        of their elongation along azimuth 30°.

!!! check "Check before you move on"
    - You can point to the nugget, sill and range on a variogram plot.
    - You can say why a sample 70 m away may fall outside the search while one 110 m away falls inside.
    - You know which of `fit` and `predict` takes the samples.

## Choosing an estimator

<div class="bt-compare" markdown>

| Estimator | Uses the variogram | Mean | Reports a variance |
| --- | --- | --- | --- |
| Inverse distance | No | Local, implicit | No |
| Simple kriging | Yes | Known, global | Yes |
| Ordinary kriging | Yes | Unknown, local | Yes |

</div>

Ordinary kriging forces its weights to sum to one, which removes the need to know the mean. Simple kriging takes that mean as
an input.

<div class="grid cards bt-api-map" markdown>

-   **Describe continuity**

    ---

    [`experimental_variogram`](../api/variography/experimental_variogram.md) computes the pairs;
    [`Variogram`](../api/variography/Variogram.md) holds the fitted model.

-   **Estimate**

    ---

    [`Search`](../api/estimation/Search.md) sets the neighborhood;
    [`OrdinaryKriging`](../api/estimation/OrdinaryKriging.md) fits and predicts.

-   **Check**

    ---

    `cross_validate` on a fitted estimator re-estimates each sample with itself left out.

</div>

!!! seealso "See also"
    - [Ordinary kriging example](../examples/06-kriging/01-ordinary-kriging/example_06_01.md)
    - [Samples and support](../learn/01-samples-and-support/learn_01.md)
    - [How Boitatá is organized](../guide/organization.md)

### Chapter trail

Learn pages open with a chapter trail, which the build injects. This page shows a static copy:

<ul class="bt-trail">
<li class="done"><a href="#">1 Samples and support</a></li>
<li class="here"><span>2 Describing data</span></li>
<li><span>3 Spatial continuity</span></li>
<li><span>4 Kriging</span></li>
</ul>
