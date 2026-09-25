# 6. Drillholes

5 277 holes with collar, survey (dip positive down, azimuth clockwise from north) and assay tables.
`Drillholes` desurveys every hole by minimum curvature and places samples and composites along the traces.

![holes](holes.png)

Compositing to 2 m regularizes support: most assays are 1 m, a few much longer.
Grades are length-weighted, composites never cross a domain change when one is given, and unsampled core is not counted as zero grade.

![compositing](compositing.png)

```python
dh = cs.Drillholes(collar, survey, assay)  # HOLEID, X, Y, Z, DEPTH, AZIMUTH, DIP, FROM, TO
samples = dh.samples()  # PointSet at interval midpoints
composites = dh.composite(2.0, ["ZN", "PB", "CU", "AG", "AU"])
paths = dh.paths()  # hole, depth, x, y, z of the traces
```

[`example.py`](example.py)
