# 6. Drillholes

5 277 holes with collar, survey (dip positive down, azimuth clockwise from north), assay and geology tables.
`Drillholes` desurveys every hole by minimum curvature and places samples and composites along the traces.

![holes](holes.png)

Assays and lithology come in separate interval tables. `merge_intervals` splits both at every boundary so each piece carries its grades and its lithology; compositing by `LITH` then never averages across a contact.

![compositing](compositing.png)

Compositing to 2 m regularizes support: most assays are 1 m, some much longer. Grades are length-weighted, unsampled core is not read as zero, and composites with no assay are dropped.

![domains](domains.png)

```python
intervals = cs.merge_intervals(assay, geology)  # HOLEID, FROM, TO + columns of both
dh = cs.Drillholes(collar, survey, intervals)  # X, Y, Z, DEPTH, AZIMUTH, DIP
composites = dh.composite(2.0, ["ZN", "PB", "CU", "AG", "AU"], domain="LITH")
paths = dh.paths()  # hole, depth, x, y, z of the traces
```

[`example.py`](example.py)
