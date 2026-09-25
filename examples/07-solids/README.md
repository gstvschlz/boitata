# 7. Solids and block models

A wireframe bounds a domain. Here an ellipsoid is fitted to the Zn > 5 % composites of the cluster seen in [chapter 6](../06-drillholes/README.md).
`Mesh.contains` tests points by generalized winding number; `Mesh.proportion` samples 4 × 4 × 4 points in each block.

![solid](solid.png)

Check: block proportions sum to 6.78 million m³ against 6.83 million m³ for the exact ellipsoid.
Composites inside average 3.8 % Zn against 2.9 % outside.
Blocks more than half inside form a masked `BlockModel`; `block_shell` returns its visible faces for plotting.

```python
solid = cs.Mesh(vertices, triangles)
inside = solid.contains(composites.coords)
blocks = cs.BlockModel(origin=lo, size=(10, 10, 10), count=count)
proportion = solid.proportion(blocks)
ore = blocks.with_column("inside", proportion).mask(proportion > 0.5)
vertices, triangles, _ = cs.block_shell(ore)
```

[`example.py`](example.py)
