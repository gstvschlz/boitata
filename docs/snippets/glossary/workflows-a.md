*[solid]: a closed triangle mesh that bounds a volume, such as an ore body or a vein.
*[solids]: closed triangle meshes that each bound a volume, such as ore bodies or veins.
*[wireframe]: a triangle mesh drawn as its edges; a closed one is a solid.
*[partial block]: a block that a solid's surface crosses, so it lies partly inside and partly outside.
*[discretization]: splitting a block into a regular grid of points to measure what fraction lies inside a solid.
*[parent block]: a block of the regular grid a sub-blocked model starts from.
*[sub-block]: a smaller cell a parent block is split into where a contact crosses it.
*[sub-blocks]: smaller cells a parent block is split into where a contact crosses it.
*[sub-cell]: one cell of the regular sub-grid inside a parent block; neighbors with one label merge into a sub-block.
*[misplaced volume]: labeled volume outside its solid plus solid volume outside its labeled blocks.
*[winding]: the order of a triangle's corners; counter-clockwise seen from outside makes its normal point out.
*[boundary edge]: a mesh edge used by only one triangle, the rim of a hole.
*[non-manifold edge]: a mesh edge shared by three or more triangles.
*[degenerate face]: a triangle with a repeated corner or zero area.
