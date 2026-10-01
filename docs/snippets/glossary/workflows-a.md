*[solid]: A closed triangle mesh that bounds a volume, such as an ore body or a vein.
*[solids]: Closed triangle meshes that each bound a volume, such as ore bodies or veins.
*[wireframe]: A triangle mesh drawn as its edges; a closed one is a solid.
*[partial block]: A block that a solid's surface crosses, so it is partly inside and partly outside.
*[discretization]: Splitting a block into a regular grid of points to measure what fraction lies inside a solid.
*[parent block]: A block of the regular grid a sub-blocked model starts from.
*[sub-block]: A smaller cell a parent block is split into where a contact crosses it.
*[sub-blocks]: Smaller cells a parent block is split into where a contact crosses it.
*[sub-cell]: One cell of the regular sub-grid inside a parent block; neighbors with one label merge into a sub-block.
*[misplaced volume]: Labeled volume outside its solid plus solid volume outside its labeled blocks.
*[winding]: The order of a triangle's corners; counter-clockwise seen from outside makes its normal point out.
*[boundary edge]: A mesh edge used by only one triangle, the rim of a hole.
*[non-manifold edge]: A mesh edge shared by three or more triangles.
*[degenerate face]: A triangle with a repeated corner or zero area.
