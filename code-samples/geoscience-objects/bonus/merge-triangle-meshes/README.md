# Merge Triangle Meshes
This folder contains a Jupyter notebook showing how to select any number of existing triangle mesh objects from an Evo workspace, merge them into a single mesh, and publish the result either as a new triangle mesh object or as a new version of an existing triangle mesh object.

## Inputs
Two or more [triangle mesh](https://developer.seequent.com/docs/data-structures/geoscience-objects/schemas/objects/triangle-mesh/) objects in the selected Evo workspace. The notebook lists every triangle mesh in the workspace and you tick the ones to merge.

All selected meshes must use the same coordinate reference system, as the notebook does not reproject data. A mesh with an unspecified coordinate reference system cannot be merged with a mesh that has one.

## Outputs
- A new triangle mesh object with the name and path you provide, containing the vertices and triangles of all selected meshes. The triangle indices of each mesh are offset by the total number of vertices in the meshes before it.
- Alternatively, tick **Update existing merged mesh** and select a target mesh to publish the merge as a new version of that mesh. The target keeps its ID, name, and path. Meshes ticked for merging cannot be chosen as the target.

Only the mesh geometry is merged. Attributes, parts, and edges of the source meshes are not carried over.
