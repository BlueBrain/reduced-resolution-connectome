# reduced-resolution-connectome

Generate connection matrices of a model with reduced resolution, such as voxel-to-voxel, or region-to-region

Values in the resulting connection matrix will be a volumetric density of synapses between pairs of nodes.
"Node" in this context means a group of neurons that are grouped together by an algorithm (see below).

Usage: The main script to execute is reduced-resolution/count_connections.py:
count_connections.py config.json path/to/CircuitConfig

All configuration is performed in a json-encoded config file:

# Configuration file
All entries are optional with a "reasonable" default value

"output_filename" (default=None): path to where the results go (pandas.to_pickle encoded). If None, the results are written in stdout.

"projection" (default=None): If specified, then the synapses in that projection are used (for example, a white-matter-like connectome).
Otherwise, the regular connectome is used.

"constraints" (default={"synapse_class": "EXC"}): A dict of property names and valid values that a neuron has to fulfill to be considered. 
For example, the default limits neurons to excitatory classes. If more than one key/value pair is provided then a neuron has to fulfill all of them.
The empty dict allows every neuron to be considered. Exceptionally, if a key is named "target" then the value is evaluated as the name of a cell target
that is evaluated as circ.cells.ids(value). Only neurons in that cell target will be considered. 

"node_method": Specifies how neurons are associated with nodes of the output connectome / graph. Explained in detail below.

## Node methods
A node method is an algorithm that associates each neuron (that passes the "constraints" above) with a "node". Each "node" is
one unit in the output connectome. This can for example be a group of voxels, but also an m-type, or an e-type.

### Node method specifications
The value of "node_method" must be a dict with two entries:

"function": Name of the python file that contains the algorithm that decides for each neuron the node it is going to be associated with.
"kwargs": Additional kwargs to be provided to the algorithm. Their nature depends on the value of "function".

Currently three node methods are provided in this repository.

### Existing node methods:
####nodes_by_region

One of the simplest methods. Simply, each neuron is associated with the "region" it is contained in. Resulting connectivity is the density of connections between the regions.

additional kwargs:  None

####nodes_by_fm_pixels

Groups neurons together by the pixel of a flatmap they are mapped to. Resolution of the flatmap can be dynamically over- or under-sampled

additional kwargs: 
  - flatmap_fn (Required): Path to the .nrrd flatmap to use. The volume in the .nrrd file must associate each voxel with an integer-valued 2d coordinate.
  - volume_mask (Optional): Path to a bool-valued .nrrd volume. Only the subvolume of "True"-valued voxels will be considered.
If not specified, then the entire atlas volume will be considered.
  - subsample (int, Optional): If specified, it reduced the resolution of the flatmap by grouping together n by n pixels (n=subsample).
This can be used to reduce the number of resulting nodes and decrease the resolution of the connectome further.
  - supersample (int, Optional): The opposite of subsample. If specified, increases the resolution of the flatmap by the specified factor.
If specified, the value of "subsample" is ignored. This increases the number of nodes.

####nodes_by_side_view

Groups neurons together similarly to nodes_by_fm_pixels, but instead of using pixels of a "top-down" view, it will use a sideway view, i.e. the layers remain visible.
It achieves this in the following way: First, find the center of the volume of interest in the provided flatmap. Second, out of all voxels mapped to the most central pixel, find the one that is most central in 3d coordinates.
Third, get the 3d coordinates of the center of this most central voxel and the orientation associated with it. 
Fourth, use the extracted center and orientation to define a local, orthonormal coordinate system with origin at the center and y-axis orthogonal to layers.
Fifth calculate two coordinates: The first is a linear combination of the local x and z coordinates, the second is the local y coordinate.
Sixth, bin the two coordinates with the provided resolution.

additonal kwargs:
  - flatmap_fn (Required): Path to the .nrrd flatmap to use. The volume in the .nrrd file must associate each voxel with an integer-valued 2d coordinate. Used the find the center of the volume of interest.
  - volume_mask (Optional): Path to a bool-valued .nrrd volume. Only the subvolume of "True"-valued voxels will be considered.
If not specified, then the entire atlas volume will be considered.
  - xz_orientation: (Optional, default: [1, 0]): Specifies the angle at which the x- and z-coordinates are flattened (coordinates that are largely parallel to the layers).
This can be thought of as the angle at which you "slice" the volume. For example, the default value of [1, 0] will use the x-coordinate and discard the z-coordinate.
  - resolution: (Optional, default: 25.0): Resolution in um at which to bin the resulting "sideway" coordinates.
