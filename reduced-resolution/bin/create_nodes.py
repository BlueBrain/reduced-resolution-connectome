import numpy
import pandas


"""
Contains different methods to create lookups of 'node association'. That is, for each neuron information on
which node in the connectome it is associated with. A node association is a pandas.Series, indexed by neuron gids,
where values specify node identity, i.e. if the value associated with neurons A and B is identical, they will belong
to the same node. 
Additionally, there is a number of specific node associations: If the value is "_VOID" the neuron belongs to no
node and is skipped; the value "EXTERNAL" indicates an external input neuron and all external neurons will belong
to the same node.
A node association must have a value for each neuron in the model. If you want to exclude a neuron, set its value
to "_VOID".

Each method to generate a node association also returns a dict of volumes associated with each node
"""

str_void = "_VOID"
str_external = "EXTERNAL"


def constrain_neurons(circ, dict_constraints):
    keys = list(dict_constraints.keys())
    props = circ.cells.get(properties=keys)
    valid = numpy.ones(len(props), dtype=bool)
    for k, v in dict_constraints.items():
        if isinstance(v, list):
            valid = valid & numpy.in1d(props[k].values, v)
        else:
            valid = valid & (props[k].values == v)
    return valid


def invalidate_invalids(node_association, circ, dict_constraints):
    valid = constrain_neurons(circ, dict_constraints)
    # TODO: Used as a boolean index here. Must be in the same order! Should be checked
    node_association[~valid] = str_void


def add_externals(node_association, circ, proj):
    if proj is not None:
        proj_gids = circ.cells.ids(proj.metadata["Source"])
        external_gids = proj_gids[~numpy.in1d(proj_gids, node_association.index)]
        node_association[external_gids] = str_external


def region_volumes(circ, lst_regions):
    out = {}
    reg_map = circ.atlas.load_region_map()
    annotations = circ.atlas.load_data("brain_regions")
    for reg in lst_regions:
        out[reg] = annotations.volume(reg_map.find(reg, 'acronym', with_descendants=True))
    return out


def fm_pixel_volumes(fm, subsample):
    print("Calculating pixel volumes...")
    fmraw = numpy.floor(fm.raw.reshape((-1, 2)) / subsample).astype(int)
    raw_volumes = pandas.DataFrame(fmraw, columns=["x", "y"]).value_counts() * fm.voxel_volume
    reg_volumes = dict(zip(raw_volumes.index.values, raw_volumes.values))
    reg_volumes[str_void] = 1
    reg_volumes[str_external] = 1
    return reg_volumes


def nodes_by_brain_region(circ, dict_constraints, proj=None):
    """
    Creates a node association where one node exist per brain region.
    """
    print("Looking up neuron regions...")
    node_assoc = circ.cells.get(properties='region')
    invalidate_invalids(node_assoc, circ, dict_constraints)
    add_externals(node_assoc, circ, proj)
    print("Calculating region volumes...")
    reg_volumes = region_volumes(circ, node_assoc.drop_duplicates().values)
    return node_assoc, reg_volumes


def nodes_by_fm_pixels(circ, dict_constraints, flatmap_fn=None, proj=None, subsample=2):
    """
    Creates a node association where one node exists per n x n block of flatmap pixels (n=subsample)
    """
    import voxcell
    print("Looking up neuron flat locations...")
    fm = voxcell.VoxelData.load_nrrd(flatmap_fn)
    xyz = circ.cells.get(properties=["x", "y", "z"])
    node_assoc = pandas.Series(map(tuple, numpy.floor(fm.lookup(xyz.values) / subsample).astype(int)),
                               index=xyz.index)
    invalidate_invalids(node_assoc, circ, dict_constraints)
    add_externals(node_assoc, circ, proj)
    pxl_volumes = fm_pixel_volumes(fm, subsample)
    return node_assoc, pxl_volumes
