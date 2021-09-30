import importlib
import pandas, numpy


def fm_pixel_volumes(fm, subsample):
    print("Calculating pixel volumes...")
    fmraw = numpy.floor(fm.raw.reshape((-1, 2)) / subsample).astype(int)
    raw_volumes = pandas.DataFrame(fmraw, columns=["x", "y"]).value_counts() * fm.voxel_volume
    reg_volumes = dict(zip(raw_volumes.index.values, raw_volumes.values))
    return reg_volumes


def make_nodes(circ, proj=None, flatmap_fn=None, subsample=2):
    """
    Creates a node association where one node exists per n x n block of flatmap pixels (n=subsample)
    """
    import voxcell
    print("Looking up neuron flat locations...")
    fm = voxcell.VoxelData.load_nrrd(flatmap_fn)
    xyz = circ.cells.get(properties=["x", "y", "z"])
    node_assoc = pandas.Series(map(tuple, numpy.floor(fm.lookup(xyz.values) / subsample).astype(int)),
                               index=xyz.index)
    pxl_volumes = fm_pixel_volumes(fm, subsample)
    return node_assoc, pxl_volumes
