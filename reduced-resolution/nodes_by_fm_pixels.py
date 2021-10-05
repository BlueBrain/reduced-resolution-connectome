import importlib
import tqdm
import pandas, numpy

from scipy.spatial.transform import Rotation as R


def fm_pixel_volumes_subsampled(fm, n_subsampled):
    print("Calculating pixel volumes...")
    fmraw = numpy.floor(fm.raw.reshape((-1, 2)) / n_subsampled).astype(int)
    raw_volumes = pandas.DataFrame(fmraw, columns=["x", "y"]).value_counts() * fm.voxel_volume
    reg_volumes = dict(zip(raw_volumes.index.values, raw_volumes.values))
    return reg_volumes


def fm_pixel_volumes_supersampled(fm, node_assoc, n_supersampled):
    print("Calculating pixel volumes...")
    fmraw = numpy.floor(fm.raw.reshape((-1, 2))).astype(int)
    raw_volumes = pandas.DataFrame(fmraw, columns=["fx", "fy"]).value_counts() * fm.voxel_volume
    u_C = node_assoc.drop_duplicates()
    base_C = (u_C / n_supersampled).apply(numpy.floor).astype(int)
    out_volumes = raw_volumes[list(map(tuple, base_C.values))] / (n_supersampled ** 2)
    out_volumes.index = pandas.MultiIndex.from_frame(u_C)
    reg_volumes = dict(zip(out_volumes.index.values, out_volumes.values))
    return reg_volumes


def supersample_pixel(lcl_xyz, o_vec, nbins):
    rot = R.from_quat(numpy.hstack([o_vec[1:], o_vec[0:1]]))
    rot_x = rot.apply(numpy.array([1, 0, 0]))
    rot_z = rot.apply(numpy.array([0, 0, 1]))
    Ax = numpy.dot(lcl_xyz, rot_x)
    Az = numpy.dot(lcl_xyz, rot_z)
    bins_x = numpy.percentile(Ax, numpy.linspace(0, 100, nbins + 1))
    bins_x[-1] = bins_x[-1] + 1E-12
    bins_z = numpy.percentile(Az, numpy.linspace(0, 100, nbins + 1))
    bins_z[-1] = bins_z[-1] + 1E-12
    Ox = numpy.digitize(Ax, bins_x) - 1
    Oz = numpy.digitize(Az, bins_z) - 1
    return numpy.vstack([Ox, Oz]).transpose()


def supersampled_flat_coords(circ, target, fm, orient, n_supersampled, volume_mask=None):
    xyz = circ.cells.get(group=target, properties=['x', 'y', 'z'])
    if volume_mask is not None:
        import voxcell
        msk = voxcell.VoxelData.load_nrrd(volume_mask)
        xyz = xyz.loc[msk.lookup(xyz.values)]

    flat_coords = pandas.DataFrame(fm.lookup(xyz.values), columns=["fx", "fy"], index=xyz.index)
    flat_lo = pandas.Series(flat_coords.index, index=pandas.MultiIndex.from_frame(flat_coords))
    per_pixel_lo = flat_lo.groupby(["fx", "fy"]).apply(list)
    ss_coords = pandas.DataFrame(numpy.zeros_like(flat_coords.values),
                                 index=flat_coords.index,
                                 columns=flat_coords.columns)
    print("Supersampling...")
    for i in tqdm.tqdm(per_pixel_lo.index):
        idxx = per_pixel_lo.loc[i]
        pxl_xyz = xyz.loc[idxx]
        lcl_xyz = pxl_xyz - pxl_xyz.mean(axis=0)
        o_vec = orient.lookup(pxl_xyz.mean(axis=0).values)
        ss_coords.loc[idxx] = supersample_pixel(lcl_xyz, o_vec, n_supersampled)
    node_assoc = flat_coords * n_supersampled + ss_coords
    return node_assoc


def subsampled_flat_coords(circ, target, fm, n_subsampled, volume_mask=None):
    xyz = circ.cells.get(group=target, properties=["x", "y", "z"])
    if volume_mask is not None:
        import voxcell
        msk = voxcell.VoxelData.load_nrrd(volume_mask)
        xyz = xyz.loc[msk.lookup(xyz.values)]
    node_assoc = pandas.DataFrame(numpy.floor(fm.lookup(xyz.values) / n_subsampled).astype(int),
                                  index=xyz.index, columns=["fx", "fy"])
    return node_assoc


def make_nodes(circ, target=None, volume_mask=None, proj=None,
               flatmap_fn=None, subsample=2, supersample=None):
    """
    Creates a node association where one node exists per n x n block of flatmap pixels (n=subsample)
    """
    assert flatmap_fn is not None, "Must specify a flatmap to use!"
    import voxcell
    print("Looking up neuron flat locations...")
    fm = voxcell.VoxelData.load_nrrd(flatmap_fn)
    if supersample is not None:
        orient = circ.atlas.load_data("orientation")
        node_assoc = supersampled_flat_coords(circ, target, fm, orient, supersample, volume_mask=volume_mask)
        pxl_volumes = fm_pixel_volumes_supersampled(fm, node_assoc, supersample)
    elif subsample is not None:
        node_assoc = subsampled_flat_coords(circ, target, fm, subsample, volume_mask=volume_mask)
        pxl_volumes = fm_pixel_volumes_subsampled(fm, subsample)
    else:
        node_assoc = subsampled_flat_coords(circ, target, fm, 1, volume_mask=volume_mask)
        pxl_volumes = fm_pixel_volumes_subsampled(fm, 1)

    # node_assoc = node_assoc.apply(tuple, result_type="reduce", axis=1)  next line is faster
    node_assoc = pandas.Series(map(tuple, node_assoc.values), index=node_assoc.index).apply(str)
    node_assoc = pandas.Series(pandas.Categorical(node_assoc.values), index=node_assoc.index)
    return node_assoc, pxl_volumes
