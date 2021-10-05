import pandas, numpy

from scipy.spatial.transform import Rotation as R


def find_central_axis(circ, xyz, fm):
    from scipy.spatial import distance
    orient = circ.atlas.load_data("orientation")

    #  Find most central pixel in flatmap
    flat_coords = fm.lookup(xyz.values)
    flat_center = numpy.round(flat_coords.mean(axis=0)).astype(int)
    #  Find voxels associated with that pixel
    nz = numpy.nonzero(numpy.all(fm.raw == flat_center, axis=-1))
    nz = numpy.vstack(nz).transpose()
    #  Find most central of those voxels in 3d space
    min_idx = numpy.argmin(distance.cdist(nz, nz.mean(axis=0, keepdims=True)))
    center_xyz = nz[min_idx]
    center_direction = orient.lookup(center_xyz)
    return center_xyz, center_direction


def center_transform_volumes(tf_xyz, xy_idx, xz_orientation, resolution):
    """
    This one is a bit awkward. I found it hard to find a method that performs equally and comparably for all
    possible combinations of models / regions / parameters.
    Essentially, we know that the volume matched to a single pixel of the "center-transform" is a rectangular prism.
    Two dimensions of the prism are given by the "resolution" parameter. To get the third one we do the following:
    First, we calculate for each location its 'depth' coordinate. Then, we look up for each pixel the 'depth' of the
    locations that are matched to the pixel. Finally, we take the difference of the max and min of 'depth' values
    as the third dimension of said pixel.
    """
    orthogonal_orientation = numpy.array([-xz_orientation[1], xz_orientation[0]])
    depths = numpy.dot(tf_xyz[:, [0, 2]], orthogonal_orientation)
    per_pxl = pandas.Series(depths, index=pandas.MultiIndex.from_arrays(xy_idx.transpose(), names=["fx", "fy"]))
    per_pxl_min = per_pxl.groupby(["fx", "fy"]).apply(min)
    per_pxl_max = per_pxl.groupby(["fx", "fy"]).apply(max)
    per_pxl_depth = per_pxl_max - per_pxl_min + 25  # Minimal size 25 um, consistent with high-res volumes.
    per_pxl_vol = per_pxl_depth * resolution * resolution
    per_pxl_out = dict(zip(per_pxl_vol.index.values, per_pxl_vol.values))
    return per_pxl_out


def perform_center_transformation(xyz, center_xyz, center_direction, xz_orientation, resolution):
    lcl_xyz = xyz - center_xyz
    rot = R.from_quat(numpy.hstack([center_direction[1:], center_direction[0:1]]))  # local -> global
    rot_inv = rot.inv()  # global -> local
    tf_xyz = rot_inv.apply(lcl_xyz)
    xz_orientation = numpy.array(xz_orientation).reshape((2, 1)) / numpy.linalg.norm(xz_orientation)
    final_x = numpy.dot(tf_xyz[:, [0, 2]], xz_orientation)
    final_xy = numpy.hstack([final_x, tf_xyz[:, [1]]])

    xbins = numpy.arange(final_xy[:, 0].min(), final_xy[:, 0].max() + resolution, resolution)
    ybins = numpy.arange(final_xy[:, 1].min(), final_xy[:, 1].max() + resolution, resolution)
    x_idx = numpy.digitize(final_xy[:, 0], bins=xbins) - 1
    y_idx = numpy.digitize(final_xy[:, 1], bins=ybins)
    xy_idx = numpy.hstack([x_idx, y_idx])
    pxl_vols = center_transform_volumes(tf_xyz, xy_idx, xz_orientation, resolution)
    return xy_idx, pxl_vols


def side_view_flat_coords(circ, target, fm, xz_orientation, resolution, volume_mask=None):
    xyz = circ.cells.get(group=target, properties=['x', 'y', 'z'])
    if volume_mask is not None:
        import voxcell
        msk = voxcell.VoxelData.load_nrrd(volume_mask)
        xyz = xyz.loc[msk.lookup(xyz.values)]

    center_xyz, center_direction = find_central_axis(circ, xyz, fm)
    xy_bins, pxl_vols = perform_center_transformation(xyz, center_xyz, center_direction, xz_orientation, resolution)
    xy_out = pandas.DataFrame(xy_bins, columns=["fx", "fy"], index=xyz.index)
    return xy_out, pxl_vols


def make_nodes(circ, target=None, volume_mask=None,
               proj=None, flatmap_fn=None, xz_orientation=None, resolution=25.0):
    """
    Creates a node association where nodes are generated by digitizing a flat "sideways" view. Sideways in this
    context means a slice-like view where the y-coordinate is orthogonal to layers.
    """
    assert flatmap_fn is not None, "Must specify a flatmap to use!"
    import voxcell
    print("Looking up neuron flat locations...")
    fm = voxcell.VoxelData.load_nrrd(flatmap_fn)
    if xz_orientation is None:
        xz_orientation = numpy.array([1.0, 0.0])
    node_assoc, pxl_volumes = side_view_flat_coords(circ, target, fm, xz_orientation, resolution,
                                                    volume_mask=volume_mask)
    return node_assoc, pxl_volumes
