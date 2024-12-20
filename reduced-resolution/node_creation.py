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


def constrain_mask(gids, circ, mask_dict):
    import voxcell
    fn = mask_dict["file"]
    if "root" in mask_dict:
        import os
        if mask_dict["root"] == "_atlas":
            raise NotImplementedError()  # circ.atlas.dir_path?
        else:
            fn = os.path.join(mask_dict["root"], fn)
    msk = voxcell.VoxelData.load_nrrd(fn)
    msk_vals = msk.lookup(circ.cells.get(gids, properties=["x", "y", "z"]).values)
    return msk_vals


def constrain_neurons(gids, circ, dict_constraints):
    keys = list(dict_constraints.keys())
    props = circ.cells.get(gids, properties=keys)
    valid = numpy.ones(len(gids), dtype=bool)
    for k, v in dict_constraints.items():
        if k == "mask":
            valid = valid & constrain_mask(gids, circ, v)
        elif isinstance(v, list):
            valid = valid & numpy.in1d(props[k].values, v)
        else:
            valid = valid & (props[k].values == v)
    return valid


def invalidate_invalids(node_association, circ, dict_constraints):
    # Step 1: Set value for neurons not matching the filters to str_void
    valid = constrain_neurons(node_association.index.values, circ, dict_constraints)
    node_association[~valid] = str_void
    # Step 2: Set value for neurons not contained in the association Series to str_void
    all_gids = circ.cells.ids()
    non_listed = numpy.setdiff1d(all_gids, node_association.index.values)
    node_association = pandas.concat([node_association,
                                      pandas.Series([str_void] * len(non_listed),
                                                    index=non_listed)], axis=0)
    return node_association


def add_externals(node_association, circ, proj):
    if proj is not None:
        proj_gids = circ.cells.ids(proj.metadata["Source"])
        external_gids = proj_gids[~numpy.in1d(proj_gids, node_association.index)]
        node_association = pandas.concat([node_association,
                                          pandas.Series([str_external] * len(external_gids),
                                                        index=external_gids)], axis=0)
    return node_association


def print_association_stats(node_assoc):
    print("Received node associations for {0} neurons...".format(len(node_assoc)))
    vc = node_assoc.value_counts()
    print("Number of nodes: {0}".format(len(node_assoc.drop_duplicates())))
    print("Neurons per value: {0} ({1}) +- {2}".format(vc.mean(), vc.median(), vc.std()))


def make_nodes(circ, dict_constraints, proj, node_method, **node_kwargs):
    import importlib
    import pandas
    base_tgt = dict_constraints.pop("target", None)
    node_func = importlib.import_module(node_method)
    node_assoc, node_vol = node_func.make_nodes(circ, target=base_tgt, proj=proj, **node_kwargs)
    if isinstance(node_assoc.dtype, pandas.core.dtypes.dtypes.CategoricalDtype):
        node_assoc = pandas.Series(node_assoc.values.add_categories([str_void, str_external]),
                                   index=node_assoc.index)

    print_association_stats(node_assoc)
    node_assoc = invalidate_invalids(node_assoc, circ, dict_constraints)
    node_assoc = add_externals(node_assoc, circ, proj)
    if str_void not in node_vol:
        node_vol[str_void] = 1.0
    if str_external not in node_vol:
        node_vol[str_external] = 1.0

    return node_assoc, node_vol
