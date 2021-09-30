import numpy


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


def make_nodes(circ, dict_constraints, proj, node_method, **node_kwargs):
    import importlib
    import pandas
    node_func = importlib.import_module(node_method)
    node_assoc, node_vol = node_func.make_nodes(circ, proj=proj, **node_kwargs)
    if isinstance(node_assoc.dtype, pandas.core.dtypes.dtypes.CategoricalDtype):
        node_assoc = pandas.Series(node_assoc.values.add_categories([str_void, str_external]),
                                   index=node_assoc.index)

    invalidate_invalids(node_assoc, circ, dict_constraints)
    add_externals(node_assoc, circ, proj)
    if str_void not in node_vol:
        node_vol[str_void] = 1.0
    if str_external not in node_vol:
        node_vol[str_external] = 1.0

    return node_assoc, node_vol
