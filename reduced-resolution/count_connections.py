#!/usr/bin/env python
import numpy
import h5py
import pandas
from tqdm import tqdm

import importlib

node_commons = importlib.import_module("node_creation")
str_void = node_commons.str_void


def synapse_counts_complete(h5, reg_lookup, chunks=10000):
    L = h5['source_node_id'].shape[0]
    splits = numpy.arange(0, L + chunks, chunks)

    midxx = pandas.MultiIndex.from_tuples([], names=["Source node", "Target node"])
    counts = pandas.Series([], index=midxx, dtype=int)

    for splt_fr, splt_to in tqdm(zip(splits[:-1], splits[1:]), desc="Counting...", total=len(splits) - 1):
        son_idx_fr = h5['source_node_id'][splt_fr:splt_to]
        son_idx_to = h5['target_node_id'][splt_fr:splt_to]
        reg_fr = reg_lookup[son_idx_fr + 1]
        reg_to = reg_lookup[son_idx_to + 1]
        new_counts = pandas.DataFrame({"Source node": reg_fr.values,
                                       "Target node": reg_to.values}).value_counts()
        counts = counts.add(new_counts, fill_value=0)
    for lvl, nm in zip(counts.index.levels, counts.index.names):
        if str_void in lvl:
            counts = counts.drop(str_void, level=nm)
    return counts


def synapse_counts_subtarget(h5, reg_lookup, chunks=100000):
    midxx = pandas.MultiIndex.from_tuples([], names=["Source node", "Target node"])
    counts = pandas.Series([], index=midxx, dtype=int)
    buffer_fr = []; buffer_to = []

    use_gids = reg_lookup.index.values[reg_lookup != str_void]

    for gid in tqdm(use_gids):
        rnge = h5["indices"]["target_to_source"]["node_id_to_ranges"][gid - 1]
        for r in h5["indices"]["target_to_source"]["range_to_edge_id"][rnge[0]:rnge[1]]:
            son_idx_fr = h5["source_node_id"][r[0]:r[1]]
            son_idx_to = h5["target_node_id"][r[0]:r[1]]
            reg_fr = reg_lookup[son_idx_fr + 1]
            reg_to = reg_lookup[son_idx_to + 1]
            buffer_fr.append(reg_fr.values); buffer_to.append(reg_to.values)
        if len(buffer_fr) >= chunks:
            new_counts = pandas.DataFrame({"Source node": numpy.hstack(buffer_fr),
                                           "Target node": numpy.hstack(buffer_to)}).value_counts()
            counts = counts.add(new_counts, fill_value=0)
            buffer_fr = []; buffer_to = []
    if len(buffer_fr) > 0:
        new_counts = pandas.DataFrame({"Source node": numpy.hstack(buffer_fr),
                                       "Target node": numpy.hstack(buffer_to)}).value_counts()
        counts = counts.add(new_counts, fill_value=0)
    for lvl, nm in zip(counts.index.levels, counts.index.names):
        if str_void in lvl:
            counts = counts.drop(str_void, level=nm)
    return counts


def count(circ, dict_constraints, node_method, node_kwargs, projection=None):
    if projection is None:
        h5 = h5py.File(circ.config["connectome"], "r")['edges/default']
        proj = None
    else:
        print("Using projection {0}...".format(projection))
        proj = circ.projection(projection)
        h5 = h5py.File(proj.metadata["Path"], "r")['edges/default']

    node_assoc, node_vol = node_commons.make_nodes(circ, dict_constraints, proj, node_method, **node_kwargs)

    if (node_assoc == "_VOID").mean() > 0.75:
        print("For subtarget...")
        raw_counts = synapse_counts_subtarget(h5, node_assoc, chunks=50000)
    else:
        print("For all neurons...")
        raw_counts = synapse_counts_complete(h5, node_assoc, chunks=5000000)

    idx_tgt = raw_counts.index.names.index("Target node")
    for reg_tuple in raw_counts.index:
        raw_counts[reg_tuple] /= node_vol[reg_tuple[idx_tgt]]

    raw_counts.name = "Density"
    out = pandas.concat([raw_counts.index.to_frame(), raw_counts], axis=1)
    return out


def main():
    import sys
    import bluepy
    import getopt
    import json

    opts, args = getopt.getopt(sys.argv[1:], "")

    if len(args) != 2:
        print("Usage: {0} config.json CircuitConfig".format(__file__))
        sys.exit(2)
    fn_circ = args[1]
    fn_cfg = args[0]
    with open(fn_cfg, "r") as fid:
        cfg = json.load(fid)

    fn_out = cfg.get("output_filename", None)
    node_method = cfg.get("node_method", {"function": "nodes_by_region", "kwargs": {}})
    proj_name = cfg.get("projection", None)
    neuron_constraints = cfg.get("constraints", {"synapse_class": "EXC"})

    circ = bluepy.Circuit(fn_circ)
    res = count(circ, neuron_constraints, node_method["function"], node_method.get("kwargs", {}),
                projection=proj_name)
    if fn_out is None:
        print(res)
    else:
        res.to_pickle(fn_out)


if __name__ == "__main__":
    main()
