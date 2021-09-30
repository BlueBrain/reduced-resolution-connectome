#!/usr/bin/env python
import numpy
import h5py
import pandas
from tqdm import tqdm

from .create_nodes import str_external, str_void
from .create_nodes import nodes_by_fm_pixels, nodes_by_brain_region


def synapses_region_counts(h5, reg_lookup, chunks=10000):
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


def count(circ, dict_constraints, projection=None, flatmap=None):
    if projection is None:
        h5 = h5py.File(circ.config["connectome"], "r")['edges/default']
        proj = None
    else:
        print("Using projection {0}...".format(projection))
        proj = circ.projection(projection)
        h5 = h5py.File(proj.metadata["Path"], "r")['edges/default']

    if flatmap is None:
        node_assoc, node_vol = nodes_by_brain_region(circ, dict_constraints, proj=proj)
    else:
        node_assoc, node_vol = nodes_by_fm_pixels(circ, dict_constraints, flatmap_fn=flatmap, proj=proj)
    raw_counts = synapses_region_counts(h5, node_assoc, chunks=5000000)
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

    opts, args = getopt.getopt(sys.argv[1:], "o:p:f:")
    opts = dict(opts)
    if len(args) != 1:
        print("Usage: {0} (-p projection_name) (-o output_filename) CircuitConfig".format(__file__))
        sys.exit(2)
    fn_circ = args[0]
    fn_out = opts.get("-o", None)
    proj_name = opts.get("-p", None)
    flatmap = opts.get("-f", None)

    circ = bluepy.Circuit(fn_circ)
    res = count(circ, {"synapse_class": "EXC"}, projection=proj_name, flatmap=flatmap)  # hard coded here.
    if fn_out is None:
        print(res)
    else:
        res.to_pickle(fn_out)


if __name__ == "__main__":
    main()
