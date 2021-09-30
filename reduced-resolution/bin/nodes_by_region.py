
def region_volumes(circ, lst_regions):
    out = {}
    reg_map = circ.atlas.load_region_map()
    annotations = circ.atlas.load_data("brain_regions")
    for reg in lst_regions:
        out[reg] = annotations.volume(reg_map.find(reg, 'acronym', with_descendants=True))
    return out


def make_nodes(circ, proj=None):
    """
    Creates a node association where one node exist per brain region.
    """
    print("Looking up neuron regions...")
    node_assoc = circ.cells.get(properties='region')
    print("Calculating region volumes...")
    reg_volumes = region_volumes(circ, node_assoc.drop_duplicates().values)
    return node_assoc, reg_volumes
