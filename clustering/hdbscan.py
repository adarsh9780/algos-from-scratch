import numpy as np


def get_distance_matrix(data: np.array) -> np.array:
    cols = data[:, np.newaxis]
    rows = data[np.newaxis, :]

    return np.abs(cols - rows)


def get_core_distance(distance_matrix: np.array, k: int = 2) -> list:
    sorted_distances = np.sort(distance_matrix, axis=1)
    return sorted_distances[:, k]


def get_mrd_matrix(data: np.array) -> np.array:
    d = get_distance_matrix(data)
    c = get_core_distance(d)
    cols = c[:, np.newaxis]
    rows = c[np.newaxis, :]

    x = np.maximum(cols, rows)
    out = np.maximum(x, d)

    return out


def get_mst(mrd: np.array, npoints: int) -> list:
    visiting_nodes = [0]  # start with first point in the data, index = 0
    remaining_nodes = [i for i in range(1, npoints)]

    mst = []
    while remaining_nodes:
        mask = np.full(mrd.shape, np.inf)
        mask[np.ix_(visiting_nodes, remaining_nodes)] = mrd[
            np.ix_(visiting_nodes, remaining_nodes)
        ]

        min_value_idx_flat = np.argmin(mask)
        x, y = np.unravel_index(min_value_idx_flat, mrd.shape)

        m = [x, y, mrd[x, y]]
        mst.append(m)

        visiting_nodes.append(y)
        remaining_nodes.remove(y)

    mst = sorted(mst, key=lambda x: x[2])

    return mst


def get_linkage_matrix(mst: list, npoints: int) -> list:
    cnames = {i: i for i in range(npoints)}  # index: name
    csizes = {i: 1 for i in range(npoints)}  # name: size
    new_name = npoints

    linkage_matrix = []

    for edge in mst:
        p1, p2, d = edge
        p1name = cnames[p1]
        p2name = cnames[p2]

        if p1name == p2name:
            continue

        new_size = csizes[p1name] + csizes[p2name]
        csizes[new_name] = new_size
        for pt, name in cnames.items():
            if (name == p1name) or (name == p2name):
                cnames[pt] = new_name

        linkage_matrix.append([new_name, p1name, p2name, float(d), new_size])
        new_name += 1

    return linkage_matrix, csizes


def get_stability_score(linkage_matrix: list, csizes: dict, min_cluster_size: int):
    # sort the linkage matrix based on size
    lm = linkage_matrix[::-1]
    last_index = lm[0][0]

    active = {last_index: {"birth": 0, "stab": 0}}
    finished = {}
    parents = {}

    for row in lm:
        cluster_index, c1, c2, distance, _ = row

        if cluster_index not in active:
            continue

        c1size = csizes[c1]
        c2size = csizes[c2]
        scores = active[cluster_index]
        stab = scores["stab"]
        birth = scores["birth"]
        leave = 1 / distance

        # stab += (npoints leaving) * (leave - birth)

        if (c1size < min_cluster_size) and (c2size < min_cluster_size):
            stab += (c1size + c2size) * (leave - birth)
            finished[cluster_index] = {"birth": birth, "stab": stab}
        elif (c1size >= min_cluster_size) and (c2size >= min_cluster_size):
            stab += (c1size + c2size) * (leave - birth)
            active[c1] = {"birth": leave, "stab": 0}
            active[c2] = {"birth": leave, "stab": 0}
            finished[cluster_index] = {"birth": birth, "stab": stab}
            parents[c1] = cluster_index
            parents[c2] = cluster_index
        else:
            surviving_cluster = c1 if c1size > c2size else c2
            min_size = min(c1size, c2size)
            stab += (min_size) * (leave - birth)
            active[surviving_cluster] = {"birth": birth, "stab": stab}
            parents[surviving_cluster] = parents.get(cluster_index)

        active.pop(cluster_index)

    return finished, parents


def eom(finished: dict, parents: dict) -> set:
    # 1. Build an adjacency map: parent -> list of children
    children_of = {}
    for child, parent in parents.items():
        if parent is not None and child in finished:
            children_of.setdefault(parent, []).append(child)

    # 2. Track stability (starts with individual cluster stability)
    prop_stab = {cid: data["stab"] for cid, data in finished.items()}
    selected_clusters = set()

    # 3. Process bottom-up (lowest ID up to root)
    for cluster in sorted(finished.keys()):
        kids = children_of.get(cluster, [])

        if not kids:
            # Leaf cluster: select it by default
            selected_clusters.add(cluster)
        else:
            kids_total = sum(prop_stab[k] for k in kids)
            if kids_total > finished[cluster]["stab"]:
                # Children win: propagate their score up
                prop_stab[cluster] = kids_total
            else:
                # Parent wins: select parent, discard all its descendants
                selected_clusters.add(cluster)
                selected_clusters.difference_update(kids)

    return selected_clusters


def get_points(look_up: int, kids: dict, npoints: int):
    # base case
    if look_up < npoints:
        return [look_up]

    k1, k2 = kids[look_up]

    return get_points(k1, kids, npoints) + get_points(k2, kids, npoints)


def give_label(final: set, linkage_matrix: list, npoints: int):
    kids = {row[0]: (row[1], row[2]) for row in linkage_matrix}
    labels = {i: -1 for i in range(npoints)}
    for cluster in final:
        ps = get_points(cluster, kids, npoints)
        for p in ps:
            labels[p] = cluster

    return labels


if __name__ == "__main__":
    data = np.array(
        [
            # Quirk 1: Duplicate points inside a dense cluster
            1.0,
            2.0,
            2.0,
            3.0,
            4.0,
            # Quirk 2: The "Bridge" point
            17.0,
            # Quirk 3: Very sparse cluster (spacing 5.0 vs 1.0)
            35.0,
            40.0,
            45.0,
            50.0,
            55.0,
            # Quirk 4: The "Micro-cluster" trap
            90.0,
            90.5,
            # Outlier
            200.0,
        ],
        dtype=float,
    )
    npoints = data.shape[0]
    mrd = get_mrd_matrix(data)
    mst = get_mst(mrd, npoints)
    linkage_matrix, csizes = get_linkage_matrix(mst, npoints)
    finished, parents = get_stability_score(
        linkage_matrix, min_cluster_size=4, csizes=csizes
    )
    final = eom(finished, parents)
    labels = give_label(final, linkage_matrix, npoints)
