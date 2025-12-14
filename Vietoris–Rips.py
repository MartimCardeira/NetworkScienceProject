from itertools import combinations

import numpy as np
import pandas as pd
import xgi
import gudhi
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection
import networkx as nx
from scipy.spatial.distance import cdist

#Define our experiment variables
n = 100
r = 0.1 #this will be the variable we change in the experiment
max_simplex_order = 2 #we'll limit the network to only triangles and below
np.random.seed(654277791) #perfect seed for point cloud, connectivity starts at r=0.130

def generate_random_points(n):
    """
    :param n: The number of points to sample
    :return: The point cloud, formatted as List[List[float]] for compatibility with GUDHI's RipsComplex.
    """
    points = np.random.rand(n, 2)
    points_formatted = [[row[0], row[1]] for row in points.tolist()]
    return points_formatted

def generate_rips_complex(points, max_simplex_order, r, verbose=False):
    """
    Create a Rips complex from a point cloud, as well as the corresponding simplex tree.
    :param points: The point cloud
    :param max_simplex_order: The maximum n-simplex to consider.
    :param r: The maximum edge length to consider.
    :param verbose: Print debugging information if True.
    :return: A tuple containing the Rips complex and the simplex tree respectively.
    """
    rips_complex = gudhi.RipsComplex(points=points, max_edge_length=r)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_simplex_order)
    if verbose:
        result_str = 'Rips complex is of dimension ' + repr(simplex_tree.dimension()) + ' - ' + \
                     repr(simplex_tree.num_simplices()) + ' simplices - ' + \
                     repr(simplex_tree.num_vertices()) + ' vertices.'
        print(result_str)
    return (rips_complex, simplex_tree)

def visualize(simplex_tree, r):
    """
    Plots the 1-skeleton of the simplex tree.
    :param simplex_tree: Used for getting the triangles and edges.
    :param r: For titling the plot.
    :return:
    """
    edges = []
    triangles = []
    for filtered_value in simplex_tree.get_filtration():
        if len(filtered_value[0]) == 2:
            edges.append(filtered_value[0])
        elif len(filtered_value[0]) == 3:
            triangles.append(filtered_value[0])

    fig, ax = plt.subplots(figsize=(6, 6))

    tri_coords = []
    for tri in triangles:
        tri_coords.append([points[tri[0]], points[tri[1]], points[tri[2]]])

    triangle_collection = PolyCollection(
        tri_coords,
        alpha=0.3,
        edgecolor=None,
    )
    ax.add_collection(triangle_collection)

    for u, v in edges:
        x = [points[u][0], points[v][0]]
        y = [points[u][1], points[v][1]]
        ax.plot(x, y, linewidth=1)

    points_np = np.array(points)
    ax.scatter(points_np[:, 0], points_np[:, 1], s=20, color="black")

    ax.set_aspect("equal")
    ax.set_title(f"Vietoris–Rips Complex (r={r:.3f})")
    plt.show()


def generate_simplicial_complex(simplex_tree):
    '''
    Turns the given simplex tree into XGI's SimplicialComplex representation, ready for Kuramoto simulation.
    :param simplex_tree: The simplex tree to convert.
    :return: The converted SimplicialComplex.
    '''
    simplices = []
    for filtered_value in simplex_tree.get_filtration():
        simplices.append(filtered_value[0])
    SC = xgi.SimplicialComplex(simplices)
    return SC


def run_kuramoto(SC, r_val, omega, theta0, sigma=0.4, T=30, n_steps=5000, graph=False):
    order = 2  # The order of the simplices we are applying oscillators to.
    (
        theta,
        theta_minus,
        theta_plus,
        om1_dict,
        o_dict,
        op1_dict,
    ) = xgi.synchronization.simulate_simplicial_kuramoto(
        SC, None, order, omega, sigma, theta0, T, n_steps, True
    )

    r = xgi.synchronization.compute_simplicial_order_parameter(theta_minus, theta_plus)

    if graph:
        fig, ax = plt.subplots(figsize=(6,4))

        ax.plot(np.linspace(0, T, n_steps), r)
        ax.set_ylabel("Order parameter")
        ax.set_xlabel("Time (s)")
        ax.set_title(f"Simplicial order parameter, Sigma value {sigma:.3f}")
        ax.set_ylim((0, 1))

        ax.set_title(f"Vietoris–Rips Complex (r={r_val:.3f}) with sigma={sigma:.3f}")
        plt.show()

    return r


def isSynchronized(values):
    r_threshold = 0.80
    std_threshold = 0.002
    if(np.std(values) > std_threshold):
        return False
    if(np.mean(values) < r_threshold):
        return False
    return True

def find_critical_sigma(SC, r, sigma_low=0.50, sigma_high=0.65, tol=0.0005, order=2):
    """
    Finds the critical sigma using a binary search between sigma_low and sigma_high.
    Assumes the system is unsynchronized at sigma_low and synchronized at sigma_high.
    """
    n = len(SC.edges.filterby("order", order))
    #if the upper bound is still not synchronized, increase it until it is
    while True:
        result = run_kuramoto(SC, r, omega[:n], theta0[:n], sigma=sigma_high)
        if isSynchronized(result[4000:5000]):
            break
        sigma_high += 0.05   # expand upper bound

    #binary search
    while sigma_high - sigma_low > tol:
        sigma_mid = 0.5 * (sigma_low + sigma_high)
        result = run_kuramoto(SC, r, omega[:n], theta0[:n], sigma=sigma_mid)

        if isSynchronized(result[4000:5000]):
            sigma_high = sigma_mid  # sync happens → lower the upper bound
        else:
            sigma_low = sigma_mid   # no sync → raise the lower bound

    # Return mid-point as estimate
    return 0.5 * (sigma_low + sigma_high)

def extract_vr_statistics(points, simplex_tree, SC):
    """
    Extracts geometric, graph-theoretic, simplicial, and topological statistics
    from a Vietoris-Rips complex.

    Returns a dictionary suitable for appending to a dataframe.
    """

    # ---------------------------
    # 0. Basic quantities
    # ---------------------------
    num_points = len(points)

    # Convert SC to 1-skeleton NetworkX graph
    # G = SC.get_skeleton(1).to_networkx()
    G = nx.Graph()
    for node in SC.nodes:
        G.add_node(node)

    for filtered_value in simplex_tree.get_filtration():
        if len(filtered_value[0]) == 2:
            G.add_edge(filtered_value[0][0], filtered_value[0][1])
            #edges.append(filtered_value[0])

    # Nodes / edges / triangles
    num_edges = G.number_of_edges()
    triangles = SC.edges.filterby("order", 2)
    num_triangles = len(triangles)

    # ---------------------------
    # 1. Simple graph statistics
    # ---------------------------
    degrees = np.array([deg for _, deg in G.degree()])
    avg_degree = float(np.mean(degrees))
    max_degree = float(np.max(degrees))
    density = float(nx.density(G))

    # Giant component size
    comps = list(nx.connected_components(G))
    giant_component = max(comps, key=len)
    giant_fraction = len(giant_component) / num_points

    # Algebraic connectivity (Fiedler value)
    try:
        L = nx.laplacian_matrix(G).astype(float).todense()
        eigs = np.linalg.eigvals(L)
        eigs_sorted = np.sort(eigs.real)
        fiedler = float(eigs_sorted[1])  # second-smallest eigenvalue
    except Exception:
        fiedler = np.nan  # Graph may be too small / disconnected

    # ---------------------------
    # 2. Triangle stats
    # ---------------------------

    #triangle degree per edge (number of triangles incident to an edge)
    tri = {}
    for u, v in G.edges():
        # intersection of neighbor sets = common neighbors
        tri[(u, v)] = len(set(G[u]) & set(G[v]))

    values = np.array(list(tri.values()))

    avg_triangle_degree = values.mean()
    variance_triangle_degree = values.var(ddof=0)

    # triangle participation per node
    triangles_per_node = nx.triangles(G)   # dict: node → number of triangles
    values = np.array(list(triangles_per_node.values()))

    avg_triangle_participation = values.mean()
    variance_triangle_participation = values.var(ddof=0)

    # ---------------------------
    # 3. Topological invariants (Betti numbers & persistence)
    # ---------------------------
    try:
        simplex_tree.compute_persistence()
        bettis = simplex_tree.betti_numbers()
        betti_0 = bettis[0] if len(bettis) > 0 else np.nan
        betti_1 = bettis[1] if len(bettis) > 1 else np.nan
        betti_2 = bettis[2] if len(bettis) > 2 else np.nan
    except Exception:
        betti_0 = betti_1 = betti_2 = np.nan

    # Total persistence
    try:
        pers = simplex_tree.persistence()
        total_persistence = sum([(d - b) for (_, (b, d)) in pers if d < float("inf")])
    except Exception:
        total_persistence = np.nan

    # ---------------------------
    # 4. Geometric statistics
    # ---------------------------
    pts = np.array(points)

    # Average edge length
    edge_lengths = []
    for u, v in G.edges():
        edge_lengths.append(np.linalg.norm(pts[u] - pts[v]))

    edge_lengths = np.array(edge_lengths)
    avg_edge_length = float(np.mean(edge_lengths)) if len(edge_lengths) else np.nan

    # Local density: number of neighbors within VR radius r
    D = cdist(pts, pts)

    # "r" is the maximum edge length in the VR construction:
    # approximate by maximum edge in simplex tree filtration
    try:
        r_est = max([fv[1] for fv in simplex_tree.get_filtration()])
    except Exception:
        r_est = np.nan

    local_density = float(np.mean(np.sum(D < r_est, axis=1)))

    # ------------------------------------------
    # 5. Boundary, Hodge Laplacian, Spectral gap
    # ------------------------------------------
    edge_ids = SC.edges.filterby("order", 1)
    tri_ids  = SC.edges.filterby("order", 2)

    edges = [tuple(sorted(SC.edges.members(eid))) for eid in edge_ids]
    triangles = [tuple(sorted(SC.edges.members(tid))) for tid in tri_ids]


    edge_index = {e: i for i, e in enumerate(edges)}

    # Boundary matrix shape (#edges × #triangles)
    B2 = np.zeros((len(edges), len(triangles)))

    for j, tri in enumerate(triangles):
        for e in combinations(tri, 2):
            e = tuple(sorted(e))
            i = edge_index[e]
            B2[i, j] = 1

    L2 = B2.T @ B2     # triangle × triangle Laplacian
    eigs = np.linalg.eigvalsh(L2)

    # Identify zero and positive eigenvalues
    zero_mask = eigs <= 1e-8
    positive = eigs[~zero_mask]

    if positive.size == 0:
        L2_gap = 0.0
    else:
        L2_gap = float(positive[0])

    L2_max = float(eigs[-1])
    L2_trace = float(np.trace(L2))

    if L2_gap > 1e-8:
        L2_cond = float(L2_max / L2_gap)
    else:
        L2_cond = np.inf

    # ---------------------------
    # 6. Package into dict
    # ---------------------------
    return {
        # Basic
        "num_points": num_points,
        "num_edges": num_edges,
        "num_triangles": num_triangles,

        # Graph (1-skeleton)
        "avg_degree": avg_degree,
        "max_degree": max_degree,
        "density": density,
        "giant_fraction": giant_fraction,
        "fiedler_value": fiedler,

        # Triangle stats
        "avg_triangles_per_edge": avg_triangle_degree,
        "variance_triangles_per_edge": variance_triangle_degree,
        "avg_triangles_per_node": avg_triangle_participation,
        "variance_triangles_per_node": variance_triangle_participation,

        # Topology
        "betti_0": betti_0,
        "betti_1": betti_1,
        "betti_2": betti_2,
        "total_persistence": total_persistence,

        # Geometry
        "avg_edge_length": avg_edge_length,
        "local_density": local_density,
        "r_est": r_est,

        # Spectral gap
        "L2_spectral_gap": L2_gap,
        "L2_max_eigenvalue": L2_max,
        "L2_trace": L2_trace,
        "L2_cond": L2_cond,

    }


points = generate_random_points(n)
omega = np.random.rand(400, 1)  # random theta values in [0,1] for each oscillator.
theta0 = (
    2 * np.pi * np.random.rand(400, 1)
)  # random initial values for each oscillator.


r_values = np.linspace(0.130, 0.150, 21)

results = []
for r in r_values:
    (rips_complex, simplex_tree) = generate_rips_complex(
        points, max_simplex_order, r, verbose=False
    )
    #visualize(simplex_tree, r)

    SC = generate_simplicial_complex(simplex_tree)

    stats = extract_vr_statistics(points, simplex_tree, SC) #find the graph statistics

    sigma = 0.52
    critical_sigma = find_critical_sigma(SC, r)

    stats["r_value"] = r #append the r value
    stats["critical_sigma"] = critical_sigma #append the critical_sigma

    results.append(stats)

df = pd.DataFrame(results)
df.to_csv("experiment_results_with_stats.csv", index=False)
print("Saved results to experiment_results.csv")



#below are values we got for r = 0.130
#good: 0.0003286361914407874, 0.53
#good: 0.0015781802873912623, 0.525
#notg: 0.0027772762531045494, 0.524
#notg: 0.00747488337492376, 0.52
#notg: 0.008890396088027115, 0.51