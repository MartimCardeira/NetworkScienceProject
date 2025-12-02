import random
import numpy as np
import pandas as pd
import xgi
import matplotlib.pyplot as plt
from multiprocessing import Pool


def generate_graph(p, target_triangles, draw=False):
    SC = xgi.SimplicialComplex()
    nodes = 3
    triangles = 1
    SC.add_nodes_from([0, 1, 2])
    SC.add_simplex([0, 1, 2])

    while triangles < target_triangles:
        roll = np.random.random()

        if roll <= p:  # add 2 nodes
            target_node = np.random.randint(nodes)
            SC.add_node(nodes)
            nodes += 1
            SC.add_node(nodes)
            nodes += 1
            SC.add_simplex([target_node, nodes - 1, nodes - 2])

        else:  # add 1 node
            candidate_edges = SC.edges.filterby("order", 1).members()
            target_node1, target_node2 = random.choice(candidate_edges)
            SC.add_node(nodes)
            nodes += 1
            SC.add_simplex([target_node1, target_node2, nodes - 1])

        triangles += 1

    if(draw):
        xgi.draw(SC)
        plt.show()

    return SC


def L2_stats(SC):
    L2 = xgi.hodge_laplacian(SC, order=2, orientations=None, index=False)
    eigs = np.linalg.eigvalsh(L2)

    positive = eigs[eigs > 1e-8]
    L2_gap = float(positive[0]) if len(positive) else 0.0
    L2_max = float(eigs[-1])
    L2_trace = float(np.trace(L2))
    L2_cond = float(L2_max / L2_gap) if L2_gap > 1e-8 else np.inf

    return L2_gap, L2_trace, L2_max, L2_cond


def run_kuramoto(SC, omega, theta0, sigma, T=30, n_steps=5000):
    order = 2
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
    return r


def is_synchronized(values, r_threshold=0.80, std_threshold=0.002):
    return (np.std(values) < std_threshold) and (np.mean(values) > r_threshold)


def find_critical_sigma(SC, omega, theta0, sigma_low=0.2, sigma_high=1.0, tol=0.001):
    n = len(SC.edges.filterby("order", 2))

    # Expand upper bound until synchronized
    while True:
        r_vals = run_kuramoto(SC, omega[:n], theta0[:n], sigma=sigma_high)
        if is_synchronized(r_vals[-1000:]):   # last 1000 steps
            break
        sigma_high += 0.05

    # Binary search
    while sigma_high - sigma_low > tol:
        sigma_mid = 0.5 * (sigma_low + sigma_high)
        r_vals = run_kuramoto(SC, omega[:n], theta0[:n], sigma=sigma_mid)

        if is_synchronized(r_vals[-1000:]):
            sigma_high = sigma_mid
        else:
            sigma_low = sigma_mid

    return 0.5 * (sigma_low + sigma_high)


def run_single_experiment(args):
    """
    Callable for multiprocessing. Takes arguments as a tuple:
    (p, target_triangles, omega, theta0)
    Returns one run's results.
    """
    p, target_triangles, omega, theta0 = args

    SC = generate_graph(p, target_triangles)

    L2_gap, L2_trace, L2_max, L2_cond = L2_stats(SC)
    critical_sigma = find_critical_sigma(SC, omega, theta0)

    return L2_gap, L2_trace, L2_max, L2_cond, critical_sigma


"""
Main experiment loop
"""

target_triangles = 50
p_values = np.linspace(0, 1, 11)
num_runs = 50

omega  = np.random.rand(target_triangles, 1)
theta0 = 2 * np.pi * np.random.rand(target_triangles, 1)

results = []
pool = Pool()   # use all CPU cores

for p in p_values:
    print(f"\nRunning experiments for p = {p:.2f}")

    # Prepare arguments for all runs
    tasks = [(p, target_triangles, omega, theta0) for _ in range(num_runs)]

    # Run in parallel
    outputs = pool.map(run_single_experiment, tasks)

    # Unpack results
    L2_gaps        = [o[0] for o in outputs]
    L2_traces      = [o[1] for o in outputs]
    L2_maxes       = [o[2] for o in outputs]
    L2_conds       = [o[3] for o in outputs]
    critical_sigma = [o[4] for o in outputs]

    # Store averages
    results.append({
        "p": p,
        "mean_L2_gap": float(np.mean(L2_gaps)),
        "std_L2_gap":  float(np.std(L2_gaps)),
        "mean_L2_trace": float(np.mean(L2_traces)),
        "mean_L2_max": float(np.mean(L2_maxes)),
        "mean_L2_cond": float(np.mean(L2_conds)),
        "mean_critical_sigma": float(np.mean(critical_sigma)),
        "std_critical_sigma":  float(np.std(critical_sigma)),
    })

pool.close()
pool.join()

df = pd.DataFrame(results)
df.to_csv("p_experiment_results_averaged.csv", index=False)
print(df)