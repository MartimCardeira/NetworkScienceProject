import random
import numpy as np
import pandas as pd
import xgi
import matplotlib.pyplot as plt
from multiprocessing import Pool

#my model but now generalised to tetrahedrons: set the amount of tetrahedrons we want in our network.
#start with one tetrahedron
#for each new tetrahedron we add, we have three cases
#the probabilities for each case follow the Bernstein polynomial, allowing one parameter p to dictate the probabilities
#of each case: https://en.wikipedia.org/wiki/Bernstein_polynomial
#with probability p^2, add 3 new nodes and connect them to an existing node
#with probability 2p(1-p), add 2 new nodes and connect them to an existing edge
#with probability (1-p)^2, add 1 new node and connect it to an existing face
#keeping adding triangles until we've reached the target number.

np.random.seed(2275)

def generate_graph(p, target_tetrahedrons, draw=False):
    SC = xgi.SimplicialComplex()
    nodes = 4
    tetrahedrons = 1
    SC.add_nodes_from([0, 1, 2, 3])
    SC.add_simplex([0, 1, 2, 3])

    while tetrahedrons < target_tetrahedrons:
        roll = np.random.random()

        if roll <= p**2:  # add 3 nodes
            target_node = np.random.randint(nodes)
            SC.add_node(nodes)
            nodes += 1
            SC.add_node(nodes)
            nodes += 1
            SC.add_node(nodes)
            nodes += 1
            SC.add_simplex([target_node, nodes - 1, nodes - 2, nodes - 3])

        elif roll <= p**2 + 2*p*(1-p):  # add 2 nodes
            candidate_edges = list(SC.edges.filterby("order", 1).members())
            target_node1, target_node2 = random.choice(candidate_edges)
            SC.add_node(nodes)
            nodes += 1
            SC.add_node(nodes)
            nodes += 1
            SC.add_simplex([target_node1, target_node2, nodes - 1, nodes - 2])

        else: # add 1 node
            candidate_triangles = list(SC.edges.filterby("order", 2).members())
            target_node1, target_node2, target_node3 = random.choice(candidate_triangles)
            SC.add_node(nodes)
            nodes += 1
            SC.add_simplex([target_node1, target_node2, target_node3, nodes - 1])

        tetrahedrons += 1

    if(draw):
        xgi.draw(SC)
        plt.show()

    return SC


def L3_stats(SC):
    L3 = xgi.hodge_laplacian(SC, order=3, orientations=None, index=False)
    eigs = np.linalg.eigvalsh(L3)

    positive = eigs[eigs > 1e-8]
    L3_gap = float(positive[0]) if len(positive) else 0.0
    L3_max = float(eigs[-1])
    L3_trace = float(np.trace(L3))
    L3_cond = float(L3_max / L3_gap) if L3_gap > 1e-8 else np.inf

    return L3_gap, L3_trace, L3_max, L3_cond


def run_kuramoto(SC, omega, theta0, sigma, T=30, n_steps=5000):
    order = 3
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
    #n = len(SC.edges.filterby("order", 3))

    # Expand upper bound until synchronized
    while True:
        r_vals = run_kuramoto(SC, omega, theta0, sigma=sigma_high)
        if is_synchronized(r_vals[-1000:]):   # last 1000 steps
            break
        sigma_high += 0.05

    # Binary search
    while sigma_high - sigma_low > tol:
        sigma_mid = 0.5 * (sigma_low + sigma_high)
        r_vals = run_kuramoto(SC, omega, theta0, sigma=sigma_mid)

        if is_synchronized(r_vals[-1000:]):
            sigma_high = sigma_mid
        else:
            sigma_low = sigma_mid

    return 0.5 * (sigma_low + sigma_high)


def run_single_experiment(args):
    """
    Callable for multiprocessing. Takes arguments as a tuple:
    (p, target_tetrahedrons, omega, theta0)
    Returns one run's results.
    """
    p, target_tetrahedrons, omega, theta0 = args

    SC = generate_graph(p, target_tetrahedrons)

    L3_gap, L3_trace, L3_max, L3_cond = L3_stats(SC)
    critical_sigma = find_critical_sigma(SC, omega, theta0)

    return L3_gap, L3_trace, L3_max, L3_cond, critical_sigma


"""
Main experiment loop
"""
if __name__ == "__main__":
    target_tetrahedrons = 50
    p_values = np.linspace(0.00, 1.00, 101)
    num_runs = 100

    omega  = np.random.rand(target_tetrahedrons, 1)
    theta0 = 2 * np.pi * np.random.rand(target_tetrahedrons, 1)

    results = []
    pool = Pool()   # use all CPU cores

    for p in p_values:
        print(f"\nRunning experiments for p = {p:.2f}")

        # Prepare arguments for all runs
        tasks = [(p, target_tetrahedrons, omega, theta0) for _ in range(num_runs)]

        # Run in parallel
        outputs = pool.map(run_single_experiment, tasks)

        # Unpack results
        L3_gaps        = [o[0] for o in outputs]
        L3_traces      = [o[1] for o in outputs]
        L3_maxes       = [o[2] for o in outputs]
        L3_conds       = [o[3] for o in outputs]
        critical_sigma = [o[4] for o in outputs]

        # Store averages
        results.append({
            "p": p,
            "mean_lambda_2": float(np.mean(L3_gaps)),
            "std_lambda_2":  float(np.std(L3_gaps)),
            "mean_trace": float(np.mean(L3_traces)),
            "mean_lambda_max": float(np.mean(L3_maxes)),
            "mean_lambda_cond": float(np.mean(L3_conds)),
            "mean_critical_sigma": float(np.mean(critical_sigma)),
            "std_critical_sigma":  float(np.std(critical_sigma)),
        })

    pool.close()
    pool.join()

    df = pd.DataFrame(results)
    df.to_csv("p_experiment_results_tetra_averaged.csv", index=False)
    print(df)