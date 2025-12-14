import random
import numpy as np
import pandas as pd
import xgi
import matplotlib.pyplot as plt
from multiprocessing import Pool

#my model but now generalised to 5-cells: set the amount of 5-cells we want in our network.
#start with one 5-cell
#for each new 5-cell we add, we have four cases
#the probabilities for each case follow the Bernstein polynomial, allowing one parameter p to dictate the probabilities
#of each case: https://en.wikipedia.org/wiki/Bernstein_polynomial
#with probability p^3, add 4 new nodes and connect them to an existing node
#with probability 3p^2(1-p), add 3 new nodes and connect them to an existing edge
#with probability 3p(1-p)^2, add 2 new nodes and connect them to an existing triangle
#with probability (1-p)^3, add 1 new node and connect it to an existing tetrahedron
#keeping adding triangles until we've reached the target number.

np.random.seed(2275)

def generate_graph(p, target_fivecells, draw=False):
    SC = xgi.SimplicialComplex()
    nodes = 5
    fivecells = 1
    SC.add_nodes_from([0, 1, 2, 3, 4])
    SC.add_simplex([0, 1, 2, 3, 4])

    while fivecells < target_fivecells:
        roll = np.random.random()

        if roll <= p**3:  # add 4 nodes
            target_node = np.random.randint(nodes)
            SC.add_node(nodes)
            nodes += 1
            SC.add_node(nodes)
            nodes += 1
            SC.add_node(nodes)
            nodes += 1
            SC.add_node(nodes)
            nodes += 1
            SC.add_simplex([target_node, nodes - 1, nodes - 2, nodes - 3, nodes - 4])

        elif roll <= p**3 + 3*(p**2)*(1-p):  # add 3 nodes
            candidate_edges = list(SC.edges.filterby("order", 1).members())
            target_node1, target_node2 = random.choice(candidate_edges)
            SC.add_node(nodes)
            nodes += 1
            SC.add_node(nodes)
            nodes += 1
            SC.add_node(nodes)
            nodes += 1
            SC.add_simplex([target_node1, target_node2, nodes - 1, nodes - 2, nodes - 3])

        elif roll <= p**3 + 3*(p**2)*(1-p) + 3*p*(1-p)**2: # add 2 nodes
            candidate_triangles = list(SC.edges.filterby("order", 2).members())
            target_node1, target_node2, target_node3 = random.choice(candidate_triangles)
            SC.add_node(nodes)
            nodes += 1
            SC.add_node(nodes)
            nodes += 1
            SC.add_simplex([target_node1, target_node2, target_node3, nodes - 1, nodes - 2])

        else:  # add 1 node
            candidate_tetrahedrons = list(SC.edges.filterby("order", 3).members())
            target_node1, target_node2, target_node3, target_node4 = random.choice(candidate_tetrahedrons)
            SC.add_node(nodes)
            nodes += 1
            SC.add_simplex([target_node1, target_node2, target_node3, target_node4, nodes - 1])

        fivecells += 1

    if(draw):
        xgi.draw(SC)
        plt.show()

    return SC


def L4_stats(SC):
    L4 = xgi.hodge_laplacian(SC, order=4, orientations=None, index=False)
    eigs = np.linalg.eigvalsh(L4)

    positive = eigs[eigs > 1e-8]
    L4_gap = float(positive[0]) if len(positive) else 0.0
    L4_max = float(eigs[-1])
    L4_trace = float(np.trace(L4))
    L4_cond = float(L4_max / L4_gap) if L4_gap > 1e-8 else np.inf

    return L4_gap, L4_trace, L4_max, L4_cond


def run_kuramoto(SC, omega, theta0, sigma, T=30, n_steps=5000):
    order = 4
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


def find_critical_sigma(SC, omega, theta0, sigma_low=0.0, sigma_high=0.3, tol=0.001):
    #just a small sanity check
    n4 = len(SC.edges.filterby("order", 4))
    assert omega.shape[0] == n4
    assert theta0.shape[0] == n4

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
    (p, target_fivecells, omega, theta0)
    Returns one run's results.
    """
    p, target_fivecells, omega, theta0 = args

    SC = generate_graph(p, target_fivecells)

    L4_gap, L4_trace, L4_max, L4_cond = L4_stats(SC)
    critical_sigma = find_critical_sigma(SC, omega, theta0)

    return L4_gap, L4_trace, L4_max, L4_cond, critical_sigma


"""
Main experiment loop
"""
if __name__ == "__main__":
    target_fivecells = 50
    p_values = np.linspace(0.00, 1.00, 101)
    num_runs = 100

    omega  = np.random.rand(target_fivecells, 1)
    theta0 = 2 * np.pi * np.random.rand(target_fivecells, 1)

    results = []
    pool = Pool()   # use all CPU cores

    for p in p_values:
        print(f"\nRunning experiments for p = {p:.2f}")

        # Prepare arguments for all runs
        tasks = [(p, target_fivecells, omega, theta0) for _ in range(num_runs)]

        # Run in parallel
        outputs = pool.map(run_single_experiment, tasks)

        # Unpack results
        L4_gaps        = [o[0] for o in outputs]
        L4_traces      = [o[1] for o in outputs]
        L4_maxes       = [o[2] for o in outputs]
        L4_conds       = [o[3] for o in outputs]
        critical_sigma = [o[4] for o in outputs]

        # Store averages
        results.append({
            "p": p,
            "mean_lambda_2": float(np.mean(L4_gaps)),
            "std_lambda_2":  float(np.std(L4_gaps)),
            "mean_trace": float(np.mean(L4_traces)),
            "mean_lambda_max": float(np.mean(L4_maxes)),
            "mean_lambda_cond": float(np.mean(L4_conds)),
            "mean_critical_sigma": float(np.mean(critical_sigma)),
            "std_critical_sigma":  float(np.std(critical_sigma)),
        })

    pool.close()
    pool.join()

    df = pd.DataFrame(results)
    df.to_csv("5cell_experiment.csv", index=False)
    print(df)