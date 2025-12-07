import random
import numpy as np
import pandas as pd
import xgi
import matplotlib.pyplot as plt
from multiprocessing import Pool

#my model but now generalised to 6-cells: set the amount of 6-cells we want in our network.
#start with one 6-cell
#for each new 6-cell we add, we have four cases
#the probabilities for each case follow the Bernstein polynomial, allowing one parameter p to dictate the probabilities
#of each case: https://en.wikipedia.org/wiki/Bernstein_polynomial
#with probability p^4, add 5 new nodes
#with probability 4p^3(1-p), add 4 new nodes
#with probability 6p^2(1-p)^2, add 3 new nodes
#with probability 4p(1-p)^3, add 2 new nodes
#with probability (1-p)^4, add 1 new node
#keeping adding 6-cells until we've reached the target number.

np.random.seed(2275)

def generate_graph(p, target_sixcells, draw=False):
    SC = xgi.SimplicialComplex()
    nodes = 6
    sixcells = 1
    SC.add_nodes_from([0, 1, 2, 3, 4, 5])
    SC.add_simplex([0, 1, 2, 3, 4, 5])

    while sixcells < target_sixcells:
        roll = np.random.random()

        if roll <= p**4:
            target_node = np.random.randint(nodes)
            SC.add_node(nodes);
            nodes += 1
            SC.add_node(nodes);
            nodes += 1
            SC.add_node(nodes);
            nodes += 1
            SC.add_node(nodes);
            nodes += 1
            SC.add_node(nodes);
            nodes += 1
            SC.add_simplex([target_node, nodes - 1, nodes - 2, nodes - 3, nodes - 4, nodes - 5])

        elif roll <= p**4 + 4*p**3*(1-p):
            candidate_edges = list(SC.edges.filterby("order", 1).members())
            u, v = random.choice(candidate_edges)
            SC.add_node(nodes);
            nodes += 1
            SC.add_node(nodes);
            nodes += 1
            SC.add_node(nodes);
            nodes += 1
            SC.add_node(nodes);
            nodes += 1
            SC.add_simplex([u, v, nodes - 1, nodes - 2, nodes - 3, nodes - 4])

        elif roll <= p**4 + 4*p**3*(1-p) + 6*p**2*(1-p)**2:
            # CASE 2: attach to a triangle, add 3 new nodes
            candidate_triangles = list(SC.edges.filterby("order", 2).members())
            a, b, c = random.choice(candidate_triangles)
            SC.add_node(nodes);
            nodes += 1
            SC.add_node(nodes);
            nodes += 1
            SC.add_node(nodes);
            nodes += 1
            SC.add_simplex([a, b, c, nodes - 1, nodes - 2, nodes - 3])

        elif roll <= p**4 + 4*p**3*(1-p) + 6*p**2*(1-p)**2 + 4*p*(1-p)**3:
            candidate_tetrahedra = list(SC.edges.filterby("order", 3).members())
            a, b, c, d = random.choice(candidate_tetrahedra)
            SC.add_node(nodes);
            nodes += 1
            SC.add_node(nodes);
            nodes += 1
            SC.add_simplex([a, b, c, d, nodes - 1, nodes - 2])

        else:
            candidate_4faces = list(SC.edges.filterby("order", 4).members())
            a, b, c, d, e = random.choice(candidate_4faces)
            SC.add_node(nodes);
            nodes += 1
            SC.add_simplex([a, b, c, d, e, nodes - 1])

        sixcells += 1

    if(draw):
        xgi.draw(SC)
        plt.show()

    return SC


def L5_stats(SC):
    L5 = xgi.hodge_laplacian(SC, order=5, orientations=None, index=False)
    eigs = np.linalg.eigvalsh(L5)

    positive = eigs[eigs > 1e-8]
    L5_gap = float(positive[0]) if len(positive) else 0.0
    L5_max = float(eigs[-1])
    L5_trace = float(np.trace(L5))
    L5_cond = float(L5_max / L5_gap) if L5_gap > 1e-8 else np.inf

    return L5_gap, L5_trace, L5_max, L5_cond


def run_kuramoto(SC, omega, theta0, sigma, T=30, n_steps=5000):
    order = 5
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


def find_critical_sigma(SC, omega, theta0, sigma_low=0.0, sigma_high=0.25, tol=0.001):
    #just a small sanity check
    n4 = len(SC.edges.filterby("order", 5))
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
    (p, target_sixcells, omega, theta0)
    Returns one run's results.
    """
    p, target_sixcells, omega, theta0 = args

    SC = generate_graph(p, target_sixcells)

    L5_gap, L5_trace, L5_max, L5_cond = L5_stats(SC)
    critical_sigma = find_critical_sigma(SC, omega, theta0)

    return L5_gap, L5_trace, L5_max, L5_cond, critical_sigma


"""
Main experiment loop
"""
if __name__ == "__main__":
    target_sixcells = 50
    p_values = np.linspace(0.00, 1.00, 101)
    num_runs = 100

    omega  = np.random.rand(target_sixcells, 1)
    theta0 = 2 * np.pi * np.random.rand(target_sixcells, 1)

    results = []
    pool = Pool()   # use all CPU cores

    for p in p_values:
        print(f"\nRunning experiments for p = {p:.2f}")

        # Prepare arguments for all runs
        tasks = [(p, target_sixcells, omega, theta0) for _ in range(num_runs)]

        # Run in parallel
        outputs = pool.map(run_single_experiment, tasks)

        # Unpack results
        L5_gaps        = [o[0] for o in outputs]
        L5_traces      = [o[1] for o in outputs]
        L5_maxes       = [o[2] for o in outputs]
        L5_conds       = [o[3] for o in outputs]
        critical_sigma = [o[4] for o in outputs]

        # Store averages
        results.append({
            "p": p,
            "mean_L5_gap": float(np.mean(L5_gaps)),
            "std_L5_gap":  float(np.std(L5_gaps)),
            "mean_L5_trace": float(np.mean(L5_traces)),
            "mean_L5_max": float(np.mean(L5_maxes)),
            "mean_L5_cond": float(np.mean(L5_conds)),
            "mean_critical_sigma": float(np.mean(critical_sigma)),
            "std_critical_sigma":  float(np.std(critical_sigma)),
        })

    pool.close()
    pool.join()

    df = pd.DataFrame(results)
    df.to_csv("p_experiment_results_sixcell_averaged.csv", index=False)
    print(df)