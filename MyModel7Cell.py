import random
import numpy as np
import pandas as pd
import xgi
import matplotlib.pyplot as plt
from multiprocessing import Pool

#by now the general pattern should be clear from the previous models. Using the bernstein polynomial for the probablities.

np.random.seed(2275)

def generate_graph(p, target_sevencells, draw=False):
    SC = xgi.SimplicialComplex()
    nodes = 7
    sevencells = 1
    SC.add_nodes_from([0, 1, 2, 3, 4, 5, 6])
    SC.add_simplex([0, 1, 2, 3, 4, 5, 6])

    while sevencells < target_sevencells:
        roll = np.random.random()

        if roll <= p**5:
            target_node = np.random.randint(nodes)
            SC.add_node(nodes);
            nodes += 1
            SC.add_node(nodes);
            nodes += 1
            SC.add_node(nodes);
            nodes += 1
            SC.add_node(nodes);
            nodes += 1
            SC.add_node(nodes)
            nodes += 1
            SC.add_node(nodes);
            nodes += 1
            SC.add_simplex([target_node, nodes - 1, nodes - 2, nodes - 3, nodes - 4, nodes - 5, nodes - 6])

        elif roll <= p**5 + 5*p**4*(1-p):
            candidate_edges = list(SC.edges.filterby("order", 1).members())
            u, v = random.choice(candidate_edges)
            SC.add_node(nodes);
            nodes += 1
            SC.add_node(nodes);
            nodes += 1
            SC.add_node(nodes);
            nodes += 1
            SC.add_node(nodes)
            nodes += 1
            SC.add_node(nodes);
            nodes += 1
            SC.add_simplex([u, v, nodes - 1, nodes - 2, nodes - 3, nodes - 4, nodes - 5])

        elif roll <= p**5 + 5*p**4*(1-p) + 10*p**3*(1-p)**2:
            candidate_triangles = list(SC.edges.filterby("order", 2).members())
            a, b, c = random.choice(candidate_triangles)
            SC.add_node(nodes);
            nodes += 1
            SC.add_node(nodes);
            nodes += 1
            SC.add_node(nodes)
            nodes += 1
            SC.add_node(nodes);
            nodes += 1
            SC.add_simplex([a, b, c, nodes - 1, nodes - 2, nodes - 3, nodes - 4])

        elif roll <= p**5 + 5*p**4*(1-p) + 10*p**3*(1-p)**2 + 10*p**2*(1-p)**3:
            candidate_tetrahedra = list(SC.edges.filterby("order", 3).members())
            a, b, c, d = random.choice(candidate_tetrahedra)
            SC.add_node(nodes)
            nodes += 1
            SC.add_node(nodes)
            nodes += 1
            SC.add_node(nodes);
            nodes += 1
            SC.add_simplex([a, b, c, d, nodes - 1, nodes - 2, nodes - 3])

        elif roll <= p**5 + 5*p**4*(1-p) + 10*p**3*(1-p)**2 + 10*p**2*(1-p)**3 + 5*p*(1-p)**4:
            candidate_5cells = list(SC.edges.filterby("order", 4).members())
            a, b, c, d, e = random.choice(candidate_5cells)
            SC.add_node(nodes);
            nodes += 1
            SC.add_node(nodes);
            nodes += 1
            SC.add_simplex([a, b, c, d, e, nodes - 1, nodes - 2])

        else:
            candidate_6cells = list(SC.edges.filterby("order", 5).members())
            a, b, c, d, e, f = random.choice(candidate_6cells)
            SC.add_node(nodes);
            nodes += 1
            SC.add_simplex([a, b, c, d, e, f, nodes - 1])

        sevencells += 1

    if(draw):
        xgi.draw(SC)
        plt.show()

    return SC


def L6_stats(SC):
    L6 = xgi.hodge_laplacian(SC, order=6, orientations=None, index=False)
    eigs = np.linalg.eigvalsh(L6)

    positive = eigs[eigs > 1e-8]
    L6_gap = float(positive[0]) if len(positive) else 0.0
    L6_max = float(eigs[-1])
    L6_trace = float(np.trace(L6))
    L6_cond = float(L6_max / L6_gap) if L6_gap > 1e-8 else np.inf

    return L6_gap, L6_trace, L6_max, L6_cond


def run_kuramoto(SC, omega, theta0, sigma, T=30, n_steps=5000):
    order = 6
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
    n6 = len(SC.edges.filterby("order", 6))
    assert omega.shape[0] == n6
    assert theta0.shape[0] == n6

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
    (p, target_sevencells, omega, theta0)
    Returns one run's results.
    """
    p, target_sevencells, omega, theta0 = args

    SC = generate_graph(p, target_sevencells)

    L6_gap, L6_trace, L6_max, L6_cond = L6_stats(SC)
    critical_sigma = find_critical_sigma(SC, omega, theta0)

    return L6_gap, L6_trace, L6_max, L6_cond, critical_sigma


"""
Main experiment loop
"""
if __name__ == "__main__":
    target_sevencells = 50
    p_values = np.linspace(0.00, 1.00, 101)
    num_runs = 100

    omega  = np.random.rand(target_sevencells, 1)
    theta0 = 2 * np.pi * np.random.rand(target_sevencells, 1)

    results = []
    pool = Pool()   # use all CPU cores

    for p in p_values:
        print(f"\nRunning experiments for p = {p:.2f}")

        # Prepare arguments for all runs
        tasks = [(p, target_sevencells, omega, theta0) for _ in range(num_runs)]

        # Run in parallel
        outputs = pool.map(run_single_experiment, tasks)

        # Unpack results
        L6_gaps        = [o[0] for o in outputs]
        L6_traces      = [o[1] for o in outputs]
        L6_maxes       = [o[2] for o in outputs]
        L6_conds       = [o[3] for o in outputs]
        critical_sigma = [o[4] for o in outputs]

        # Store averages
        results.append({
            "p": p,
            "mean_lambda_2": float(np.mean(L6_gaps)),
            "std_lambda_2":  float(np.std(L6_gaps)),
            "mean_trace": float(np.mean(L6_traces)),
            "mean_lambda_max": float(np.mean(L6_maxes)),
            "mean_lambda_cond": float(np.mean(L6_conds)),
            "mean_critical_sigma": float(np.mean(critical_sigma)),
            "std_critical_sigma":  float(np.std(critical_sigma)),
        })

    pool.close()
    pool.join()

    df = pd.DataFrame(results)
    df.to_csv("7cell_experiment.csv", index=False)
    print(df)