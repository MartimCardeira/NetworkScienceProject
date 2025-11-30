import xgi
import matplotlib.pyplot as plt
import numpy as np
from xgi import SimplicialComplex

#This line samples a random simplicial complex of 20 nodes with edge and triangle probabilities of 0.05 and 0.005 respectively.
#S = xgi.random_simplicial_complex(30, [0.05,0.005], seed=None)
#xgi.write_hif(S, "simplex_json.json")
S = xgi.read_hif("simplex_json.json")


#Visualize it real quick
xgi.draw(S)
plt.show()

#To see the simplices this graph contains...
print(S.edges.members())

#Keep it simple, orientation for all simplices is 0, we only care about varying topology
#This applies the orientation to all edges, and all higher-order simplices have edges in them, so yes it applies to all orders simplices.
orientations = {idd: 0 for idd in list(S.edges.filterby("order", 1, mode="geq"))}

#Specifiy the parameters of our Kuramoto model.
order = 2 #The order of the simplices we are applying oscillators to.
n = len(S.edges.filterby("order", order))  #Number of oscillating simplices

omega = np.random.rand(n, 1) #random theta values in [0,1] for each oscillator.
theta0 = 2 * np.pi * np.random.rand(n, 1) #random initial values for each oscillator.

sigma = 0.4 #Coupling strength. Unfortunately, XGI's model is simplified and doesn't allow different coupling strengths for different orders.
T = 30 #How long we run the simulation for (in seconds).
n_steps = 5000 #We are approximating the dynamics discretely. More steps means more accurate results, but will take more time.

for x in range(21):
    sigma = x/20.0

    #Run the simulation
    (
        theta,
        theta_minus,
        theta_plus,
        om1_dict,
        o_dict,
        op1_dict,
    ) = xgi.synchronization.simulate_simplicial_kuramoto(
        S, orientations, order, omega, sigma, theta0, T, n_steps, True
    )

    #Find out how synchronized we are after T seconds.
    r = xgi.synchronization.compute_simplicial_order_parameter(theta_minus, theta_plus)

    fig, axs = plt.subplots(2, 1)
    fig.set_figheight(7)
    fig.set_figwidth(8)

    labels_list = [
        "[%s]" % ", ".join(map(str, list(S.edges.members()[idx])))
        for idx in list(o_dict.values())
    ]


    axs[0].plot(np.linspace(0, T, n_steps), np.sin(np.transpose(theta)))
    axs[0].set_title(f"Phases evolution dynamics, Sigma value {sigma}")
    axs[0].legend(labels_list)

    axs[1].plot(np.linspace(0, T, n_steps), r)
    axs[1].set_title("Simplicial order parameter")
    axs[1].set_ylim((0, 1))

    plt.show()

print(r[4999])
