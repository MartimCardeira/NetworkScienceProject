import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_vips  = pd.read_csv("vips_experiment.csv")
df_tri   = pd.read_csv("triangle_experiment.csv")
df_tetra = pd.read_csv("tetrahedron_experiment.csv")
df_5     = pd.read_csv("5cell_experiment.csv")
df_6     = pd.read_csv("6cell_experiment.csv")
df_7     = pd.read_csv("7cell_experiment.csv")

datasets = [
    (df_tri,   "Triangles (order=2)",     "blue"),
    (df_tetra, "Tetrahedra (order=3)",    "green"),
    (df_5,     "5-cells (order=4)",       "orange"),
    (df_6,     "6-cells (order=5)",       "red"),
    (df_7,     "7-cells (order=6)",       "purple"),
]

#EXPERIMENT 1 PLOTS

# 1. Critical sigma vs. r value

plt.figure(figsize=(7, 5))

plt.scatter(
    df_vips["r_value"],
    df_vips["critical_sigma"],
    s=60,
    alpha=0.8
)

plt.xlabel("r")
plt.ylabel("Critical σ")
plt.title("Critical Sigma vs R value")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 2. Number of Triangles vs. R

x = df_vips["r_value"].values
y = df_vips["num_triangles"].values
m, b = np.polyfit(x, y, 1)
y_fit = m * x + b

# R^2
r2 = 1 - np.sum((y - y_fit)**2) / np.sum((y - y.mean())**2)

plt.figure(figsize=(7, 5))
plt.scatter(
    x,
    y,
    s=60,
    alpha=0.8
)

idx = np.argsort(x)
plt.plot(
    x[idx],
    y_fit[idx],
    linestyle="--",
    linewidth=2
)
plt.xlabel("r")
plt.ylabel("Number of Triangles")
plt.title("Number of Triangles vs R value")
plt.grid(True, alpha=0.3)

plt.text(
    0.05, 0.95,
    f"$R^2 = {r2:.3f}$",
    transform=plt.gca().transAxes,
    fontsize=12,
    verticalalignment="top"
)

plt.tight_layout()
plt.show()

# 3. Critical sigma vs. Average Node Degree

plt.figure(figsize=(7, 5))

plt.scatter(
    df_vips["avg_degree"],
    df_vips["critical_sigma"],
    s=60,
    alpha=0.8
)

plt.xlabel("Average Node Degree")
plt.ylabel("Critical σ")
plt.title("Critical Sigma vs Average Node Degree")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 4. Critical sigma vs. Number of Triangles

plt.figure(figsize=(7, 5))

plt.scatter(
    df_vips["num_triangles"],
    df_vips["critical_sigma"],
    s=60,
    alpha=0.8
)

plt.xlabel("Number of Triangles")
plt.ylabel("Critical σ")
plt.title("Critical Sigma vs Number of Triangles")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 5. Critical sigma vs. lambda_2

plt.figure(figsize=(7, 5))

plt.scatter(
    df_vips["L2_spectral_gap"],
    df_vips["critical_sigma"],
    s=60,
    alpha=0.8
)

plt.xlabel("λ₂")
plt.ylabel("Critical σ")
plt.title("Critical Sigma vs Hodge Laplacian λ₂")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 6. Critical sigma vs. 1st Betti number

plt.figure(figsize=(7, 5))

plt.scatter(
    df_vips["betti_1"],
    df_vips["critical_sigma"],
    s=60,
    alpha=0.8
)

plt.xlabel("β₁")
plt.ylabel("Critical σ")
plt.title("Critical Sigma vs 1st Betti Number β₁")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

#EXPERIMENT 2 PLOTS

# 1. lambda_2 vs. p

plt.figure(figsize=(10, 6))

for df, label, color in datasets:
    plt.plot(
        df["p"],
        df["mean_lambda_2"],
        label=label,
        color=color,
        #linewidth=2
    )
    plt.fill_between(
        df["p"],
        df["mean_lambda_2"] - df["std_lambda_2"],
        df["mean_lambda_2"] + df["std_lambda_2"],
        color=color,
        alpha=0.1,
        linewidth=0,
        zorder=1
    )

plt.xlabel("p")
plt.ylabel("λ₂")
plt.title("Hodge Laplacian λ₂ vs. p")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# 2. Critical sigma vs. lambda_2

plt.figure(figsize=(10, 6))

for index, (df, label, color) in enumerate(datasets):

    plt.plot(
        df["mean_lambda_2"],
        df["mean_critical_sigma"],
        label=f"{label} λ₂",
        color=color,
        #linewidth=2
    )
    plt.fill_between(
        df["mean_lambda_2"],
        df["mean_critical_sigma"] - df["std_critical_sigma"],
        df["mean_critical_sigma"] + df["std_critical_sigma"],
        color=color,
        alpha=0.1,
        linewidth=0,
        zorder=1
    )
    """
    plt.axhline(
        y=1.0/float(index+3),
        color=color,
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label=f"σ = {1.0/float(index+3):.3f}"
    )
    """

plt.xlabel("Mean λ₂")
plt.ylabel("Mean Critical σ")
plt.title("Critical Sigma vs. Hodge Laplacian λ₂")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# 3. Critical sigma vs. lambda_max

plt.figure(figsize=(10, 6))

for df, label, color in datasets:

    plt.plot(
        df["mean_lambda_max"],
        df["mean_critical_sigma"],
        label=f"{label} λₘₐₓ",
        color=color,
        #linewidth=2
    )
    plt.fill_between(
        df["mean_lambda_max"],
        df["mean_critical_sigma"] - df["std_critical_sigma"],
        df["mean_critical_sigma"] + df["std_critical_sigma"],
        color=color,
        alpha=0.1,
        linewidth=0,
        zorder=1
    )

plt.xlabel("Mean λₘₐₓ")
plt.ylabel("Mean Critical σ")
plt.title("Critical Sigma vs. Hodge Laplacian λₘₐₓ")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# 4. Critical sigma vs. lambda_cond

plt.figure(figsize=(10, 6))

for df, label, color in datasets:

    plt.plot(
        df["mean_lambda_cond"],
        df["mean_critical_sigma"],
        label=f"{label} λₘₐₓ/λ₂",
        color=color,
        #linewidth=2
    )
    plt.fill_between(
        df["mean_lambda_cond"],
        df["mean_critical_sigma"] - df["std_critical_sigma"],
        df["mean_critical_sigma"] + df["std_critical_sigma"],
        color=color,
        alpha=0.1,
        linewidth=0,
        zorder=1
    )

plt.xlabel("Mean λₘₐₓ/λ₂")
plt.ylabel("Mean Critical σ")
plt.title("Critical Sigma vs. Hodge Laplacian λₘₐₓ/λ₂")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

#Bonus, lambda2 vs crit but without error bars and with the boundaries
# 2. Critical sigma vs. lambda_2

plt.figure(figsize=(10, 6))

for index, (df, label, color) in enumerate(datasets):

    plt.plot(
        df["mean_lambda_2"],
        df["mean_critical_sigma"],
        label=f"{label} λ₂",
        color=color,
        #linewidth=2
    )
    plt.axhline(
        y=1.0/float(index+3),
        color=color,
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label=f"σ = {1.0/float(index+3):.3f}"
    )
plt.xlabel("Mean λ₂")
plt.ylabel("Mean Critical σ")
plt.title("Critical Sigma vs. Hodge Laplacian λ₂")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()