import pandas as pd
import matplotlib.pyplot as plt

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



#EXPERIMENT 2 PLOTS

# 1. lambda_2 vs. p

plt.figure(figsize=(10, 6))

for df, label, color in datasets:
    plt.plot(
        df["p"],
        df["mean_lambda_2"],   # <-- correct column name
        label=label,
        color=color,
        linewidth=2
    )

plt.xlabel("p")
plt.ylabel("Mean Critical σ")
plt.title("Hodge Laplacian λ₂ vs. p")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# 2. Critical sigma vs. lambda_2

plt.figure(figsize=(10, 6))

for df, label, color in datasets:

    plt.plot(
        df["mean_lambda_2"],
        df["mean_critical_sigma"],
        label=f"{label} λ₂",
        color=color,
        linewidth=2
    )

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
        linewidth=2
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
        linewidth=2
    )

plt.xlabel("Mean λₘₐₓ/λ₂")
plt.ylabel("Mean Critical σ")
plt.title("Critical Sigma vs. Hodge Laplacian λₘₐₓ/λ₂")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()