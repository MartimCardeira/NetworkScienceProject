import pandas as pd
import matplotlib.pyplot as plt

df_tri   = pd.read_csv("triangle.csv")
df_tetra = pd.read_csv("tetrahedron.csv")
df_5     = pd.read_csv("5cell.csv")
df_6     = pd.read_csv("6cell.csv")
df_7     = pd.read_csv("7cell.csv")     # <-- NEW

datasets = [
    (df_tri,   "Triangles (order=2)",     "blue"),
    (df_tetra, "Tetrahedra (order=3)",    "green"),
    (df_5,     "5-cells (order=4)",       "orange"),
    (df_6,     "6-cells (order=5)",       "red"),
    (df_7,     "7-cells (order=6)",       "purple"),
]
#EXPERIMENT 1 PLOTS



#EXPERIMENT 2 PLOTS

# 1. Critical sigma vs. p

plt.figure(figsize=(10, 6))

for df, label, color in datasets:
    plt.plot(
        df["p"],
        df["sigma"],   # <-- correct column name
        label=label,
        color=color,
        linewidth=2
    )

plt.xlabel("p")
plt.ylabel("Mean Critical σ")
plt.title("Critical Sigma vs. p for Generalized Growth Models")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# 2. Critical sigma vs. spectral gap

plt.figure(figsize=(10, 6))

for df, label, color in datasets:

    plt.plot(
        df["spectral_gap"],
        df["sigma"],
        label=f"{label} (spectral_gap)",
        color=color,
        linewidth=2
    )

plt.xlabel("L_n Spectral Gap")
plt.ylabel("Mean Critical σ")
plt.title("Critical Sigma vs. First n-Order Laplacian Eigenvalue")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# 3. Critical sigma vs. Max eigenvalue

plt.figure(figsize=(10, 6))

for df, label, color in datasets:

    plt.plot(
        df["eig_max"],
        df["sigma"],
        label=f"{label} (eig_max)",
        color=color,
        linewidth=2
    )

plt.xlabel("L_n Largest Eigenvalue")
plt.ylabel("Mean Critical σ")
plt.title("Critical Sigma vs. Largest n-Order Laplacian Eigenvalue")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()