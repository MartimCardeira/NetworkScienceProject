import pandas as pd
import matplotlib.pyplot as plt

# Load your CSVs (rename these if your filenames differ)
df_tri = pd.read_csv("triangle.csv")
df_tetra = pd.read_csv("tetrahedron.csv")
df_5 = pd.read_csv("5cell.csv")
df_6 = pd.read_csv("6cell.csv")

plt.figure(figsize=(10,6))

# Plot all curves
plt.plot(df_tri["p"],   df_tri["sigma"],   label="Triangles (order=2)",  color="blue")
plt.plot(df_tetra["p"], df_tetra["sigma"], label="Tetrahedra (order=3)", color="green")
plt.plot(df_5["p"],     df_5["sigma"],     label="5-cells (order=4)",    color="orange")
plt.plot(df_6["p"],     df_6["sigma"],     label="6-cells (order=5)",    color="red")

# Make it pretty
plt.xlabel("p")
plt.ylabel("Mean Critical σ")
plt.title("Comparison of Critical Sigma across Generalized Growth Models")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.show()


#second plot
plt.figure(figsize=(10,6))

# Plot all curves
plt.plot(df_tri["spectral_gap"],   df_tri["sigma"],   label="Triangles (order=2)",  color="blue")
plt.plot(df_tetra["spectral_gap"], df_tetra["sigma"], label="Tetrahedra (order=3)", color="green")
plt.plot(df_5["spectral_gap"],     df_5["sigma"],     label="5-cells (order=4)",    color="orange")
plt.plot(df_6["spectral_gap"],     df_6["sigma"],     label="6-cells (order=5)",    color="red")

# Make it pretty
plt.xlabel("L_n spectral gap")
plt.ylabel("Mean Critical σ")
plt.title("Comparison of the n-order Laplacian Gap across Generalized Growth Models")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.show()
