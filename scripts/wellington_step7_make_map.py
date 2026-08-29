import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

DATA_DIR = "../data"
OUT_DIR = "../output"


def load_tif(path):
    with rasterio.open(path) as src:
        return src.read(1)


landcover_2012 = load_tif(f"{DATA_DIR}/wellington_landcover_2012.tif")
landcover_2023 = load_tif(f"{DATA_DIR}/wellington_landcover_2023.tif")
landcover_2035 = np.load(f"{OUT_DIR}/simulated_landcover_2035.npy")

built_2012 = landcover_2012 == 3
built_2023 = landcover_2023 == 3
built_2035 = landcover_2035 == 3

growth_map = np.zeros(landcover_2012.shape, dtype="uint8")
growth_map[built_2012] = 1
growth_map[built_2023 & ~built_2012] = 2
growth_map[built_2035 & ~built_2023] = 3

COLORS = ["#e8e8e0", "#7a1f1f", "#e67e22", "#f1c40f"]
LABELS = ["Vegetation / Agriculture / Water", "Built-up as of 2012 (baseline)",
          "New growth 2012\u21922023 (real)", "Projected growth 2023\u21922035 (simulated)"]
CMAP = ListedColormap(COLORS)

ever_built = built_2012 | built_2023 | built_2035
rows = np.any(ever_built, axis=1)
cols = np.any(ever_built, axis=0)
rmin, rmax = np.where(rows)[0][[0, -1]]
cmin, cmax = np.where(cols)[0][[0, -1]]
BUFFER = 40
rmin = max(0, rmin - BUFFER)
rmax = min(growth_map.shape[0], rmax + BUFFER)
cmin = max(0, cmin - BUFFER)
cmax = min(growth_map.shape[1], cmax + BUFFER)
cropped = growth_map[rmin:rmax, cmin:cmax]

fig, ax = plt.subplots(figsize=(13, 11))
ax.imshow(cropped, cmap=CMAP, vmin=0, vmax=3, interpolation="nearest")
ax.set_xticks([])
ax.set_yticks([])
ax.set_title(
    "Wellington Region: Urban Growth 2012 \u2192 2023 (real) \u2192 2035 (simulated)",
    fontsize=15, fontweight="bold", pad=15,
)

legend_elements = [Patch(facecolor=COLORS[i], label=LABELS[i]) for i in range(4)]
ax.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, -0.02),
          ncol=1, fontsize=10, frameon=False)

fig.text(0.5, 0.01, "Data: LINZ Land Cover Database (LCDB v6.0) | Roads: LINZ Topo50",
          ha="center", fontsize=9, color="gray")

plt.tight_layout(rect=[0, 0.08, 1, 1])
out_path = f"{OUT_DIR}/wellington_growth_map.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight")
print(f"Saved: {out_path}")

new_2012_2023 = int((built_2023 & ~built_2012).sum())
new_2023_2035 = int((built_2035 & ~built_2023).sum())
print(f"New built pixels 2012->2023 (real): {new_2012_2023} (~{new_2012_2023*0.09:.1f} km2)")
print(f"New built pixels 2023->2035 (simulated): {new_2023_2035} (~{new_2023_2035*0.09:.1f} km2)")
