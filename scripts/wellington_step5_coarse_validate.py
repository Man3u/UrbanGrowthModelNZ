import numpy as np
import rasterio
from scipy.stats import pearsonr

DATA_DIR = "../data"
OUT_DIR = "../output"
BLOCK = 10


def load(path):
    with rasterio.open(path) as src:
        return src.read(1)


landcover_2018 = load(f"{DATA_DIR}/wellington_landcover_2018.tif")
landcover_2023_real = load(f"{DATA_DIR}/wellington_landcover_2023.tif")
simulated_2023 = np.load(f"{OUT_DIR}/simulated_landcover_2023.npy")

real_new_built = (landcover_2023_real == 3) & (landcover_2018 != 3)
sim_new_built = (simulated_2023 == 3) & (landcover_2018 != 3)


def block_sum(arr, block):
    h, w = arr.shape
    h_trim = h - (h % block)
    w_trim = w - (w % block)
    arr = arr[:h_trim, :w_trim]
    return arr.reshape(h_trim // block, block, w_trim // block, block).sum(axis=(1, 3))


real_density = block_sum(real_new_built.astype(int), BLOCK)
sim_density = block_sum(sim_new_built.astype(int), BLOCK)

corr, pval = pearsonr(real_density.ravel(), sim_density.ravel())
print(f"{BLOCK*30}m neighborhood cells: {real_density.size} total")
print(f"Pearson correlation (real vs simulated growth density per cell): {corr:.4f} (p={pval:.2e})")

real_has_growth = real_density > 0
sim_has_growth = sim_density > 0
cells_with_real_growth = real_has_growth.sum()
correctly_flagged = (real_has_growth & sim_has_growth).sum()
print(f"\nCells with real new growth: {cells_with_real_growth}")
print(f"Of those, also flagged by simulation: {correctly_flagged} "
      f"({correctly_flagged/max(cells_with_real_growth,1)*100:.1f}%)")

total_eligible_cells = real_density.size
expected_random_overlap = cells_with_real_growth * (sim_has_growth.sum() / total_eligible_cells)
print(f"\nFor reference, random placement of the same total simulated growth "
      f"would be expected to hit ~{expected_random_overlap:.0f} of those cells by chance alone")
