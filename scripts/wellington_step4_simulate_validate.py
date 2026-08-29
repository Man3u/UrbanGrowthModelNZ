import os
import numpy as np
import rasterio
from scipy.ndimage import distance_transform_edt
from sklearn.metrics import cohen_kappa_score, accuracy_score
import joblib

DATA_DIR = "../data"
OUT_DIR = "../output"
DRIVER_DIR = "../output/drivers"

CLASS_NAMES = {0: "Water", 1: "Vegetation", 2: "Pasture/Agriculture", 3: "Built-up"}


def load(path):
    with rasterio.open(path) as src:
        return src.read(1), src.transform, src.crs


landcover_2018, transform, crs = load(f"{DATA_DIR}/wellington_landcover_2018.tif")
landcover_2023_real, _, _ = load(f"{DATA_DIR}/wellington_landcover_2023.tif")
pixel_size_m = transform.a

transition_matrix = np.load(f"{OUT_DIR}/transition_matrix_2012_2018.npy")
model = joblib.load(f"{OUT_DIR}/xgboost_transition_model.pkl")

built_2018 = landcover_2018 == 3
water_2018 = landcover_2018 == 0
dist_to_built_2018 = distance_transform_edt(~built_2018) * pixel_size_m
dist_to_water_2018 = distance_transform_edt(~water_2018) * pixel_size_m

with rasterio.open(f"{DRIVER_DIR}/slope_aligned.tif") as src:
    slope = src.read(1)
with rasterio.open(f"{DRIVER_DIR}/dist_to_roads.tif") as src:
    dist_to_roads = src.read(1)

n_classes = len(CLASS_NAMES)
class_counts_2018 = {c: int((landcover_2018 == c).sum()) for c in range(n_classes)}
expected_new_built = 0
for c in range(n_classes):
    if c == 3:
        continue
    expected_new_built += class_counts_2018[c] * transition_matrix[c, 3]
expected_new_built = int(round(expected_new_built))
print(f"Expected new built-up pixels (2018 -> 2023, per Markov matrix): {expected_new_built}")

eligible_mask = landcover_2018 != 3
eligible_idx = np.where(eligible_mask.ravel())[0]

X_eligible = np.column_stack([
    dist_to_built_2018.ravel()[eligible_idx],
    dist_to_water_2018.ravel()[eligible_idx],
    slope.ravel()[eligible_idx],
    dist_to_roads.ravel()[eligible_idx],
])
probs = model.predict_proba(X_eligible)[:, 1]

top_n_idx = eligible_idx[np.argsort(-probs)[:expected_new_built]]

simulated_2023 = landcover_2018.copy()
flat = simulated_2023.ravel()
flat[top_n_idx] = 3
simulated_2023 = flat.reshape(landcover_2018.shape)

sim_built = (simulated_2023 == 3)
real_built = (landcover_2023_real == 3)

overall_acc = accuracy_score(real_built.ravel(), sim_built.ravel())
kappa = cohen_kappa_score(real_built.ravel(), sim_built.ravel())

real_new_built = real_built & (landcover_2018 != 3)
sim_new_built = sim_built & (landcover_2018 != 3)
hits = (real_new_built & sim_new_built).sum()
recall_new_growth = hits / max(real_new_built.sum(), 1)

print(f"\n--- Validation: simulated 2023 vs real 2023 LCDB ---")
print(f"Overall accuracy (built vs non-built, whole map): {overall_acc:.4f}")
print(f"Cohen's Kappa (built vs non-built, whole map): {kappa:.4f}")
print(f"Real new built-up pixels (2018->2023): {int(real_new_built.sum())}")
print(f"Simulated new built-up pixels: {int(sim_new_built.sum())}")
print(f"Of real new growth, correctly located by simulation: {hits} ({recall_new_growth*100:.1f}%)")

np.save(f"{OUT_DIR}/simulated_landcover_2023.npy", simulated_2023)
print(f"\nSaved: {OUT_DIR}/simulated_landcover_2023.npy")
