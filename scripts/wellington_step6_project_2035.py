import os
import numpy as np
import rasterio
from scipy.ndimage import distance_transform_edt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import xgboost as xgb
import joblib

DATA_DIR = "../data"
OUT_DIR = "../output"
DRIVER_DIR = "../output/drivers"

CLASS_NAMES = {0: "Water", 1: "Vegetation", 2: "Pasture/Agriculture", 3: "Built-up"}


def load(path):
    with rasterio.open(path) as src:
        return src.read(1), src.transform, src.crs


landcover_2018, transform, crs = load(f"{DATA_DIR}/wellington_landcover_2018.tif")
landcover_2023, _, _ = load(f"{DATA_DIR}/wellington_landcover_2023.tif")
pixel_size_m = transform.a

with rasterio.open(f"{DRIVER_DIR}/slope_aligned.tif") as src:
    slope = src.read(1)
with rasterio.open(f"{DRIVER_DIR}/dist_to_roads.tif") as src:
    dist_to_roads = src.read(1)

n_classes = len(CLASS_NAMES)
transition_counts = np.zeros((n_classes, n_classes))
for i in range(n_classes):
    for j in range(n_classes):
        transition_counts[i, j] = np.sum((landcover_2018 == i) & (landcover_2023 == j))
row_sums = transition_counts.sum(axis=1, keepdims=True)
transition_matrix_5yr = np.divide(
    transition_counts, row_sums, out=np.zeros_like(transition_counts), where=row_sums != 0
)
np.save(f"{OUT_DIR}/transition_matrix_2018_2023.npy", transition_matrix_5yr)
print("Recalibrated Markov transition matrix on 2018 -> 2023 (most recent real interval)\n")

built_2018 = landcover_2018 == 3
water_2018 = landcover_2018 == 0
dist_to_built_2018 = distance_transform_edt(~built_2018) * pixel_size_m
dist_to_water_2018 = distance_transform_edt(~water_2018) * pixel_size_m

eligible = (landcover_2018 != 3) & (landcover_2018 != 0)
X = np.column_stack([
    dist_to_built_2018[eligible], dist_to_water_2018[eligible],
    slope[eligible], dist_to_roads[eligible],
])
y = (landcover_2023[eligible] == 3).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
model = xgb.XGBClassifier(
    n_estimators=200, max_depth=5, learning_rate=0.1,
    scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
    eval_metric="logloss", random_state=42,
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
print(f"Retrained on 2018->2023: held-out accuracy {accuracy_score(y_test, y_pred):.4f}, "
      f"ROC-AUC {roc_auc_score(y_test, y_prob):.4f}\n")
joblib.dump(model, f"{OUT_DIR}/xgboost_transition_model_2018_2023.pkl")

STEPS = [2027, 2031, 2035]
YEARS_PER_STEP = 4
BASE_INTERVAL_YEARS = 5

current_state = landcover_2023.copy()
current_year = 2023

for target_year in STEPS:
    step_years = target_year - current_year
    scale = step_years / BASE_INTERVAL_YEARS

    built_mask = current_state == 3
    water_mask = current_state == 0
    dist_to_built = distance_transform_edt(~built_mask) * pixel_size_m
    dist_to_water = distance_transform_edt(~water_mask) * pixel_size_m

    class_counts = {c: int((current_state == c).sum()) for c in range(n_classes)}
    expected_new_built = 0
    for c in range(n_classes):
        if c == 3:
            continue
        expected_new_built += class_counts[c] * transition_matrix_5yr[c, 3] * scale
    expected_new_built = int(round(expected_new_built))

    eligible_mask = (current_state != 3) & (current_state != 0)
    eligible_idx = np.where(eligible_mask.ravel())[0]

    if expected_new_built > len(eligible_idx):
        print(f"  WARNING: Markov demand ({expected_new_built}) exceeds remaining "
              f"eligible land ({len(eligible_idx)}) - clamping to available land.")
        expected_new_built = len(eligible_idx)

    X_step = np.column_stack([
        dist_to_built.ravel()[eligible_idx], dist_to_water.ravel()[eligible_idx],
        slope.ravel()[eligible_idx], dist_to_roads.ravel()[eligible_idx],
    ])
    probs = model.predict_proba(X_step)[:, 1]
    top_n_idx = eligible_idx[np.argsort(-probs)[:expected_new_built]]

    flat = current_state.ravel().copy()
    flat[top_n_idx] = 3
    current_state = flat.reshape(current_state.shape)

    built_pct = (current_state == 3).sum() / current_state.size * 100
    print(f"{current_year} -> {target_year} ({step_years}yr step): "
          f"+{expected_new_built} built-up pixels, now {built_pct:.4f}% of AOI")

    out_path = f"{OUT_DIR}/simulated_landcover_{target_year}.npy"
    np.save(out_path, current_state)
    print(f"  Saved: {out_path}")

    current_year = target_year

print("\nDone. Final 2035 projection saved.")
