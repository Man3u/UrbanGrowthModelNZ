import os
import numpy as np
import rasterio
from scipy.ndimage import distance_transform_edt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import xgboost as xgb
import joblib

DATA_DIR = "../data"
OUT_DIR = "../output"
DRIVER_DIR = "../output/drivers"
os.makedirs(OUT_DIR, exist_ok=True)

CLASS_NAMES = {0: "Water", 1: "Vegetation", 2: "Pasture/Agriculture", 3: "Built-up"}


def load(path):
    with rasterio.open(path) as src:
        return src.read(1), src.transform, src.crs


landcover_2012, transform, crs = load(f"{DATA_DIR}/wellington_landcover_2012.tif")
landcover_2018, _, _ = load(f"{DATA_DIR}/wellington_landcover_2018.tif")
landcover_2023, _, _ = load(f"{DATA_DIR}/wellington_landcover_2023.tif")
pixel_size_m = transform.a

n_classes = len(CLASS_NAMES)
transition_counts = np.zeros((n_classes, n_classes))
for i in range(n_classes):
    for j in range(n_classes):
        transition_counts[i, j] = np.sum((landcover_2012 == i) & (landcover_2018 == j))
row_sums = transition_counts.sum(axis=1, keepdims=True)
transition_matrix = np.divide(
    transition_counts, row_sums, out=np.zeros_like(transition_counts), where=row_sums != 0
)
np.save(f"{OUT_DIR}/transition_matrix_2012_2018.npy", transition_matrix)
print("Markov transition matrix (2012 -> 2018) recomputed and saved.\n")

built_2012 = landcover_2012 == 3
water_2012 = landcover_2012 == 0
dist_to_built_2012 = distance_transform_edt(~built_2012) * pixel_size_m
dist_to_water_2012 = distance_transform_edt(~water_2012) * pixel_size_m

with rasterio.open(f"{DRIVER_DIR}/slope_aligned.tif") as src:
    slope = src.read(1)
with rasterio.open(f"{DRIVER_DIR}/dist_to_roads.tif") as src:
    dist_to_roads = src.read(1)

eligible = landcover_2012 != 3
X = np.column_stack([
    dist_to_built_2012[eligible],
    dist_to_water_2012[eligible],
    slope[eligible],
    dist_to_roads[eligible],
])
y = (landcover_2018[eligible] == 3).astype(int)

print(f"Training examples: {len(y)} pixels ({y.sum()} converted to built-up, {y.sum()/len(y)*100:.3f}%)")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

model = xgb.XGBClassifier(
    n_estimators=200, max_depth=5, learning_rate=0.1,
    scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
    eval_metric="logloss", random_state=42,
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
print(f"\nHeld-out test accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Held-out test ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
print(classification_report(y_test, y_pred, target_names=["stayed non-built", "converted to built"]))

importances = dict(zip(
    ["dist_to_built", "dist_to_water", "slope", "dist_to_roads"], model.feature_importances_
))
print("Feature importances:", {k: round(float(v), 3) for k, v in importances.items()})

joblib.dump(model, f"{OUT_DIR}/xgboost_transition_model.pkl")
print(f"\nSaved: {OUT_DIR}/xgboost_transition_model.pkl")
