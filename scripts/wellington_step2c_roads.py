import os
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt

DATA_DIR = "../data"
OUT_DIR = "../output/drivers"
os.makedirs(OUT_DIR, exist_ok=True)

with rasterio.open(f"{DATA_DIR}/wellington_landcover_2012.tif") as ref:
    ref_transform = ref.transform
    ref_crs = ref.crs
    ref_shape = (ref.height, ref.width)
    pixel_size_m = ref_transform.a

roads = gpd.read_file(f"{DATA_DIR}/linz_roads_wellington.gpkg")
if roads.crs != ref_crs:
    roads = roads.to_crs(ref_crs)
print(f"Loaded {len(roads)} road segments, CRS: {roads.crs}")

road_shapes = [(geom, 1) for geom in roads.geometry]
road_raster = rasterize(
    road_shapes, out_shape=ref_shape, transform=ref_transform, fill=0, dtype="uint8"
)

dist_to_roads = distance_transform_edt(road_raster == 0) * pixel_size_m

out_path = f"{OUT_DIR}/dist_to_roads.tif"
with rasterio.open(
    out_path, "w", driver="GTiff", height=ref_shape[0], width=ref_shape[1],
    count=1, dtype="float32", crs=ref_crs, transform=ref_transform,
) as dst:
    dst.write(dist_to_roads.astype("float32"), 1)

print(f"Saved: {out_path}")
print(f"Distance-to-roads range: {dist_to_roads.min():.1f}m to {dist_to_roads.max():.1f}m, mean {dist_to_roads.mean():.1f}m")
