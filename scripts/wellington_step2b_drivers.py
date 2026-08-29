import os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from scipy.ndimage import distance_transform_edt

DATA_DIR = "../data"
OUT_DIR = "../output/drivers"
os.makedirs(OUT_DIR, exist_ok=True)

with rasterio.open(f"{DATA_DIR}/wellington_landcover_2012.tif") as ref:
    ref_transform = ref.transform
    ref_crs = ref.crs
    ref_shape = (ref.height, ref.width)
    landcover_2012 = ref.read(1)
    pixel_size_m = ref_transform.a  # already meters - EPSG:2193 is projected

print(f"Reference grid: {ref_shape}, CRS: {ref_crs}, pixel size: {pixel_size_m:.1f} m")


def save_raster(array, path, dtype="float32"):
    with rasterio.open(
        path, "w", driver="GTiff", height=ref_shape[0], width=ref_shape[1],
        count=1, dtype=dtype, crs=ref_crs, transform=ref_transform,
    ) as dst:
        dst.write(array.astype(dtype), 1)
    print(f"  Saved: {path}")


built_mask = landcover_2012 == 3
water_mask = landcover_2012 == 0

dist_to_built = distance_transform_edt(~built_mask) * pixel_size_m
dist_to_water = distance_transform_edt(~water_mask) * pixel_size_m

save_raster(dist_to_built, f"{OUT_DIR}/dist_to_built_2012.tif")
save_raster(dist_to_water, f"{OUT_DIR}/dist_to_water_2012.tif")

with rasterio.open(f"{DATA_DIR}/wellington_slope.tif") as src:
    slope_aligned = np.zeros(ref_shape, dtype="float32")
    reproject(
        source=rasterio.band(src, 1),
        destination=slope_aligned,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=ref_transform,
        dst_crs=ref_crs,
        resampling=Resampling.bilinear,
    )
save_raster(slope_aligned, f"{OUT_DIR}/slope_aligned.tif")

print("\nDone. All driver rasters saved to", OUT_DIR)
