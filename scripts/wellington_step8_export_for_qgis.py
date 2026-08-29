import numpy as np
import rasterio

DATA_DIR = "../data"
OUT_DIR = "../output"


def load_tif(path):
    with rasterio.open(path) as src:
        return src.read(1), src.transform, src.crs


def save_tif(array, path, transform, crs, dtype="uint8"):
    with rasterio.open(
        path, "w", driver="GTiff", height=array.shape[0], width=array.shape[1],
        count=1, dtype=dtype, crs=crs, transform=transform,
    ) as dst:
        dst.write(array.astype(dtype), 1)
    print(f"Saved: {path}")


landcover_2012, transform, crs = load_tif(f"{DATA_DIR}/wellington_landcover_2012.tif")
landcover_2023, _, _ = load_tif(f"{DATA_DIR}/wellington_landcover_2023.tif")
landcover_2035 = np.load(f"{OUT_DIR}/simulated_landcover_2035.npy")

save_tif(landcover_2035, f"{DATA_DIR}/wellington_landcover_2035.tif", transform, crs)

built_2012 = landcover_2012 == 3
built_2023 = landcover_2023 == 3
built_2035 = landcover_2035 == 3

growth_map = np.zeros(landcover_2012.shape, dtype="uint8")
growth_map[built_2012] = 1
growth_map[built_2023 & ~built_2012] = 2
growth_map[built_2035 & ~built_2023] = 3

save_tif(growth_map, f"{DATA_DIR}/wellington_growth_map.tif", transform, crs)

print("\nDone. Both rasters are ready to load into QGIS.")
