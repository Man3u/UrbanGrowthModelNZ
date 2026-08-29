import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds

DATA_DIR = "../data"
YEARS = [2012, 2018, 2023]
RESOLUTION_M = 30

# Water=0, Vegetation=1, Pasture/Agriculture=2, Built-up=3
CLASS_MAP = {
    # Water
    20: 0, 21: 0, 22: 0,
    # Built-up
    1: 3, 2: 3, 5: 3, 6: 3,
    # Pasture / Agriculture
    30: 2, 33: 2, 40: 2, 41: 2, 44: 2,
    # Vegetation (forest, scrub, wetland veg, tussock, bare natural/rock/snow)
    10: 1, 11: 1, 12: 1, 14: 1, 15: 1, 16: 1,
    43: 1, 45: 1, 46: 1, 47: 1, 50: 1, 51: 1, 52: 1,
    54: 1, 55: 1, 56: 1, 58: 1, 64: 1, 68: 1, 69: 1, 70: 1, 71: 1,
}

gdf = gpd.read_file(f"{DATA_DIR}/lcdb_wellington.gpkg")
print(f"Loaded {len(gdf)} polygons, CRS: {gdf.crs}")

gdf = gdf.to_crs(epsg=2193)
print(f"Reprojected to EPSG:2193")

minx, miny, maxx, maxy = gdf.total_bounds
width = int(np.ceil((maxx - minx) / RESOLUTION_M))
height = int(np.ceil((maxy - miny) / RESOLUTION_M))
transform = from_bounds(minx, miny, maxx, maxy, width, height)
print(f"Grid: {height} x {width} at {RESOLUTION_M}m")

for year in YEARS:
    col = f"Class_{year}"
    codes = gdf[col]
    unmapped = sorted(set(codes.unique()) - set(CLASS_MAP.keys()))
    if unmapped:
        print(f"  WARNING {year}: unmapped LCDB codes found: {unmapped} - check these against the LCDB class list")

    mapped = codes.map(CLASS_MAP)
    valid = gdf[mapped.notna()].copy()
    valid["class_id"] = mapped[mapped.notna()].astype(int)

    shapes = [(geom, val) for geom, val in zip(valid.geometry, valid["class_id"])]
    raster = rasterize(
        shapes, out_shape=(height, width), transform=transform,
        fill=1, dtype="uint8",  # default to Vegetation for any gaps
    )

    out_path = f"{DATA_DIR}/wellington_landcover_{year}.tif"
    with rasterio.open(
        out_path, "w", driver="GTiff", height=height, width=width,
        count=1, dtype="uint8", crs="EPSG:2193", transform=transform,
    ) as dst:
        dst.write(raster, 1)

    total = raster.size
    print(f"{year}: Water {np.mean(raster==0)*100:.1f}% | Vegetation {np.mean(raster==1)*100:.1f}% | "
          f"Pasture/Ag {np.mean(raster==2)*100:.1f}% | Built-up {np.mean(raster==3)*100:.1f}%")
    print(f"  Saved: {out_path}\n")

print("Done.")
