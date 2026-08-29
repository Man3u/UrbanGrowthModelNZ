import rasterio
import numpy as np

for year in [2012, 2018, 2023]:
    with rasterio.open(f"../data/wellington_landcover_{year}.tif") as src:
        arr = src.read(1)
    total = arr.size
    built = (arr == 3).sum()
    print(f"{year}: {built} built-up pixels / {total} total = {built/total*100:.4f}%")
