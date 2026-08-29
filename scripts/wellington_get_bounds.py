import geopandas as gpd

gdf = gpd.read_file("../data/lcdb_wellington.gpkg")
bounds = gdf.to_crs(epsg=4326).total_bounds
print(f"AOI bounds (lon/lat): {list(bounds)}")
