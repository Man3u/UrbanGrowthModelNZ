import geopandas as gpd

gdf = gpd.read_file("../data/lcdb_wellington.gpkg")
print("Columns:", list(gdf.columns))
print("\nCRS:", gdf.crs)
print("\nRow count:", len(gdf))
print("\nSample Class_2018 values:")
print(gdf.filter(regex="Class_").head())
