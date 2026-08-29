import geopandas as gpd

gdf = gpd.read_file("../data/linz_placenames_wellington.gpkg")
print("Columns:", list(gdf.columns))
print("Row count:", len(gdf))
print("\nSample rows:")
print(gdf.head(10))
