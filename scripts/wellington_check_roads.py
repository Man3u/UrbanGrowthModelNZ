import geopandas as gpd

gdf = gpd.read_file("../data/linz_roads_wellington.gpkg")
print("Columns:", list(gdf.columns))
print("CRS:", gdf.crs)
print("Row count:", len(gdf))
print("Geometry types:", gdf.geometry.geom_type.unique())
