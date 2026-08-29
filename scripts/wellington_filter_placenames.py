import geopandas as gpd

gdf = gpd.read_file("../data/linz_placenames_wellington.gpkg")

targets = ["Wellington", "Lower Hutt", "Porirua", "Upper Hutt", "Paraparaumu"]
matches = gdf[gdf["name"].isin(targets)]
print(matches[["name", "feat_type"]].to_string())
