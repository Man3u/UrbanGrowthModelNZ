import ee

ee.Initialize(project="floodmapping-506505")

AOI = ee.Geometry.Rectangle([174.3793932553768, -41.6138067392097, 176.30734240500703, -40.39444216558586])

dem = ee.Image("USGS/SRTMGL1_003").select("elevation").clip(AOI)
slope = ee.Terrain.slope(dem).rename("slope")

task = ee.batch.Export.image.toDrive(
    image=slope,
    description="Slope_Wellington",
    folder="UrbanGrowthExports",
    fileNamePrefix="wellington_slope",
    region=AOI,
    scale=30,
    maxPixels=1e10,
)
task.start()
print("Started export: Slope_Wellington")
print("Done. Check https://code.earthengine.google.com/tasks")
