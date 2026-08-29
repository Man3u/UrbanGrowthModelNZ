# Wellington Region Urban Growth Projection Model

A CA-Markov + XGBoost land cover change model for the Greater Wellington Region, New Zealand, projecting urban growth from 2023 to 2035 using verified historical land cover data (2012, 2018, 2023) and validating the model against real outcomes before forecasting.

This is a calibrated, one-shot projection pipeline (calibrate → validate → project once to 2035), not an interactive or re-runnable simulator with adjustable scenarios, the name reflects that.

## Key Finding

The model correctly predicted the *scale* of Wellington's urban growth: simulating 2018→2023 forward produced 7,603 new built-up pixels against a real outcome of 7,273, within 4.5%. At a 300m neighborhood scale, the model identified genuine growth hotspots roughly 39x better than random chance (25.4% hit rate vs. ~0.6% expected by chance). Distance to the road network turned out to be the single dominant predictor of where growth occurs (81.6% feature importance), far ahead of distance to existing development, slope, or distance to water, confirming that new development follows transport corridors far more than it follows any other physical driver. Projecting forward, built-up area is expected to grow from 1.14% to 1.21% of the region by 2035, adding roughly 15.4 km² of new development, consistent with the 13.4 km² added over the previous 11 years (2012–2023).

## Methodology Note: Why New Zealand, Not the Original Study Area

This project originally targeted Hyderabad, India, using Landsat 8 imagery classified with spectral index thresholds (NDVI/NDBI/MNDWI) in the absence of hand-labeled ground truth. That approach produced land cover maps that were internally inconsistent across years, built-up area appeared to swing implausibly (e.g., dropping by nearly half between 2013 and 2019, then partially recovering by 2024) because bare/fallow agricultural land and impervious urban surfaces are spectrally similar in a semi-arid, dry-season context, and auto-labeled training data had no way to correct for it. Several fixes were attempted (pooling training samples across years into a single classifier, adding a texture-based discriminator) and each improved the picture without fully resolving it.

The project was rebuilt on Wellington, New Zealand specifically because New Zealand's Land Cover Database (LCDB), maintained by Manaaki Whenua / LINZ, provides real, independently verified land cover classifications at five-yearly intervals (1996, 2001, 2008, 2012, 2018, 2023) — removing the ground-truth problem entirely. Every result in this repository is built on that verified data, not on inferred classification.

## Methodology

**Study area:** Greater Wellington Region (bounding box: 174.38°E–176.31°E, 41.61°S–40.39°S), rasterized to a 30m grid in EPSG:2193 (NZTM2000).

**Land cover classes:** LCDB's ~33 detailed classes were reclassified into four categories, Water, Vegetation, Pasture/Agriculture, and Built-up for the 2012, 2018, and 2023 editions.

**Markov transition matrix:** Computed by cross-tabulating land cover class at 2012 against 2018 (and separately 2018 against 2023 for the forward projection), giving the historical probability of each class transitioning to each other class.

**Transition-potential model:** An XGBoost binary classifier predicts, for every non-built pixel, its probability of converting to built-up, using four driver variables: distance to existing built-up area, distance to water, slope (from SRTM 30m DEM), and distance to the road network (from LINZ's official Topo50 road centrelines). Roads dominate the model (81.6% feature importance).

**Validation:** The calibrated model was used to simulate 2018→2023 forward, and the result was checked against the real 2023 LCDB edition — both for aggregate growth quantity (strong match) and spatial location (modest at the exact-pixel level, meaningfully better than chance at a 300m neighborhood scale, consistent with the fact that specific development sites are also shaped by zoning and consent decisions that generic physical drivers can't fully capture).

**Forward simulation:** The model was recalibrated on the most recent complete interval (2018→2023) and projected forward in three ~4-year steps (2023→2027→2031→2035), recomputing distance-to-built-up at each step so new growth influences where subsequent growth is likely to occur consistent with how cities actually expand outward from existing development.

## Data Sources

| Dataset | Source | Use |
|---|---|---|
| Land Cover Database (LCDB v6.0) | LINZ / Manaaki Whenua, via LRIS Portal | Land cover classification, 2012/2018/2023 editions |
| NZ Road Centrelines (Topo, 1:50k) | LINZ Data Service | Distance-to-roads driver |
| NZ Place Names (NZGB) | LINZ Data Service | City/town labels for final map |
| SRTM 30m DEM | USGS, via Google Earth Engine | Slope driver |

## Repository Structure

```
UrbanGrowthSimulator/
├── scripts/          # Python + Earth Engine processing scripts
├── data/             # LCDB editions, roads, place names, slope (not tracked if large)
├── output/           # Transition matrices, trained models, simulated land cover
└── maps/             # Final QGIS map exports
```

## Tech Stack

Python (rasterio, geopandas, scikit-learn, XGBoost, scipy), Google Earth Engine (SRTM slope export), QGIS (final cartography), LINZ Data Service & LRIS Portal (source data).

## Author

Manu
