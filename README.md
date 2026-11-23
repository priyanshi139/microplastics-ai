# AI for Microplastic Pollution Assessment: a reproducible analysis of the NOAA NCEI database

Research project: *Artificial Intelligence for Microplastic Pollution Assessment and Sustainable Environmental Management: Recent Advances, Challenges, and Future Perspectives.*

All analysis uses the **real, unmodified NOAA export** in `data/raw/`. No synthetic data and no imputed targets are used.

## Data source and citation

NOAA NCEI Global Marine Microplastics Data Collection (1972-present), exported through the official NOAA ArcGIS REST API.
DOI: https://doi.org/10.25921/d1bs-3m40
Data accessed: **12 November 2026**
Database description: Nyadjro et al., *Scientific Data* 2023, https://doi.org/10.1038/s41597-023-02632-y

Original file: `data/raw/NOAA_Marine_Microplastics_original.csv` (20,698,156 bytes)
SHA256: `6f83419d249dabbc7380d67b8a81dd5c8a2f4343ea80254989f236800d1cd1e3`

## What the data contains

- 29,075 records, 38 columns, sample dates 1972-04-25 to 2026-04-28, 96 NCEI accession numbers.
- Four measurement groups with **incompatible units, never mixed**:

| Group | Records | Unit |
|---|---|---|
| Ocean water | 16,987 | pieces/m3 |
| Nurdle Patrol (citizen science) | 10,615 | pieces/10 min |
| Beach | 893 | pieces/m3 |
| Ocean sediment | 580 | pieces/kg dry weight |

- No environmental covariates (temperature, salinity, currents) are present; only location, date, sampling method, mesh size and depth.

## Repository layout

```
data/raw/        original NOAA export, unchanged
data/cleaned/    cleaned master file, per-group files, India subset
scripts/         01_audit, 02_clean, 03_eda, 04_maps_hotspots, 05_ml_experiments, 07_india_analysis
notebooks/       05b_xgboost_shap_colab.ipynb (run in Google Colab)
figures/         EDA and ML figures
maps/            observation, hotspot and India maps
reports/         audit report, cleaning log, numeric outputs, model comparison
```

## How to reproduce

Run from the repository root:

```
pip install pandas numpy scikit-learn matplotlib scipy
python scripts/01_audit.py
python scripts/02_clean.py
python scripts/03_eda.py
python scripts/04_maps_hotspots.py
python scripts/05_ml_experiments.py
python scripts/07_india_analysis.py
```

XGBoost and SHAP: open `notebooks/05b_xgboost_shap_colab.ipynb` in Google Colab, upload `data/cleaned/NOAA_water_pieces_per_m3.csv` when asked, and run all cells.

Environments used for the reported ML numbers: laptop run (scripts 05) Python 3.10, scikit-learn 1.7.2, pandas 2.3.2, NumPy 2.2.6, SciPy 1.15.3, matplotlib 3.10.5; Colab run (XGBoost/SHAP) xgboost 3.4.1, shap 0.54.1, scikit-learn 1.6.1. Random seed 42. Spatial-validation scores vary slightly between library versions because the fold assignment can differ.

## Methods in brief

1. **Audit and cleaning:** no records deleted (29,075 before and after). 485 possible replicates (311 are zero values) are flagged, not removed. Outliers are flagged, not removed. Missing values are not imputed in the data; label variants (e.g. "USA"/"United States") are harmonised in new `_clean` columns only.
2. **EDA, maps, hotspots:** hotspots use only a comparable surface-net subset (Neuston/Manta net, mesh 0.3-0.4 mm, depth <= 0.5 m; 13,200 water records). Sampling density is not treated as pollution intensity (Spearman between cell sample count and cell median concentration = -0.055).
3. **ML (ocean water only):** target log10(pieces/m3 + 0.001); features latitude, longitude, year, month, sampling method, mesh size, water sample depth. Random 5-fold versus spatial block 5-fold (10 x 10 degree blocks) cross-validation. Preprocessing is fitted inside each training fold.
4. **India subset:** 189 Indian-origin records (Gulf of Mannar, Goa, two Maharashtra nurdle beaches), analysed descriptively only.

## Main results (all from the data in this repository)

| Model (all features) | R2 random 5-fold | R2 spatial block 5-fold |
|---|---|---|
| Mean baseline | -0.000 | -0.027 |
| Linear Regression | 0.540 | 0.456 |
| Random Forest | 0.778 | 0.569 |
| HistGradientBoosting | 0.765 | 0.636 |
| XGBoost (Colab) | 0.759 | 0.619 |

- Random splitting overstates performance; spatial validation lowers it for every model. Fold-to-fold variation is large (spatial R2 of Random Forest ranged 0.27-0.77), so the differences between the three tree models are not treated as significant. No significance test was run.
- Mesh size has by far the largest permutation importance (Random Forest, 0.762) and mean |SHAP| (XGBoost, 0.547). The models mainly learn the sampling protocol, not an environmental signal. Importance is not causation.
- Apparent increases in concentration over time coincide with smaller mesh sizes and changed sampling methods (Spearman mesh size vs year = -0.679), so the temporal trend cannot be attributed to environmental change.

## Limitations

- Heterogeneous methods, mesh sizes and units; strong geographic bias (Atlantic and Pacific dominate; Indian Ocean has few records).
- India coverage is limited to three regions and is not representative of the Indian coastline. Goa (steel bucket) and Gulf of Mannar (plankton net) water values are not comparable.
- Hotspots describe high concentrations in the compiled data, not verified pollution sources.
- No environmental covariates, so the models cannot explain why concentrations differ.

## AI-assistance disclosure

Analysis code and drafts were developed with assistance from Claude (Anthropic). Results marked as reproduced were re-run by the author. Check your university's policy on disclosing AI assistance and edit this line as required.
