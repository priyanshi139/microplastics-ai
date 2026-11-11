# Dataset Audit Report: NOAA NCEI Marine Microplastics export
Source file: Hub_Microplastics_Replace_5095440664801582733.csv (user-downloaded via NOAA ArcGIS REST API; original preserved as data/raw/NOAA_Marine_Microplastics_original.csv)
SHA256: 6f83419d249dabbc7380d67b8a81dd5c8a2f4343ea80254989f236800d1cd1e3 | Size: 20,698,156 bytes (about 19.7 MiB)
Records: 29,075 | Columns: 38 | Sample dates: 1972-04-25 to 2026-04-28 | 96 NCEI accessions / 96 DOIs
Metadata file: not received in this conversation (only the CSV was uploaded). Column meanings below are read from the data itself; confirm against NOAA documentation.

## Measurement groups (units are never mixed)
| Group | Records | Unit |
|---|---|---|
| Ocean water | 16,987 | pieces/m3 |
| Nurdle Patrol (citizen science, count in "Standardized Nurdle  Amount"; "Microplastics Measurement" empty) | 10,615 | pieces/10 mins |
| Beach | 893 | pieces/m3 |
| Ocean sediment | 580 | pieces kg-1 d.w. |

## Quality findings
- No exact duplicates, no duplicate IDs, no invalid or (0,0) coordinates, all dates parsed.
- 485 records match another record on every scientific field but have unique IDs (311 are zero values). Flagged as possible replicates, not removed.
- Missing: Country 59.8%, Region 54.2%, Subregion 93.7%, Ocean 4.8% (all from one nurdle organisation), Mesh size 36.7% (all nurdle records), Water sample depth 41.8%.
- 5,210 zero measurements (non-detects) among 18,460 measurements; heavily right-skewed (log scale required).
- Study type: Professional Science 16,857 | Citizen Science 12,218. Strong spatial and organisational clustering (Atlantic 19,394; Pacific 7,102; Indian 777; Arctic 386; Southern 27).
- Label issues: "USA"/"United States", "Columbia", "Saudia Arabia", "Eastern China Sea"/"East China Sea". Fixed in new *_clean columns only.
- Environmental variables present: depths, mesh size, sampling method only. No temperature, salinity, chlorophyll, wind or distance-to-coast.
- Beach values reach 800,000 pieces/m3 and are a different matrix from water; never compare directly.
