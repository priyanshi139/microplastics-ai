"""Phase 2: non-destructive cleaning of the real NOAA NCEI export.
Every decision is logged. No imputation of the target, no synthetic rows, no outlier deletion."""
import pandas as pd, numpy as np, json
RAW = 'data/raw/NOAA_Marine_Microplastics_original.csv'
df = pd.read_csv(RAW, low_memory=False, encoding='utf-8-sig')
log = {}
log['records_raw'] = len(df)

# 1. Dates
df['sample_datetime'] = pd.to_datetime(df['Sample Date'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
df['year'] = df['sample_datetime'].dt.year
df['month'] = df['sample_datetime'].dt.month
log['dates_unparsed'] = int(df['sample_datetime'].isna().sum())

# 2. Coordinates
bad = (df['Latitude (degree)'].abs() > 90) | (df['Longitude (degree)'].abs() > 180) | df[['Latitude (degree)', 'Longitude (degree)']].isna().any(axis=1)
log['invalid_coordinates'] = int(bad.sum())
df = df.loc[~bad].copy()

# 3. Exact duplicates (all columns incl. IDs, and Unique ID)
n0 = len(df)
df = df.drop_duplicates()
df = df.drop_duplicates(subset='Unique ID')
log['exact_duplicates_removed'] = n0 - len(df)

# 4. Possible replicates: identical on every scientific field but different Unique ID.
#    These are NOT removed (replicate tows / non-detect zeros are legitimate); they are flagged only.
sci = [c for c in df.columns if c not in ['ObjectId', 'GlobalID', 'Unique ID', 'x', 'y', 'Symbology']]
df['possible_replicate_flag'] = df.duplicated(subset=sci, keep='first')
log['possible_replicates_flagged_not_removed'] = int(df['possible_replicate_flag'].sum())
log['possible_replicates_with_zero_value'] = int((df['possible_replicate_flag'] & (df['Microplastics Measurement'] == 0)).sum())

# 5. Harmonise obvious spelling variants in NEW columns (originals untouched)
country_map = {'USA': 'United States', 'Columbia': 'Colombia', 'Saudia Arabia': 'Saudi Arabia'}
df['Country_clean'] = df['Country'].replace(country_map)
region_map = {'Eastern China Sea': 'East China Sea',
              'Coastal Waters of Southeast Alaska and British Columbia': 'The Coastal Waters of Southeast Alaska and British Columbia'}
df['Region_clean'] = df['Region'].replace(region_map)
log['country_variants_fixed'] = {k: int((df['Country'] == k).sum()) for k in country_map}

# 6. Analysis groups: unit and matrix never mixed. 'value' is only comparable WITHIN a group.
def group(r):
    if r['Marine Setting'] == 'Nurdle Patrol': return 'nurdle_pieces_per_10min'
    if r['Marine Setting'] == 'Ocean sediment': return 'sediment_pieces_per_kg_dw'
    if r['Marine Setting'] == 'Beach': return 'beach_pieces_per_m3'
    if r['Marine Setting'] == 'Ocean water': return 'water_pieces_per_m3'
    return 'other'
df['analysis_group'] = df.apply(group, axis=1)
df['value'] = np.where(df['analysis_group'] == 'nurdle_pieces_per_10min',
                       df['Standardized Nurdle  Amount'], df['Microplastics Measurement'])
# consistency check: group <-> unit must be one-to-one
chk = pd.crosstab(df['analysis_group'], df['Unit'])
log['group_unit_crosstab'] = chk.to_dict()
log['records_without_value'] = int(df['value'].isna().sum())

# 7. Missing values: documented, NOT imputed (no imputation of target; features left NaN, models must handle explicitly)
df['mesh_missing_flag'] = df['Mesh Size (mm)'].isna()
df['ocean_missing_flag'] = df['Ocean'].isna()
df['is_zero'] = df['value'] == 0
df['log10_value'] = np.log10(df['value'] + 1e-6)  # explicit offset documented in report

# 8. Outlier FLAGS (log scale, IQR x3, within group). Nothing is deleted.
df['outlier_flag'] = False
for g, s in df.groupby('analysis_group'):
    pos = np.log10(s['value'].where(s['value'] > 0))
    q1, q3 = pos.quantile([.25, .75]); iqr = q3 - q1
    lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
    df.loc[s.index, 'outlier_flag'] = ((pos < lo) | (pos > hi)).fillna(False)
log['outliers_flagged_not_removed'] = df.groupby('analysis_group')['outlier_flag'].sum().astype(int).to_dict()

# 9. Save
log['records_clean'] = len(df)
log['records_per_group'] = df['analysis_group'].value_counts().to_dict()
df.to_csv('data/cleaned/NOAA_microplastics_cleaned_all.csv', index=False)
for g in df['analysis_group'].unique():
    df[df['analysis_group'] == g].to_csv(f'data/cleaned/NOAA_{g}.csv', index=False)
json.dump(log, open('reports/02_cleaning_log.json', 'w'), indent=2, default=str)
print(json.dumps(log, indent=2, default=str))
print(df.groupby('analysis_group')['value'].describe().to_string())
