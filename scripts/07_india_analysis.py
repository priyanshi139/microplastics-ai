"""Phase 7: India-focused descriptive analysis of the real NOAA data.
India is NOT identifiable from the Country column (only 2 labelled records), so the subset is defined explicitly:
  records inside the box 5-25N, 66-90E that come from Indian institutions (Thoothukudi, Goa University)
  OR nurdle records whose Country == 'India'.
Records in the box from other origins (5 Gyres open-ocean south of Sri Lanka, Adventure Scientist Sri Lanka coast,
Sri Lanka nurdle beaches) are listed and EXCLUDED. Descriptive only: n is far too small and methods differ for ML or hotspots."""
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
df = pd.read_csv('data/cleaned/NOAA_microplastics_cleaned_all.csv', low_memory=False)
LAT, LON = 'Latitude (degree)', 'Longitude (degree)'
box = df[df[LAT].between(5, 25) & df[LON].between(66, 90)].copy()
indian_org = box['Organization'].str.contains('Thoothukudi|Goa University', na=False)
india_nurdle = (box['analysis_group'] == 'nurdle_pieces_per_10min') & (box['Country'] == 'India')
box['india_subset'] = indian_org | india_nurdle
ind = box[box.india_subset].copy(); exc = box[~box.india_subset]
ind['site'] = np.where(ind['Organization'].str.contains('Thoothukudi'), 'Gulf of Mannar (Thoothukudi, Tamil Nadu)',
              np.where(ind['Organization'].str.contains('Goa'), 'Goa beaches (Arabian Sea)', 'Maharashtra beaches (nurdle)'))
out = []
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); out.append(s)
P(f'Box records: {len(box)} | India subset: {len(ind)} | excluded (not Indian-origin): {len(exc)}')
P('Excluded by organisation / group:\n' + exc.groupby(['Organization', 'analysis_group']).size().to_string())
P('Excluded lat/lon ranges:\n' + exc.groupby('Organization')[[LAT, LON]].agg(['min', 'max']).round(2).to_string())
P('\nIndia subset by site x group (unit):\n' + ind.groupby(['site', 'analysis_group', 'Unit']).size().to_string())
g = ind.groupby(['site', 'analysis_group']).agg(n=('value', 'size'), years=('year', lambda s: f'{s.min()}-{s.max()}'), lat=(LAT, 'mean'), lon=(LON, 'mean'),
        zeros=('value', lambda s: int((s == 0).sum())), median=('value', 'median'), p90=('value', lambda s: s.quantile(.9)), max=('value', 'max'),
        methods=('Sampling Method', lambda s: '; '.join(s.unique())), mesh_mm=('Mesh Size (mm)', lambda s: f'{s.min():.3g}-{s.max():.3g}' if s.notna().any() else 'NA'))
P('\nSummary:\n' + g.round(3).to_string()); g.round(3).to_csv('reports/07_india_summary.csv')
ind.to_csv('data/cleaned/NOAA_india_subset.csv', index=False)
P('\nConcentration class (NOAA-assigned) by site:\n' + pd.crosstab(ind['site'] + ' | ' + ind['analysis_group'], ind['Concentration Class']).to_string())
P('\nIndian coastline states with any data: Tamil Nadu (Gulf of Mannar), Goa, Maharashtra (2 nurdle beaches). No data from Kerala, Karnataka, Gujarat, Andhra Pradesh, Odisha, West Bengal or the island territories in this subset.')
# Thoothukudi paired water vs sediment
th = ind[ind.site.str.contains('Mannar')]
w = th[th.analysis_group == 'water_pieces_per_m3']; s = th[th.analysis_group == 'sediment_pieces_per_kg_dw']
P(f'\nThoothukudi: water n={len(w)}, sediment n={len(s)}; stations share coordinates: {w[[LAT, LON]].round(4).drop_duplicates().shape[0]} water coords, {s[[LAT, LON]].round(4).drop_duplicates().shape[0]} sediment coords')

# Figures
fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
cm = {'Gulf of Mannar (Thoothukudi, Tamil Nadu)': '#d62728', 'Goa beaches (Arabian Sea)': '#1f77b4', 'Maharashtra beaches (nurdle)': '#2ca02c'}
for k, c in cm.items():
    s_ = ind[ind.site == k]; ax[0].scatter(s_[LON], s_[LAT], s=14, c=c, label=f'{k} (n={len(s_)})')
ax[0].scatter(exc[LON], exc[LAT], s=14, c='lightgray', marker='x', label=f'excluded, not Indian-origin (n={len(exc)})')
ax[0].set_xlim(66, 90); ax[0].set_ylim(5, 25); ax[0].grid(alpha=.3); ax[0].legend(fontsize=6, loc='upper right'); ax[0].set_xlabel('Longitude'); ax[0].set_ylabel('Latitude'); ax[0].set_title('India subset and excluded points')
wt = ind[ind.analysis_group == 'water_pieces_per_m3']
data = [np.log10(wt.loc[wt.site == k, 'value'] + 1e-3) for k in ['Gulf of Mannar (Thoothukudi, Tamil Nadu)', 'Goa beaches (Arabian Sea)']]
ax[1].boxplot(data, tick_labels=['Mannar\n(plankton net)', 'Goa\n(steel bucket)'], showfliers=True); ax[1].set_ylabel('log10(pieces/m$^3$ + 1e-3)')
ax[1].set_title('Water: different methods, NOT directly comparable')
plt.tight_layout(); plt.savefig('maps/map06_india_subset_and_methods.png', dpi=150); plt.close()
open('reports/07_india_numbers.txt', 'w').write('\n'.join(out))
