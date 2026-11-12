"""Phase 4: maps and hotspot analysis on the real NOAA data.
No GeoPandas/Folium needed (plain lat/lon scatter; points outline coasts). A Folium version is in README notes.
Key rule: sampling density != pollution intensity. Hotspots are defined by concentration, with minimum sample size,
and only on a methodologically comparable subset (surface nets, similar mesh) to limit method confounding."""
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.stats import spearmanr
plt.rcParams.update({'figure.dpi': 150, 'font.size': 8})
df = pd.read_csv('data/cleaned/NOAA_microplastics_cleaned_all.csv', low_memory=False)
LAT, LON = 'Latitude (degree)', 'Longitude (degree)'
txt = []
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); txt.append(s)

# ---------- Map 1: all observations by group (sampling coverage, NOT intensity)
cols = {'water_pieces_per_m3': '#1f77b4', 'beach_pieces_per_m3': '#d62728', 'sediment_pieces_per_kg_dw': '#2ca02c', 'nurdle_pieces_per_10min': '#ff7f0e'}
fig, ax = plt.subplots(figsize=(11, 5.2))
for g, c in cols.items():
    s = df[df.analysis_group == g]; ax.scatter(s[LON], s[LAT], s=2, c=c, alpha=.5, label=f'{g} (n={len(s)})', linewidths=0)
ax.set_xlim(-180, 180); ax.set_ylim(-80, 90); ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude'); ax.grid(alpha=.3)
ax.legend(markerscale=5, loc='lower left'); ax.set_title('Sampling coverage of the NOAA database (shows WHERE people sampled, not pollution level)')
plt.tight_layout(); plt.savefig('maps/map01_world_observations_by_group.png'); plt.close()

# ---------- Map 2: water concentration (log10), high values drawn on top
w = df[df.analysis_group == 'water_pieces_per_m3'].sort_values('log10_value')
fig, ax = plt.subplots(figsize=(11, 5.2))
sc = ax.scatter(w[LON], w[LAT], c=w['log10_value'].clip(-4, 4), s=3, cmap='viridis', linewidths=0)
plt.colorbar(sc, label='log10(pieces/m$^3$ + 1e-6), clipped to [-4, 4]')
ax.set_xlim(-180, 180); ax.set_ylim(-80, 90); ax.grid(alpha=.3); ax.set_title('Ocean-water microplastic concentration (all methods, n=%d)' % len(w))
plt.tight_layout(); plt.savefig('maps/map02_world_water_concentration.png'); plt.close()

# ---------- Comparable subset for hotspot work
comp = w[(w['Sampling Method'].isin(['Neuston net', 'Manta net'])) & (w['Mesh Size (mm)'].between(0.3, 0.4)) &
         (w['Water Sample Depth (m)'].fillna(0) <= 0.5)].copy()
P(f'Water records: {len(w)} | comparable surface-net subset (Neuston/Manta, mesh 0.3-0.4 mm, depth<=0.5 m): {len(comp)}')
P('Comparable subset by ocean:\n' + comp['Ocean'].fillna('NA').value_counts().to_string())

# ---------- Sampling density vs intensity on 5-degree cells
comp['cy'] = (np.floor(comp[LAT] / 5) * 5 + 2.5); comp['cx'] = (np.floor(comp[LON] / 5) * 5 + 2.5)
cell = comp.groupby(['cy', 'cx'])['value'].agg(n='count', median='median', p90=lambda x: x.quantile(.9), frac_gt1=lambda x: (x > 1).mean()).reset_index()
rho, p = spearmanr(cell['n'], cell['median'])
P(f'\n5-deg cells: {len(cell)} | Spearman(sample count, median concentration) = {rho:.3f} (p={p:.3g})  -> density is NOT a proxy for intensity')
ok = cell[cell.n >= 10].copy()
thr = ok['median'].quantile(.9)
ok['hotspot'] = ok['median'] >= thr
P(f'Cells with n>=10: {len(ok)} | hotspot rule: median >= 90th percentile of cell medians ({thr:.4g} pieces/m3) -> {ok.hotspot.sum()} cells')
hs = ok[ok.hotspot].sort_values('median', ascending=False)
P('Hotspot cells (centre lat, lon, n, median, p90, share >1 piece/m3):\n' + hs.round(3).to_string(index=False))
ok.to_csv('reports/04_cell_summary_comparable_nets.csv', index=False)

fig, ax = plt.subplots(figsize=(11, 5.2))
sc = ax.scatter(ok['cx'], ok['cy'], c=np.log10(ok['median'] + 1e-4), s=(ok['n'].clip(upper=400) ** .5) * 6, cmap='YlOrRd', edgecolors='gray', linewidths=.4)
ax.scatter(hs['cx'], hs['cy'], s=(hs['n'].clip(upper=400) ** .5) * 6, facecolors='none', edgecolors='black', linewidths=1.5, label='hotspot cell (top 10% median, n>=10)')
plt.colorbar(sc, label='log10(median pieces/m$^3$)'); ax.set_xlim(-180, 180); ax.set_ylim(-80, 90); ax.grid(alpha=.3)
ax.set_title('Cell medians, comparable surface-net subset (marker size = sample count)'); ax.legend(loc='lower left')
plt.tight_layout(); plt.savefig('maps/map03_hotspot_cells_comparable_nets.png'); plt.close()

# ---------- DBSCAN: density clusters vs high-concentration clusters (haversine, eps = 300 km)
R = 6371.0; eps = 300 / R
def db(d, min_samples):
    X = np.radians(d[[LAT, LON]].values)
    return DBSCAN(eps=eps, min_samples=min_samples, metric='haversine', algorithm='ball_tree').fit_predict(X)
comp['c_all'] = db(comp, 20)
hi = comp[comp['value'] >= comp['value'].quantile(.9)].copy(); hi['c_hi'] = db(hi, 10)
P(f'\nDBSCAN (eps=300 km) on all comparable points (min 20): {comp.c_all.max()+1} sampling-density clusters, noise={(comp.c_all==-1).sum()}')
P(f'DBSCAN on the top-10% concentration points (value >= {comp["value"].quantile(.9):.4g}; min 10): {hi.c_hi.max()+1} high-concentration clusters')
rows = []
for k, s in hi[hi.c_hi >= 0].groupby('c_hi'):
    allin = comp[comp[LAT].between(s[LAT].min() - .5, s[LAT].max() + .5) & comp[LON].between(s[LON].min() - .5, s[LON].max() + .5)]  # bounding box of the cluster
    rows.append(dict(cluster=k, n_high=len(s), lat=s[LAT].mean(), lon=s[LON].mean(), n_all_in_bbox=len(allin), share_high_in_bbox=len(s) / max(len(allin), 1),
                     median_value=s.value.median(), orgs='; '.join(s['Organization'].value_counts().head(2).index)))
cl = pd.DataFrame(rows).sort_values('n_high', ascending=False)
P('High-concentration clusters:\n' + cl.round(3).to_string(index=False)); cl.to_csv('reports/04_dbscan_high_concentration_clusters.csv', index=False)
fig, ax = plt.subplots(figsize=(11, 5.2))
ax.scatter(comp[LON], comp[LAT], s=2, c='lightgray', label='comparable surface-net samples')
cc = hi[hi.c_hi >= 0]; ax.scatter(cc[LON], cc[LAT], s=8, c=cc.c_hi, cmap='tab20', label='high-concentration DBSCAN clusters')
ax.set_xlim(-180, 180); ax.set_ylim(-80, 90); ax.grid(alpha=.3); ax.legend(loc='lower left', markerscale=2)
ax.set_title('DBSCAN clusters of the top-10% concentrations (comparable subset)')
plt.tight_layout(); plt.savefig('maps/map04_dbscan_high_concentration.png'); plt.close()

# ---------- India / Indian Ocean regional map (box = coordinates, NOT a country boundary)
b = df[(df[LAT].between(5, 25)) & (df[LON].between(66, 90))].copy()
P(f'\nIndia-region box (lat 5-25, lon 66-90) records: {len(b)}')
P(b.groupby(['analysis_group']).size().to_string()); P(b['Organization'].value_counts().to_string())
fig, ax = plt.subplots(figsize=(7, 6)); mk = {'water_pieces_per_m3': 'o', 'sediment_pieces_per_kg_dw': 's', 'beach_pieces_per_m3': '^', 'nurdle_pieces_per_10min': 'D'}
for g, m in mk.items():
    s = b[b.analysis_group == g]
    if len(s): sc = ax.scatter(s[LON], s[LAT], c=np.log10(s['value'] + 1e-3), marker=m, s=22, cmap='viridis', edgecolors='k', linewidths=.3, label=f'{g} (n={len(s)})', vmin=-3, vmax=4)
plt.colorbar(sc, label='log10(value + 1e-3), unit differs by marker group'); ax.legend(fontsize=6); ax.grid(alpha=.3)
ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude'); ax.set_title('India region observations (box 5-25N, 66-90E): coordinates only, not an official boundary')
plt.tight_layout(); plt.savefig('maps/map05_india_region.png'); plt.close()
open('reports/04_hotspot_numbers.txt', 'w').write('\n'.join(txt))
