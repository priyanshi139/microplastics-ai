"""Phase 3: exploratory data analysis on the cleaned real NOAA data. Units are never mixed:
every concentration plot is drawn per analysis_group."""
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.dpi': 150, 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
df = pd.read_csv('data/cleaned/NOAA_microplastics_cleaned_all.csv', low_memory=False)
G = {'water_pieces_per_m3': 'Ocean water (pieces/m$^3$)', 'beach_pieces_per_m3': 'Beach (pieces/m$^3$)',
     'sediment_pieces_per_kg_dw': 'Sediment (pieces/kg d.w.)', 'nurdle_pieces_per_10min': 'Nurdle Patrol (pieces/10 min)'}
txt = []
def save(name): plt.tight_layout(); plt.savefig(f'figures/{name}.png'); plt.close()

# 1. Distribution per group (positive values, log10) + zero share
fig, ax = plt.subplots(1, 4, figsize=(13, 3))
for a, (g, lab) in zip(ax, G.items()):
    v = df.loc[df.analysis_group == g, 'value']
    a.hist(np.log10(v[v > 0]), bins=40, color='#2b6c8f'); a.set_title(lab, fontsize=8)
    a.set_xlabel('log10(value), zeros excluded'); a.set_ylabel('records')
    txt.append(f'{g}: n={len(v)}, zeros={(v==0).sum()} ({(v==0).mean()*100:.1f}%), median={v.median():.4g}, p90={v.quantile(.9):.4g}, max={v.max():.4g}')
save('fig01_value_distribution_by_group')

# 2. Records by ocean and group
ct = pd.crosstab(df['Ocean'].fillna('Not recorded'), df['analysis_group'])
ct.plot(kind='bar', stacked=True, figsize=(7, 3.5), colormap='viridis'); plt.ylabel('records'); plt.xticks(rotation=30)
save('fig02_records_by_ocean'); txt.append('\nOcean x group:\n' + ct.to_string())

# 3. Country distribution (records with country label only)
c = df['Country_clean'].value_counts().head(15)
c.sort_values().plot(kind='barh', figsize=(6, 4), color='#2b6c8f'); plt.xlabel('records (Country labelled records only)')
save('fig03_records_by_country'); txt.append('\nCountry label coverage: %.1f%% of records' % (df['Country_clean'].notna().mean() * 100))

# 4. Median water concentration by country (n>=30), sample size shown
w = df[df.analysis_group == 'water_pieces_per_m3']
s = w.groupby('Country_clean')['value'].agg(['count', 'median']).query('count>=30').sort_values('median')
if len(s):
    plt.figure(figsize=(6, 3.5)); plt.barh(s.index, s['median'], color='#c0603a'); plt.xscale('symlog', linthresh=1e-3)
    plt.xlabel('median pieces/m$^3$ (ocean water, n>=30)'); save('fig04_water_median_by_country')
    txt.append('\nWater median by country (n>=30):\n' + s.to_string())

# 5. Temporal: records and median per year, water only
y = w.groupby('year')['value'].agg(['count', 'median', lambda x: x.quantile(.9)])
y.columns = ['n', 'median', 'p90']
fig, ax = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
ax[0].bar(y.index, y['n'], color='gray'); ax[0].set_ylabel('water records')
ax[1].semilogy(y.index, y['median'].replace(0, np.nan), 'o-', label='median'); ax[1].semilogy(y.index, y['p90'].replace(0, np.nan), 's-', label='p90')
ax[1].set_ylabel('pieces/m$^3$'); ax[1].legend(); save('fig05_temporal_water')
txt.append('\nWater per year:\n' + y.to_string())

# 6. Sampling method comparison (water, top methods n>=100)
m = w.groupby('Sampling Method')['value'].agg(['count', 'median', 'mean']).query('count>=100').sort_values('median')
top = m.index.tolist()
plt.figure(figsize=(7, 3.5))
plt.boxplot([np.log10(w.loc[w['Sampling Method'] == k, 'value'].clip(lower=1e-6)) for k in top], tick_labels=top, orientation="horizontal", showfliers=False)
plt.xlabel('log10(pieces/m$^3$) (zeros = -6)'); save('fig06_method_comparison_water')
txt.append('\nWater by sampling method (n>=100):\n' + m.to_string())

# 7. Unit / group distribution
df['analysis_group'].value_counts().plot(kind='bar', figsize=(5, 3), color='#2b6c8f'); plt.ylabel('records'); plt.xticks(rotation=20)
save('fig07_unit_group_distribution')

# 8. Concentration class by group
cc = pd.crosstab(df['analysis_group'], df['Concentration Class'])[['Very Low', 'Low', 'Medium', 'High', 'Very High']]
cc.div(cc.sum(1), axis=0).plot(kind='bar', stacked=True, figsize=(7, 3.5), colormap='RdYlBu_r'); plt.ylabel('share'); plt.xticks(rotation=20)
save('fig08_concentration_class_by_group'); txt.append('\nConcentration class x group:\n' + cc.to_string())

# 9. Missing data
miss = (df.drop(columns=[c for c in df.columns if c in ['x', 'y']]).isna().mean() * 100).sort_values(ascending=False)
miss = miss[miss > 0]
miss.sort_values().plot(kind='barh', figsize=(6, 6), color='#7a7a7a'); plt.xlabel('% missing'); save('fig09_missing_data')

# 10. Correlation (water only, numeric features that are not constants); Spearman because of skew
cols = ['log10_value', 'Latitude (degree)', 'Longitude (degree)', 'Water Sample Depth (m)', 'Mesh Size (mm)', 'year']
cm = w[cols].corr(method='spearman')
plt.figure(figsize=(5, 4)); plt.imshow(cm, cmap='coolwarm', vmin=-1, vmax=1)
plt.xticks(range(len(cols)), cols, rotation=45, ha='right'); plt.yticks(range(len(cols)), cols); plt.colorbar()
for i in range(len(cols)):
    for j in range(len(cols)): plt.text(j, i, f'{cm.iloc[i,j]:.2f}', ha='center', va='center', fontsize=7)
save('fig10_spearman_water'); txt.append('\nSpearman (water):\n' + cm.round(3).to_string())
open('reports/03_eda_numbers.txt', 'w').write('\n'.join(txt)); print('\n'.join(txt))
