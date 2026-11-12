"""Phase 5: ML experiments on REAL ocean-water data only (pieces/m3). Units never mixed.
Target: log10(value + 1e-3).  Features: lat, lon, year, month, sampling method, mesh size, water sample depth.
No environmental covariates (temperature, currents, population...) exist in this file.
Leakage control: all imputation / encoding / scaling lives inside a Pipeline fitted on the training fold only.
Validation: (A) random 5-fold KFold  vs  (B) spatial block GroupKFold (10x10 degree blocks, 5 folds).
Baselines: predict the training-fold mean.  XGBoost/SHAP are in scripts/05b_colab_xgboost_shap.py (not installed here)."""
import pandas as pd, numpy as np, matplotlib, json
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
SEED = 42; OFF = 1e-3
df = pd.read_csv('data/cleaned/NOAA_water_pieces_per_m3.csv', low_memory=False)
LAT, LON = 'Latitude (degree)', 'Longitude (degree)'
need = [LAT, LON, 'year', 'month', 'Sampling Method', 'Mesh Size (mm)', 'Water Sample Depth (m)', 'value']
for c in need: assert c in df.columns, c
d = df.dropna(subset=[LAT, LON, 'year', 'value']).copy()
d['y'] = np.log10(d['value'] + OFF)
d['block'] = (np.floor(d[LAT] / 10).astype(int).astype(str) + '_' + np.floor(d[LON] / 10).astype(int).astype(str))
print('records used:', len(d), '| blocks:', d.block.nunique(), '| zeros:', (d.value == 0).mean().round(3))
num_all = [LAT, LON, 'year', 'month', 'Mesh Size (mm)', 'Water Sample Depth (m)']; cat = ['Sampling Method']
FEATSETS = {'all_features': (num_all, cat),
            'method_only (no location)': (['year', 'month', 'Mesh Size (mm)', 'Water Sample Depth (m)'], cat),
            'location_only (no method)': ([LAT, LON, 'year', 'month'], [])}
def pipe(model, num, cat, scale=False):
    nt = [('imp', SimpleImputer(strategy='median'))] + ([('sc', StandardScaler())] if scale else [])
    tr = [('num', Pipeline(nt), num)] + ([('cat', OneHotEncoder(handle_unknown='ignore', min_frequency=20), cat)] if cat else [])
    return Pipeline([('pre', ColumnTransformer(tr, sparse_threshold=0)), ('m', model)])
MODELS = {'Mean baseline': lambda n, c: pipe(DummyRegressor(), n, c),
          'Linear Regression': lambda n, c: pipe(LinearRegression(), n, c, scale=True),
          'Random Forest': lambda n, c: pipe(RandomForestRegressor(n_estimators=200, min_samples_leaf=3, n_jobs=-1, random_state=SEED), n, c),
          'HistGradientBoosting': lambda n, c: pipe(HistGradientBoostingRegressor(max_iter=300, learning_rate=.06, random_state=SEED), n, c)}
SPLITS = {'random 5-fold': (KFold(5, shuffle=True, random_state=SEED), None), 'spatial block 5-fold': (GroupKFold(5), d['block'])}
rows = []; oof_store = {}
for fs, (num, c) in FEATSETS.items():
    X = d[num + c]
    for sn, (sp, grp) in SPLITS.items():
        for mn, mk in MODELS.items():
            oof = np.zeros(len(d)); fold_r2 = []
            for tr, te in sp.split(X, d['y'], grp):
                m = mk(num, c).fit(X.iloc[tr], d['y'].iloc[tr]); oof[te] = m.predict(X.iloc[te])
                fold_r2.append(r2_score(d['y'].iloc[te], oof[te]))
            rows.append(dict(features=fs, validation=sn, model=mn, MAE=mean_absolute_error(d.y, oof), RMSE=mean_squared_error(d.y, oof) ** .5,
                             R2=r2_score(d.y, oof), R2_fold_min=min(fold_r2), R2_fold_max=max(fold_r2)))
            oof_store[(fs, sn, mn)] = oof
            print(rows[-1])
res = pd.DataFrame(rows).round(3); res.to_csv('reports/05_model_comparison.csv', index=False)
print(res.to_string(index=False))

# Plot: random vs spatial R2, all_features
a = res[res.features == 'all_features'].pivot(index='model', columns='validation', values='R2')
a.plot(kind='bar', figsize=(7, 3.5), color=['#c0603a', '#2b6c8f']); plt.axhline(0, c='k', lw=.6); plt.ylabel('R$^2$ (out-of-fold, log10 scale)'); plt.xticks(rotation=20)
plt.title('Random vs spatial-block validation (all features)'); plt.tight_layout(); plt.savefig('figures/fig11_random_vs_spatial_r2.png', dpi=150); plt.close()
# Plot: observed vs predicted, RF spatial
best = 'Random Forest'; o = oof_store[('all_features', 'spatial block 5-fold', best)]
fig, ax = plt.subplots(1, 2, figsize=(9, 4), sharex=True, sharey=True)
for a_, (sn, t) in zip(ax, [('random 5-fold', 'Random split'), ('spatial block 5-fold', 'Spatial block split')]):
    oo = oof_store[('all_features', sn, best)]; a_.scatter(d.y, oo, s=2, alpha=.3); a_.plot([-3, 5], [-3, 5], 'r--', lw=1); a_.set_title(f'RF, {t}'); a_.set_xlabel('observed log10(pieces/m$^3$+1e-3)')
ax[0].set_ylabel('predicted'); plt.tight_layout(); plt.savefig('figures/fig12_rf_observed_vs_predicted.png', dpi=150); plt.close()

# Permutation importance of RF on held-out spatial folds (all features)
num, c = FEATSETS['all_features']; X = d[num + c]; imp = []
for tr, te in GroupKFold(5).split(X, d.y, d.block):
    m = MODELS['Random Forest'](num, c).fit(X.iloc[tr], d.y.iloc[tr])
    pi = permutation_importance(m, X.iloc[te], d.y.iloc[te], n_repeats=5, random_state=SEED, scoring='r2'); imp.append(pi.importances_mean)
imp = pd.Series(np.mean(imp, axis=0), index=num + c).sort_values(); imp.to_csv('reports/05_rf_permutation_importance_spatial.csv')
imp.plot(kind='barh', figsize=(6, 3.2), color='#2b6c8f'); plt.xlabel('mean drop in R$^2$ when shuffled (held-out spatial folds)'); plt.tight_layout(); plt.savefig('figures/fig13_rf_permutation_importance.png', dpi=150); plt.close()
print('\nPermutation importance (held-out spatial folds):\n', imp.round(4).to_string())
