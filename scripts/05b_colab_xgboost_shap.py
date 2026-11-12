"""NOT EXECUTED in the authoring environment (xgboost/shap not installed there). Run in Google Colab.
!pip install xgboost shap
Upload data/cleaned/NOAA_water_pieces_per_m3.csv to the Colab session first.
Same target, features, and spatial block CV as 05_ml_experiments.py."""
import pandas as pd, numpy as np, xgboost as xgb, shap, matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
df = pd.read_csv('NOAA_water_pieces_per_m3.csv', low_memory=False)
LAT, LON = 'Latitude (degree)', 'Longitude (degree)'
d = df.dropna(subset=[LAT, LON, 'year', 'value']).copy(); d['y'] = np.log10(d['value'] + 1e-3)
d['block'] = np.floor(d[LAT] / 10).astype(int).astype(str) + '_' + np.floor(d[LON] / 10).astype(int).astype(str)
num = [LAT, LON, 'year', 'month', 'Mesh Size (mm)', 'Water Sample Depth (m)']; cat = ['Sampling Method']
def make(): 
    pre = ColumnTransformer([('num', SimpleImputer(strategy='median'), num), ('cat', OneHotEncoder(handle_unknown='ignore', min_frequency=20), cat)], sparse_threshold=0)
    return Pipeline([('pre', pre), ('m', xgb.XGBRegressor(n_estimators=400, learning_rate=.05, max_depth=6, subsample=.8, colsample_bytree=.8, random_state=42, n_jobs=-1))])
X = d[num + cat]
for name, sp, g in [('random', KFold(5, shuffle=True, random_state=42), None), ('spatial block', GroupKFold(5), d.block)]:
    oof = np.zeros(len(d))
    for tr, te in sp.split(X, d.y, g): oof[te] = make().fit(X.iloc[tr], d.y.iloc[tr]).predict(X.iloc[te])
    print(name, 'MAE %.3f RMSE %.3f R2 %.3f' % (mean_absolute_error(d.y, oof), mean_squared_error(d.y, oof) ** .5, r2_score(d.y, oof)))
# SHAP on one spatial held-out fold (explains the model; it is NOT causal evidence)
tr, te = next(GroupKFold(5).split(X, d.y, d.block)); m = make().fit(X.iloc[tr], d.y.iloc[tr])
Xt = m.named_steps['pre'].transform(X.iloc[te]); names = m.named_steps['pre'].get_feature_names_out()
sv = shap.TreeExplainer(m.named_steps['m']).shap_values(Xt); shap.summary_plot(sv, Xt, feature_names=names, show=False); plt.tight_layout(); plt.savefig('shap_summary.png', dpi=150)
