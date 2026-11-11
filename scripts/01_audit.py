"""Phase 1: audit of the real NOAA NCEI marine microplastics export. Nothing is modified."""
import pandas as pd, numpy as np, os, hashlib
P = 'data/raw/NOAA_Marine_Microplastics_original.csv'
df = pd.read_csv(P, low_memory=False, encoding='utf-8-sig')
out = []
def w(*a):
    s = ' '.join(str(x) for x in a); print(s); out.append(s)

w('FILE SIZE (bytes):', os.path.getsize(P), '| SHA256:', hashlib.sha256(open(P, 'rb').read()).hexdigest())
w('SHAPE:', df.shape)
w('\n== MISSING VALUES ==')
m = pd.DataFrame({'dtype': df.dtypes.astype(str), 'n_missing': df.isna().sum(),
                  'pct_missing': (df.isna().mean() * 100).round(2), 'n_unique': df.nunique()})
w(m.to_string())
w('\n== DUPLICATES ==')
w('fully identical rows:', df.duplicated().sum())
w('duplicate Unique ID:', df['Unique ID'].duplicated().sum())
w('duplicate GlobalID:', df['GlobalID'].duplicated().sum())
w('duplicate ObjectId:', df['ObjectId'].duplicated().sum())
cols = [c for c in df.columns if c not in ['ObjectId', 'GlobalID', 'Unique ID', 'x', 'y']]
w('duplicates ignoring IDs and x/y:', df.duplicated(subset=cols).sum())
w('\n== DATES ==')
d = pd.to_datetime(df['Sample Date'], errors='coerce', format='%m/%d/%Y %I:%M:%S %p')
w('unparsed:', d.isna().sum(), 'of', len(d), '| raw missing:', df['Sample Date'].isna().sum())
w('range:', d.min(), 'to', d.max())
w('records per year:\n', d.dt.year.value_counts().sort_index().to_string())
w('\n== COORDINATES ==')
w('lat range', df['Latitude (degree)'].min(), df['Latitude (degree)'].max(),
  '| lon range', df['Longitude (degree)'].min(), df['Longitude (degree)'].max())
w('invalid lat:', (df['Latitude (degree)'].abs() > 90).sum(), 'invalid lon:', (df['Longitude (degree)'].abs() > 180).sum())
w('(0,0) points:', ((df['Latitude (degree)'] == 0) & (df['Longitude (degree)'] == 0)).sum())
w('\n== CATEGORICALS ==')
for c in ['Ocean', 'Marine Setting', 'Sampling Method', 'Unit', 'Concentration Class',
          'Concentration Class Range', 'Study Type', 'Organization']:
    w('\n--', c); w(df[c].value_counts(dropna=False).head(40).to_string())
w('\n-- Country (top 40)'); w(df['Country'].value_counts(dropna=False).head(40).to_string())
w('\n-- Region (top 25)'); w(df['Region'].value_counts(dropna=False).head(25).to_string())
w('\n== UNIT x MARINE SETTING =='); w(pd.crosstab(df['Marine Setting'], df['Unit'], dropna=False).to_string())
w('\n== UNIT x SAMPLING METHOD =='); w(pd.crosstab(df['Sampling Method'], df['Unit'], dropna=False).to_string())
w('\n== MEASUREMENT ==')
x = df['Microplastics Measurement']
w(x.describe().to_string()); w('zeros:', (x == 0).sum(), 'negatives:', (x < 0).sum(), 'missing:', x.isna().sum())
w('by unit:'); w(df.groupby('Unit', dropna=False)['Microplastics Measurement'].describe().to_string())
w('\n== NUMERIC COLUMNS =='); w(df.select_dtypes('number').describe().T.to_string())
w('\n== ACCESSIONS ==')
w('unique NCEI accessions:', df['NCEI Accession Number'].nunique(), '| unique DOIs:', df['DOI'].nunique())
open('reports/01_dataset_audit_raw_output.txt', 'w').write('\n'.join(out))
