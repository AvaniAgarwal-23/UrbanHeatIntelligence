"""
Spatial generalization tests: can the UHI model predict cities it has never seen?
1. Leave-cities-out: 6 city-grouped folds. Train on the 2016-2023 rows of the other cities,
   predict 2025 for the held-out cities. Predictions are pooled over folds.
2. Cross-region transfer: train on Indian cities only and test on international cities, and the reverse.
Baseline: predict the training mean. Same hyperparameters as src/models/train_regression.py.
Run: python -m src.models.generalization   (writes reports/generalization.json)
"""
import json, warnings
import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from src.utils.config import load_config
from src.features.engineering import FeatureEngineer
warnings.filterwarnings("ignore")

cfg = load_config(); fe = FeatureEngineer(cfg)
def load(p):
    d = pd.read_parquet(p); d = d[d["uhi"].notna()].copy()
    X, y, cols = fe.get_feature_matrix_and_target(d); return d, X, y
dtr, Xtr, ytr = load("data/processed/splits/train_2016_2023.parquet")
dte, Xte, yte = load("data/processed/splits/test_2025.parquet")
canon = {k.lower().replace(" ", ""): k for k in cfg["cities"]}
def city_key(s): return canon.get(str(s).lower().replace(" ", ""), str(s))
ctr = dtr["city"].map(city_key).values; cte = dte["city"].map(city_key).values
region = {k: v["region"] for k, v in cfg["cities"].items()}

def models():
    return {
      "Ridge": Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler()), ("m", Ridge(alpha=10.0, random_state=42))]),
      "Random Forest": Pipeline([("i", SimpleImputer(strategy="median")), ("m", RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=4, random_state=42, n_jobs=-1))]),
      "HistGradientBoosting": Pipeline([("i", SimpleImputer(strategy="median")), ("m", HistGradientBoostingRegressor(max_iter=150, max_depth=8, learning_rate=0.08, random_state=42))]),
    }
def metr(y, p): return {"rmse": float(root_mean_squared_error(y, p)), "mae": float(mean_absolute_error(y, p)), "r2": float(r2_score(y, p))}

cities = sorted(set(cte)); rng = np.random.default_rng(42); perm = list(rng.permutation(cities))
folds = [perm[i::6] for i in range(6)]
res = {"n_cities": len(cities), "folds": [list(map(str, f)) for f in folds]}
pooled = {n: np.full(len(yte), np.nan) for n in list(models()) + ["Mean baseline"]}
for f in folds:
    te_m = np.isin(cte, f); tr_m = ~np.isin(ctr, f)
    for n, m in models().items():
        m.fit(Xtr[tr_m], ytr[tr_m]); pooled[n][te_m] = m.predict(Xte[te_m])
    pooled["Mean baseline"][te_m] = ytr[tr_m].mean()
res["leave_cities_out"] = {n: metr(yte, p) for n, p in pooled.items()}
# per-city R2/RMSE for the champion, to show where it fails
pc = []
for c in cities:
    mk = cte == c
    pc.append({"city": str(c), "rmse": float(root_mean_squared_error(yte[mk], pooled["HistGradientBoosting"][mk])), "n": int(mk.sum())})
pc.sort(key=lambda d: d["rmse"]); res["per_city_rmse_hgb"] = {"best5": pc[:5], "worst5": pc[-5:]}
res["n_cities_hgb_beats_mean"] = int(sum(
    root_mean_squared_error(yte[cte == c], pooled["HistGradientBoosting"][cte == c]) < root_mean_squared_error(yte[cte == c], pooled["Mean baseline"][cte == c]) for c in cities))
# cross region transfer
ri = np.array([region.get(c) == "Indian" for c in ctr]); rt = np.array([region.get(c) == "Indian" for c in cte])
xr = {}
for name, trm, tem in (("India to International", ri, ~rt), ("International to India", ~ri, rt)):
    xr[name] = {"n_test": int(tem.sum())}
    for n, m in models().items():
        m.fit(Xtr[trm], ytr[trm]); xr[name][n] = metr(yte[tem], m.predict(Xte[tem]))
    xr[name]["Mean baseline"] = metr(yte[tem], np.full(tem.sum(), ytr[trm].mean()))
res["cross_region"] = xr
json.dump(res, open("reports/generalization.json", "w"), indent=1)
print(json.dumps(res, indent=1))
