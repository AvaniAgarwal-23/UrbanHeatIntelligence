"""
Quantitative evaluation of SHAP explanations for the champion UHI model (2025 test set).
Checks: additivity, deletion fidelity, agreement with permutation importance,
stability (2024 vs 2025, India vs international), grouped attribution, retraining ablation.
Run: python -m src.models.shap_fidelity   (writes reports/shap_fidelity.json)
"""
import json, warnings, numpy as np, pandas as pd, joblib, shap
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import root_mean_squared_error, r2_score
warnings.filterwarnings("ignore")
from src.utils.config import load_config
from src.features.engineering import FeatureEngineer
cfg=load_config(); fe=FeatureEngineer(cfg)
pipe=joblib.load("models/champion_uhi_model.joblib")
imp=pipe.named_steps["imputer"]; reg=pipe.named_steps["regressor"]
print("PARAMS", {k:v for k,v in reg.get_params().items() if k in("max_iter","max_depth","learning_rate","l2_regularization","early_stopping","min_samples_leaf")})
def load(p):
    d=pd.read_parquet(p); d=d[d["uhi"].notna()].copy()
    X,y,cols=fe.get_feature_matrix_and_target(d); return d,X,y,cols
dtr,Xtr,ytr,cols=load("data/processed/splits/train_2016_2023.parquet")
dva,Xva,yva,_=load("data/processed/splits/val_2024.parquet")
dte,Xte,yte,_=load("data/processed/splits/test_2025.parquet")
print("N feats",len(cols),"train/val/test",len(Xtr),len(Xva),len(Xte))
I=lambda X: pd.DataFrame(imp.transform(X),columns=cols,index=X.index)
Xtr_i,Xva_i,Xte_i=I(Xtr),I(Xva),I(Xte)
pred=reg.predict(Xte_i); base_rmse=root_mean_squared_error(yte,pred)
print("test RMSE",base_rmse,"R2",r2_score(yte,pred))
ex=shap.TreeExplainer(reg)
S=ex.shap_values(Xte_i); Sva=ex.shap_values(Xva_i)
ev=float(np.ravel(ex.expected_value)[0])
add=np.abs(ev+S.sum(1)-pred)
out={"n_test":len(Xte),"n_features":len(cols),"test_rmse":base_rmse}
out["additivity"]={"mean_abs_err":float(add.mean()),"max_abs_err":float(add.max()),"expected_value":ev}
imp_s=pd.Series(np.abs(S).mean(0),index=cols).sort_values(ascending=False)
imp_v=pd.Series(np.abs(Sva).mean(0),index=cols)
out["top10"]=imp_s.head(10).round(4).to_dict()
# temporal stability 2024 vs 2025
out["stability_2024_vs_2025"]={"spearman":float(spearmanr(imp_s.reindex(cols),imp_v)[0]),
 "top10_overlap":len(set(imp_s.head(10).index)&set(imp_v.sort_values(ascending=False).head(10).index))}
# India vs international
reg_col="region" if "region" in dte.columns else None
ind=set(k for k,v in cfg["cities"].items() if v["region"]=="Indian")
cn=dte["city"].astype(str).str.lower().values
indmask=np.array([c in {x.lower() for x in ind} for c in cn])
if indmask.sum()>10 and (~indmask).sum()>10:
    a=pd.Series(np.abs(S[indmask]).mean(0),index=cols); b=pd.Series(np.abs(S[~indmask]).mean(0),index=cols)
    out["stability_india_vs_intl"]={"n_india":int(indmask.sum()),"n_intl":int((~indmask).sum()),"spearman":float(spearmanr(a,b)[0]),
     "top10_overlap":len(set(a.sort_values(ascending=False).head(10).index)&set(b.sort_values(ascending=False).head(10).index)),
     "india_top5":list(a.sort_values(ascending=False).head(5).index),"intl_top5":list(b.sort_values(ascending=False).head(5).index)}
else: out["stability_india_vs_intl"]="city name match failed: %s"%list(set(cn))[:5]
# permutation importance agreement
pi=permutation_importance(reg,Xte_i,yte,scoring="neg_root_mean_squared_error",n_repeats=10,random_state=42)
pim=pd.Series(pi.importances_mean,index=cols)
out["perm_vs_shap"]={"spearman":float(spearmanr(imp_s.reindex(cols),pim)[0]),
 "top10_overlap":len(set(imp_s.head(10).index)&set(pim.sort_values(ascending=False).head(10).index)),
 "perm_top10":pim.sort_values(ascending=False).head(10).round(4).to_dict()}
# deletion fidelity
med=Xtr_i.median(); rng=np.random.default_rng(42)
order=list(imp_s.index); res={}
def drop(feats):
    X=Xte_i.copy()
    for f in feats: X[f]=med[f]
    p=reg.predict(X); return root_mean_squared_error(yte,p), float(np.mean(np.abs(p-pred)))
for k in (1,3,5,10):
    top=drop(order[:k]); bot=drop(order[-k:])
    rr=[drop(list(rng.choice(cols,k,replace=False))) for _ in range(30)]
    res[k]={"top_k_rmse":top[0],"top_k_pred_shift":top[1],"bottom_k_rmse":bot[0],"random_k_rmse":float(np.mean([r[0] for r in rr])),"random_k_pred_shift":float(np.mean([r[1] for r in rr]))}
out["deletion"]=res
# grouped attribution
def grp(f):
    fl=f.lower()
    if "pixel_count" in fl or "landsat_obs" in fl or "population_epoch" in fl: return "data coverage"
    if fl in("latitude","longitude"): return "location"
    if fl.startswith("month_"): return "season"
    if any(t in fl for t in("ndvi","ndbi","ndwi","built_veg","veg_moisture")): return "surface indices"
    if "pop" in fl: return "population"
    return "weather"
g=pd.Series({f:grp(f) for f in cols}); tot=imp_s.sum()
out["group_share"]=(imp_s.groupby(g.reindex(imp_s.index)).sum()/tot).round(4).to_dict()
# retrain ablation
P=dict(max_iter=reg.get_params()["max_iter"],max_depth=reg.get_params()["max_depth"],learning_rate=reg.get_params()["learning_rate"],random_state=42)
def fit_eval(drop_groups):
    keep=[c for c in cols if g[c] not in drop_groups]
    m=HistGradientBoostingRegressor(**P).fit(Xtr_i[keep],ytr); p=m.predict(Xte_i[keep])
    pv=m.predict(Xva_i[keep])
    return {"n_feat":len(keep),"test_rmse":root_mean_squared_error(yte,p),"test_r2":r2_score(yte,p),"val_rmse":root_mean_squared_error(yva,pv)}
out["ablation"]={"all":fit_eval(set()),"no_coverage":fit_eval({"data coverage"}),"no_location":fit_eval({"location"}),
 "no_coverage_no_location":fit_eval({"data coverage","location"}),"no_weather":fit_eval({"weather"}),"no_surface":fit_eval({"surface indices"})}
json.dump(out,open("reports/shap_fidelity.json","w"),indent=1,default=float)
print(json.dumps(out,indent=1,default=float))
