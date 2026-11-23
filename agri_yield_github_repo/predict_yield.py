"""
Score a table of ENGINEERED features with the saved yield model.

    python predict_yield.py features.csv out.csv

`features.csv` must contain the columns listed in bundle["features"] (see the notebook, section 4, for how each is built from
raw tables, strictly from information available at the forecast cutoff) plus `vc_roll3` (3-year village-crop mean yield),
and `district` / `crop` if the recalibration factors are to be applied.

Output: forecast yield in quintal/ha per row (plus a shortfall score if `bc_trail3` is present).
NOTE: the model was trained on SYNTHETIC data. It demonstrates the pipeline; retrain on real records before any operational use.
"""
import sys, joblib, numpy as np, pandas as pd

def load(path="outputs/yield_model.joblib"):
    return joblib.load(path)

def predict(bundle, df):
    X = df[bundle["features"]]
    log_ratio = bundle["model"].predict(X)                       # model predicts log(yield / 3-yr village-crop mean)
    yhat = df[bundle["ref_column"]].values * np.exp(log_ratio)
    if bundle.get("recalibration"):
        fac = {(r["district"], r["crop"]): r["factor"] for r in bundle["recalibration"]}
        yhat = yhat * np.array([fac.get((d, c), 1.0) for d, c in zip(df["district"], df["crop"])])
    return yhat

if __name__ == "__main__":
    b = load()
    d = pd.read_csv(sys.argv[1])
    d["forecast_qtl_ha"] = predict(b, d)
    if "bc_trail3" in d:
        d["shortfall_score"] = 1 - d.groupby(["block", "crop", "year"]).forecast_qtl_ha.transform("mean") / d.bc_trail3
        d["shortfall_flag"] = d.shortfall_score >= b["shortfall_threshold"] if b["shortfall_threshold"] is not None else np.nan
    d.to_csv(sys.argv[2], index=False)
    print("wrote", sys.argv[2])
