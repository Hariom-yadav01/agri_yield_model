"""
Synthetic data generator - Nagpur-region agri-tech yield project.
ALL DATA IS SIMULATED. Seed is fixed once (no seed shopping).
Writes raw, deliberately messy tables to ./data:
  yields.csv     crop-cutting-experiment (CCE) level yields   (~300k rows)
  rainfall.csv   block x month rainfall (missing monsoon readings, -999 sentinels)
  soil.csv       village soil-test records (irregular, incomplete)
  mandi.csv      market x crop x month price (Rs/quintal and Rs/kg in ONE column)
  area_stats.csv district x crop x year sown area
  cost_index.csv crop x year cost of cultivation (Rs/ha)
  *_truth.csv    ground truth, used ONLY to audit the cleaning code, never as model input
"""
import numpy as np, pandas as pd, os

SEED = 20261119
rng = np.random.default_rng(SEED)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUT, exist_ok=True)

YEARS = np.arange(2005, 2025)
NY = len(YEARS)
DIST = ["Nagpur", "Wardha", "Amravati", "Yavatmal", "Chandrapur", "Bhandara"]
CODE = dict(zip(DIST, ["NAG", "WAR", "AMR", "YVT", "CHD", "BHN"]))
RAIN_MULT = dict(Nagpur=1.0, Wardha=0.97, Amravati=0.90, Yavatmal=0.95, Chandrapur=1.12, Bhandara=1.15)
CROPS = ["cotton", "soybean", "tur", "rice", "wheat", "gram"]
SEASON = dict(cotton="kharif", soybean="kharif", tur="kharif", rice="kharif", wheat="rabi", gram="rabi")

# crop parameters ---------------------------------------------------------------------------
CP = {  #        base  trend  Ropt  sens  ph_opt  sow_opt  irrig_p  base_N  cost/ha
    "cotton":  dict(base=8.5, trend=.012, ropt=750, sens=1.00, ph=7.5, sow=25, irr=.12, N=100, cost=38000),
    "soybean": dict(base=10., trend=.010, ropt=760, sens=1.15, ph=6.8, sow=25, irr=.04, N=25,  cost=22000),
    "tur":     dict(base=6.5, trend=.012, ropt=720, sens=0.80, ph=7.0, sow=26, irr=.03, N=20,  cost=20000),
    "rice":    dict(base=20., trend=.012, ropt=950, sens=0.90, ph=6.5, sow=27, irr=.45, N=100, cost=42000),
    "wheat":   dict(base=22., trend=.014, ropt=780, sens=0.85, ph=7.0, sow=45, irr=.70, N=110, cost=30000),
    "gram":    dict(base=9.0, trend=.012, ropt=780, sens=0.65, ph=7.2, sow=43, irr=.15, N=20,  cost=20000),
}
GROW_P = {
    "Nagpur":     dict(cotton=.80, soybean=.60, tur=.50, rice=.20, wheat=.60, gram=.60),
    "Wardha":     dict(cotton=.90, soybean=.70, tur=.60, rice=.05, wheat=.50, gram=.50),
    "Amravati":   dict(cotton=.80, soybean=.80, tur=.60, rice=.02, wheat=.50, gram=.60),
    "Yavatmal":   dict(cotton=.95, soybean=.60, tur=.80, rice=.03, wheat=.30, gram=.40),
    "Chandrapur": dict(cotton=.60, soybean=.40, tur=.30, rice=.80, wheat=.20, gram=.20),
    "Bhandara":   dict(cotton=.02, soybean=.10, tur=.30, rice=.98, wheat=.40, gram=.30),
}

# geography ---------------------------------------------------------------------------------
blocks = pd.DataFrame([(d, f"{CODE[d]}-B{b:02d}") for d in DIST for b in range(1, 11)], columns=["district", "block"])
vill = blocks.loc[blocks.index.repeat(12)].reset_index(drop=True)
vill["village_id"] = vill["block"] + "-V" + (vill.groupby("block").cumcount() + 1).astype(str).str.zfill(2)
nv = len(vill)
blk_shift = {b: rng.normal(0, .06) for b in blocks.block}
vill["oc0"] = np.clip(rng.normal(.55, .11, nv) + vill.block.map(blk_shift), .25, 1.1)
vill["ph0"] = np.where(vill.district == "Bhandara", rng.normal(6.3, .4, nv), rng.normal(7.6, .45, nv))
vill["n0"] = np.clip(230 + 140 * (vill.oc0 - .55) + rng.normal(0, 30, nv), 120, 420)
vill["p0"] = np.clip(rng.lognormal(np.log(14), .45, nv), 3, 60)
vill["k0"] = np.clip(rng.normal(320, 65, nv), 120, 560)
vill["ec0"] = np.clip(rng.lognormal(np.log(.35), .4, nv), .05, 2.5)
vill["vfe"] = rng.normal(0, .09, nv)                       # persistent village productivity
vill["irr_prop"] = rng.beta(2, 3, nv)

# which crops each village grows
rows = []
for i, v in vill.iterrows():
    p = GROW_P[v.district]
    chosen = [c for c in CROPS if rng.random() < p[c]]
    if len(chosen) < 3:
        extra = sorted([c for c in CROPS if c not in chosen], key=lambda c: -p[c])[: 3 - len(chosen)]
        chosen += extra
    rows += [(v.village_id, c) for c in chosen]
vc = pd.DataFrame(rows, columns=["village_id", "crop"]).merge(vill, on="village_id")

# weather (block x season-year x month) -----------------------------------------------------
MONTHS = ["jun", "jul", "aug", "sep", "oct", "nov", "dec", "jan", "feb"]
MNUM = [6, 7, 8, 9, 10, 11, 12, 1, 2]
CLIM = np.array([150, 270, 240, 160, 62, 16, 8, 10, 9.])
yf = np.exp(rng.normal(0, .13, NY))
for yr, f in {2009: .70, 2012: .82, 2014: .76, 2015: .72, 2018: .80, 2023: .80,
              2013: 1.20, 2019: 1.25, 2020: 1.15, 2022: 1.20, 2024: 1.05}.items():
    yf[yr - 2005] = f
dyf = {(d, y): np.exp(rng.normal(0, .10)) for d in DIST for y in YEARS}
w = []
for _, b in blocks.iterrows():
    for yi, y in enumerate(YEARS):
        byf = np.exp(rng.normal(0, .10))
        mu = CLIM * RAIN_MULT[b.district] * yf[yi] * dyf[(b.district, y)] * byf
        noise = rng.gamma(4, 1 / 4, 9)
        r = mu * noise
        if y == 2023:
            r[2] *= .50                     # very dry August 2023
        if y == 2022:
            r[1] *= 1.25                    # very wet July 2022
        rd = np.clip(np.round(1 + 2.2 * np.sqrt(r / 10) + rng.normal(0, 1.5, 9)), 0, 28)
        ds = np.clip(34 - 2.0 * (rd[1] + rd[2]) / 2 + rng.normal(0, 3), 3, 40)   # latent longest dry spell
        w.append(dict(district=b.district, block=b.block, year=y, **{f"r_{m}": r[k] for k, m in enumerate(MONTHS)},
                      **{f"rd_{m}": rd[k] for k, m in enumerate(MONTHS)}, dry=ds))
bw = pd.DataFrame(w)
bw["r_junsep"] = bw[["r_jun", "r_jul", "r_aug", "r_sep"]].sum(axis=1)
bw["r_octjan"] = bw[["r_oct", "r_nov", "r_dec", "r_jan"]].sum(axis=1)

# long rainfall table with missingness ------------------------------------------------------
lr, lt = [], []
for k, m in enumerate(MONTHS):
    t = bw[["district", "block", "year"]].copy()
    t["month"] = MNUM[k]
    t["cal_year"] = np.where(MNUM[k] <= 2, t.year + 1, t.year)
    t["rain_mm"] = bw[f"r_{m}"].round(1).values
    t["rainy_days"] = bw[f"rd_{m}"].values
    lt.append(t.copy())
lt = pd.concat(lt, ignore_index=True)
monsoon = lt.month.between(6, 9)
p_miss = np.where(monsoon, np.where(lt.rain_mm > 330, .32, .17), .04)
miss = rng.random(len(lt)) < p_miss
sentinel = miss & (rng.random(len(lt)) < .35)
rain = lt.copy()
rain["rain_mm"] = np.where(miss, np.nan, rain.rain_mm)
rain["rainy_days"] = np.where(miss, np.nan, rain.rainy_days)
rain.loc[sentinel, ["rain_mm", "rainy_days"]] = -999
rain_out = rain[["district", "block", "cal_year", "month", "rain_mm", "rainy_days"]].rename(columns={"cal_year": "year"})
spell = {"Nagpur": ["Nagpur", "NAGPUR", "nagpur "], "Amravati": ["Amravati", "Amaravati", "AMRAVATI"],
         "Yavatmal": ["Yavatmal", "Yeotmal", "YAVATMAL"], "Wardha": ["Wardha", "wardha", "WARDHA"],
         "Chandrapur": ["Chandrapur", "CHANDRAPUR", "Chandrapur "], "Bhandara": ["Bhandara", "BHANDARA", "bhandara"]}
def messy_dist(s):
    out = s.copy()
    for d, opts in spell.items():
        idx = s.index[(s == d)]
        pick = rng.choice(len(opts), size=len(idx), p=[.93, .04, .03])
        out.loc[idx] = np.array(opts)[pick]
    return out
rain_out["district"] = messy_dist(rain_out["district"]).values
rain_out.to_csv(f"{OUT}/rainfall.csv", index=False)
lt[["block", "year", "month", "rain_mm", "rainy_days"]].to_csv(f"{OUT}/rainfall_truth.csv", index=False)

# mandi base prices (before supply effect) --------------------------------------------------
PBASE = dict(cotton=2000, soybean=1300, tur=2200, rice=800, wheat=900, gram=1800)
PG = dict(cotton=.065, soybean=.065, tur=.065, rice=.06, wheat=.06, gram=.065)
SEAS_M = {  # monthly seasonal multiplier (harvest dip, lean premium)
    "kharif": np.array([1.06, 1.07, 1.08, 1.05, 1.00, .94, .94, .97, 1.00, 1.02, 1.04, 1.05]),
    "rabi":   np.array([1.05, 1.06, 1.06, 1.00, .95, .94, 1.0, 1.02, 1.03, 1.04, 1.05, 1.06]),
}
# month index 0 = Jan. kharif harvest Nov (idx 10 -> use roll to dip around Nov-Dec); rabi harvest Mar-Apr
SEAS_M["kharif"] = np.array([1.00, 1.02, 1.04, 1.05, 1.06, 1.07, 1.08, 1.06, 1.02, .97, .93, .94])
SEAS_M["rabi"] = np.array([1.04, 1.05, .96, .93, .96, 1.00, 1.03, 1.05, 1.06, 1.06, 1.05, 1.04])
NM = NY * 12
rw = {}
for c in CROPS:
    x = np.zeros(NM)
    for t in range(1, NM):
        x[t] = .97 * x[t - 1] + rng.normal(0, .02)
    rw[c] = x

# soil test table ---------------------------------------------------------------------------
srows = []
for i, v in vill.iterrows():
    if rng.random() < .08:
        continue                                             # never tested
    off = rng.integers(0, 3)
    for y in range(2005 + off, 2025, 3):
        if rng.random() < .20:
            continue
        d = y - 2005
        oc = v.oc0 + .003 * d + rng.normal(0, .03)
        rec = dict(village_id=v.village_id, test_year=y, ph=v.ph0 + rng.normal(0, .12),
                   oc_pct=oc, n_kgha=v.n0 + 1.2 * d + rng.normal(0, 12), p_kgha=v.p0 * (1 + .01 * d) * np.exp(rng.normal(0, .08)),
                   k_kgha=v.k0 + rng.normal(0, 15), ec=v.ec0 * np.exp(rng.normal(0, .1)))
        for col in ["ph", "oc_pct", "n_kgha", "p_kgha", "k_kgha", "ec"]:
            if rng.random() < .05:
                rec[col] = np.nan
        srows.append(rec)
soil = pd.DataFrame(srows).round(3)
soil.to_csv(f"{OUT}/soil.csv", index=False)

# yields ------------------------------------------------------------------------------------
vcy = vc.loc[vc.index.repeat(NY)].reset_index(drop=True)
vcy["year"] = np.tile(YEARS, len(vc))
vcy["n_cce"] = rng.integers(3, 10, len(vcy))
y = vcy.loc[vcy.index.repeat(vcy.n_cce)].reset_index(drop=True)
y = y.merge(bw[["block", "year", "r_junsep", "r_octjan", "r_jul", "r_aug", "r_jun", "dry"]], on=["block", "year"], how="left")
n = len(y)
y["season"] = y.crop.map(SEASON)
t = y.year.values - 2005

# plot attributes
y["area"] = np.clip(rng.lognormal(np.log(1.2), .75, n), .2, 12)
irr_p = y.crop.map({c: CP[c]["irr"] for c in CROPS}).values * (0.5 + y.irr_prop.values) * (1 + .08 * np.log(y.area.values))
y["irrigated"] = rng.random(n) < np.clip(irr_p, 0, .95)

# variety (12 per crop, introduced over time)
vinfo = {}
for c in CROPS:
    intro = np.sort(np.concatenate([[1995, 1998, 2001], rng.integers(2003, 2023, 9)]))
    eff = rng.normal(.02, .045, 12) + (intro - 2010) * .002
    vinfo[c] = (intro, eff)
y["variety"] = ""
y["v_eff"] = 0.0
y["v_intro"] = 0
for c in CROPS:
    intro, eff = vinfo[c]
    for yr in YEARS:
        idx = np.where((y.crop.values == c) & (y.year.values == yr))[0]
        avail = intro <= yr
        wts = np.where(avail, np.exp(-np.abs(yr - intro - 5) / 5) + .05, 0)
        pick = rng.choice(12, size=len(idx), p=wts / wts.sum())
        y.loc[idx, "variety"] = [f"{c[:3].upper()}-V{k + 1:02d}" for k in pick]
        y.loc[idx, "v_eff"] = eff[pick]
        y.loc[idx, "v_intro"] = intro[pick]

# sowing
sow_opt = y.crop.map({c: CP[c]["sow"] for c in CROPS}).values
delay = rng.poisson(1.2, n) + np.where(y.season == "kharif", (y.r_jun.values < 90) * rng.integers(1, 4, n), (y.r_junsep.values < 600) * rng.integers(0, 2, n))
y["sow_week"] = (sow_opt + delay - 1).astype(int)

# mandi: base price path (needed for pre-sowing price -> fertiliser decision)
mi = pd.MultiIndex.from_product([CROPS, range(NM)], names=["crop", "t"])
pb = pd.DataFrame(index=mi).reset_index()
pb["cal_year"] = 2005 + pb.t // 12
pb["m"] = pb.t % 12 + 1
pb["p0"] = [PBASE[c] * (1 + PG[c]) ** (tt / 12) * np.exp(rw[c][tt]) * SEAS_M[SEASON[c]][mm - 1] for c, tt, mm in zip(pb.crop, pb.t, pb.m)]
def base_ratio(c, yr):
    sm = 5 if SEASON[c] == "kharif" else 10        # pre-sowing month: May / Oct
    s = pb[(pb.crop == c)]
    cur = s[(s.cal_year == yr) & (s.m == sm)].p0.values[0]
    prev = s[(s.t < (yr - 2005) * 12 + sm - 1) & (s.t >= (yr - 2005) * 12 + sm - 37)].p0.mean() if yr > 2005 else cur
    return cur / prev
ratio = {(c, yr): base_ratio(c, yr) for c in CROPS for yr in YEARS}
y["pre_ratio"] = [ratio[(c, yr)] for c, yr in zip(y.crop, y.year)]
base_n = y.crop.map({c: CP[c]["N"] for c in CROPS}).values
fert = base_n * np.exp(rng.normal(0, .22, n)) * (1 + .6 * (y.pre_ratio.values - 1)) * (1 + .05 * np.log(y.area.values))
y["fert_n"] = fert

# true soil at the village-year (smooth), used in the yield function
d_ = y.year.values - 2005
oc = y.oc0.values + .003 * d_
ph = y.ph0.values
nn = y.n0.values + 1.2 * d_

# yield function ----------------------------------------------------------------------------
base = y.crop.map({c: CP[c]["base"] for c in CROPS}).values
trend = y.crop.map({c: CP[c]["trend"] for c in CROPS}).values
ropt = y.crop.map({c: CP[c]["ropt"] for c in CROPS}).values
sens = y.crop.map({c: CP[c]["sens"] for c in CROPS}).values
phopt = y.crop.map({c: CP[c]["ph"] for c in CROPS}).values
kh = (y.season.values == "kharif")
irr = y.irrigated.values
shield = np.where(irr, .22, 1.0)
r1 = y.r_junsep.values / ropt
r2 = y.r_octjan.values / 96
reff = np.where(kh, r1, .8 * r1 + .2 * np.minimum(r2, 2))
dr = np.maximum(0, .9 - reff)
f_water = 1 - np.clip(3.0 * sens * dr ** 1.3, 0, .6) * shield
f_excess = 1 - np.where(kh, .4 * np.maximum(0, r1 - 1.35), 0)
f_dry = np.where(kh, 1 - np.clip(y.crop.map({"cotton": .012, "soybean": .015, "tur": .008, "rice": .006, "wheat": 0, "gram": 0}).values
                                 * np.maximum(0, y.dry.values - 10), 0, .35) * np.where(irr, .4, 1), 1)
f_soil = 1 + .06 * (oc - .55) / .15 + .03 * (nn - 230) / 50 + .015 * (y.p0.values - 14) / 8 - .045 * np.abs(ph - phopt) / .5
f_n = 1 + .10 * np.tanh(1.2 * np.log(y.fert_n.values / base_n))
f_sow = 1 - np.where(kh, .012, .015) * np.maximum(0, y.sow_week.values - sow_opt - 1)
f_var = 1 + y.v_eff.values
f_irr = 1 + np.where(irr, np.where(kh, .06, .22), 0)
f_trend = (1 + trend) ** t
f_vill = np.exp(y.vfe.values)
# unobserved shocks
dc_sh = {(d, c, yr): rng.normal(0, .05) for d in DIST for c in CROPS for yr in YEARS}
bc_sh = {(b, c, yr): rng.normal(0, .05) for b in blocks.block for c in CROPS for yr in YEARS}
vc_sh = {}
sh_d = np.array([dc_sh[(d, c, yr)] for d, c, yr in zip(y.district, y.crop, y.year)])
sh_b = np.array([bc_sh[(b, c, yr)] for b, c, yr in zip(y.block, y.crop, y.year)])
key = y.village_id + "|" + y.crop + "|" + y.year.astype(str)
uk = key.unique()
vsh = dict(zip(uk, rng.normal(0, .06, len(uk))))
sh_v = key.map(vsh).values
event = np.zeros(n)
cot = y.crop.values == "cotton"
event += np.where(cot & (y.year.values == 2017), np.where(y.district.values == "Yavatmal", np.log(.78), np.log(.86)), 0)   # pink bollworm
event += np.where(cot & (y.year.values == 2018), np.log(.88), 0)
hail = (y.year.values == 2023) & (~kh) & np.isin(y.district.values, ["Amravati", "Yavatmal", "Wardha"])   # Feb-2023 hailstorm (after rabi cutoff)
vh = pd.Series(key).map(dict(zip(uk, rng.random(len(uk)) < .75))).values
event += np.where(hail & vh, np.log(.70), 0)
event += np.where((y.year.values == 2022) & (y.district.values == "Nagpur") & (y.crop.values == "soybean"), np.log(.90), 0)
event += np.where((y.year.values == 2024) & (y.district.values == "Bhandara") & (y.crop.values == "rice"), np.log(.90), 0)
resid = rng.normal(0, .10, n)
Y = (base * f_trend * f_water * f_excess * f_dry * f_soil * f_n * f_sow * f_var * f_irr * f_vill
     * np.exp(sh_d + sh_b + sh_v + event + resid))
Y = np.clip(Y, .8, None)
y["true_yield"] = Y

# post-harvest quantities that LEAK the target -----------------------------------------------
y["production_qtl"] = (Y * y.area.values).round(2)
exp_y = base * f_trend
y["insurance_claim_paid"] = (Y < .70 * exp_y) & (rng.random(n) < .8)

# unit errors ---------------------------------------------------------------------------------
u = rng.random(n)
etype = np.where(u < .035, "kg_ha", np.where(u < .055, "tonne_ha", np.where(u < .070, "qtl_acre", "ok")))
val = np.where(etype == "kg_ha", Y * 100, np.where(etype == "tonne_ha", Y / 10, np.where(etype == "qtl_acre", Y * .4047, Y)))
ok_lab = rng.choice(["qtl/ha", "Qtl/ha", "q/ha", "quintal/ha"], n, p=[.6, .2, .12, .08])
lab_true = {"kg_ha": ["kg/ha", "Kg/Ha"], "tonne_ha": ["t/ha", "tonnes/ha"], "qtl_acre": ["qtl/acre", "Qtl/acre"]}
lab = ok_lab.astype(object)
for e, opts in lab_true.items():
    m = etype == e
    k = rng.random(m.sum())
    chosen = np.where(k < .5, rng.choice(opts, m.sum()), np.where(k < .8, None, "qtl/ha"))
    lab[m] = chosen
lab[(etype == "ok") & (rng.random(n) < .03)] = None
y["yield_value"] = np.round(val, 3)
y["yield_unit"] = lab

# other columns' messiness ---------------------------------------------------------------------
y["irrigated_raw"] = np.where(y.irrigated, "Y", "N").astype(object)
y.loc[rng.random(n) < .03, "irrigated_raw"] = None
small = y.area.values < 1.5
y["fert_n_kgha"] = np.where(rng.random(n) < np.where(small, .30, .12), np.nan, y.fert_n.round(1))
y["sowing_week"] = np.where(rng.random(n) < .06, np.nan, y.sow_week)
y["district_raw"] = messy_dist(y["district"]).values
y["record_id"] = np.arange(1, n + 1)
cols = ["record_id", "district_raw", "block", "village_id", "crop", "season", "year", "variety", "area", "irrigated_raw",
        "sowing_week", "fert_n_kgha", "yield_value", "yield_unit", "production_qtl", "insurance_claim_paid"]
out = y[cols].rename(columns={"district_raw": "district", "area": "plot_area_ha", "irrigated_raw": "irrigated"})
out["insurance_claim_paid"] = np.where(out.insurance_claim_paid, "Y", "N")
dup = out.sample(frac=.003, random_state=1)
out = pd.concat([out, dup]).sample(frac=1, random_state=2).reset_index(drop=True)
out.to_csv(f"{OUT}/yields.csv", index=False)
pd.DataFrame({"record_id": y.record_id, "true_yield_qtl_ha": Y.round(3), "unit_error_type": etype}).to_csv(f"{OUT}/yields_truth.csv", index=False)

# mandi table (supply effect + messy units) -------------------------------------------------------
dcy = pd.DataFrame({"district": y.district, "crop": y.crop, "year": y.year, "ly": np.log(Y), "lexp": np.log(exp_y)}).groupby(["district", "crop", "year"]).mean()
anom = (dcy.ly - dcy.lexp).to_dict()
markets = [(d, f"{CODE[d]}-M{k}") for d in DIST for k in (1, 2, 3)]
mkt_prem = {m: rng.normal(0, .03) for _, m in markets}
mr = []
for d, m in markets:
    for c in CROPS:
        for tt in range(NM):
            if rng.random() < .10:
                continue
            cy, mo = 2005 + tt // 12, tt % 12 + 1
            if SEASON[c] == "kharif":
                Ys = cy if mo >= 11 else cy - 1
            else:
                Ys = cy - 1 if mo >= 3 else cy - 2
            a = anom.get((d, c, Ys), 0.0)
            p0 = pb.p0.values[CROPS.index(c) * NM + tt]
            price = p0 * np.exp(-.55 * a + mkt_prem[m] + rng.normal(0, .03))
            mr.append((d, m, c, cy, mo, price))
mandi = pd.DataFrame(mr, columns=["district", "market", "crop", "year", "month", "price_qtl"])
unit = rng.choice(["Rs/quintal", "Rs/kg"], len(mandi), p=[.80, .20])
mandi["modal_price"] = np.where(unit == "Rs/kg", mandi.price_qtl / 100, mandi.price_qtl).round(2)
lab_u = np.where(unit == "Rs/kg", rng.choice(["Rs/kg", "INR/kg", "Rs per kg"], len(mandi)), rng.choice(["Rs/quintal", "Rs/qtl", "INR/quintal"], len(mandi), p=[.6, .25, .15])).astype(object)
lab_u[rng.random(len(mandi)) < .12] = None
mandi["price_unit"] = lab_u
mandi_out = mandi[["district", "market", "crop", "year", "month", "modal_price", "price_unit"]].copy()
mandi_out["district"] = messy_dist(mandi_out["district"]).values
mandi_out.to_csv(f"{OUT}/mandi.csv", index=False)
mandi[["market", "crop", "year", "month", "price_qtl"]].to_csv(f"{OUT}/mandi_truth.csv", index=False)

# area stats + cost index --------------------------------------------------------------------------
a = []
for d in DIST:
    for c in CROPS:
        for yr in YEARS:
            base_area = {"cotton": 90, "soybean": 80, "tur": 45, "rice": 40, "wheat": 35, "gram": 30}[c] * 1000 * GROW_P[d][c] * (1 + .01 * (yr - 2005))
            a.append((d, c, yr, round(base_area * np.exp(rng.normal(0, .08)), 0)))
pd.DataFrame(a, columns=["district", "crop", "year", "sown_area_ha"]).to_csv(f"{OUT}/area_stats.csv", index=False)
fi = np.exp(np.cumsum(rng.normal(.04, .05, NY)))
fi[2022 - 2005] *= 1.25
fi[2023 - 2005:] *= 1.10
cr = []
for c in CROPS:
    for i, yr in enumerate(YEARS):
        cr.append((c, yr, round(CP[c]["cost"] * 1.07 ** i * (0.85 + .15 * fi[i] / fi[0] / (1.04 ** i)), 0), round(100 * fi[i] / fi[0], 1)))
pd.DataFrame(cr, columns=["crop", "year", "cost_rs_per_ha", "fert_price_index"]).to_csv(f"{OUT}/cost_index.csv", index=False)

tot = sum(len(pd.read_csv(f"{OUT}/{f}.csv")) for f in ["yields", "rainfall", "soil", "mandi", "area_stats", "cost_index"])
print({f: len(pd.read_csv(f"{OUT}/{f}.csv")) for f in ["yields", "rainfall", "soil", "mandi", "area_stats", "cost_index"]}, "TOTAL", tot)
