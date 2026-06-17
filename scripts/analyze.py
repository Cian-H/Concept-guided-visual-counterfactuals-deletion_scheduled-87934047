#!/usr/bin/env python3
"""Turn sweep.py outputs into paper-ready per-table CSVs and figures (no GPU).

Reads <sweep_out>/{sweep_scores.csv, calib.json, real_scores.csv, frd_all.csv,
iou_all.csv, cav_stability.csv}. For every classifier we derive all probability
variants from the saved logits:
  XRV : buggy = sigmoid(op_norm)  | uncal = op_norm  | platt = sigmoid(a*logit+b)
  Ark+: uncal = sigmoid(logit)    | platt = sigmoid(a*logit+b)
CFR/CPS use a probability-shift threshold (default 0.1) on the best alpha in the
intended direction (exaggerate: neg sources, +α, ΔP>thr ; curtail: pos sources,
−α, ΔP<−thr). Writes one CSV per table + a set of figures under <out>/.
"""
import os, json, argparse, itertools
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("--sweep", default="sweep_out")
ap.add_argument("--out", default="sweep_out/paper_artifacts")
ap.add_argument("--thr", type=float, default=0.1)
A = ap.parse_args()
TB, FG = f"{A.out}/tables", f"{A.out}/figs"
os.makedirs(TB, exist_ok=True); os.makedirs(FG, exist_ok=True)
sig = lambda v: 1.0 / (1.0 + np.exp(-v))
ORDER = ["Cardiomegaly", "Edema", "Consolidation", "Atelectasis", "Pleural Effusion"]
def T(df, n): df.to_csv(f"{TB}/{n}", index=False); print("table", n, df.shape)
def Fsave(fig, n): fig.tight_layout(); fig.savefig(f"{FG}/{n}", dpi=140, bbox_inches="tight"); plt.close(fig); print("fig", n)
calib = json.load(open(f"{A.sweep}/calib.json"))

# ---------- derive probability reps from logits ----------
def reps_for(row):
    ds, p = row["dataset"], row["pathology"]
    cx = calib["XRV"][ds][p]; ca = calib["Ark+"][ds][p]
    return {
        ("XRV", "buggy"): sig(row["xrv_opnorm"]),
        ("XRV", "uncal"): row["xrv_opnorm"],
        ("XRV", "platt"): sig(cx["a"] * row["xrv_logit"] + cx["b"]),
        ("Ark+", "uncal"): sig(row["ark_logit"]),
        ("Ark+", "platt"): sig(ca["a"] * row["ark_logit"] + ca["b"]),
    }
sw = pd.read_csv(f"{A.sweep}/sweep_scores.csv")
recs = []
for _, r in sw.iterrows():
    for (clf, cal), pr in reps_for(r).items():
        recs.append((r.dataset, r.backbone, r.conv, r.method, r.pathology, r.src_class,
                     r.source, r.alpha, clf, cal, float(pr)))
L = pd.DataFrame(recs, columns=["dataset", "backbone", "conv", "method", "pathology",
                                "src_class", "source", "alpha", "classifier", "calibration", "P"])

# ---------- T_rawP_curves: mean/se P per alpha ----------
g = L.groupby(["dataset", "backbone", "conv", "method", "pathology", "src_class", "classifier", "calibration", "alpha"])
curves = g["P"].agg(["mean", "sem", "count"]).reset_index().rename(columns={"mean": "P_mean", "sem": "P_se", "count": "n"})
curves["P_se"] = curves["P_se"].fillna(0.0)  # single-source groups -> sem NaN
T(curves.round(4), "t_rawP_curves.csv")

# ---------- T_cfr_full: CFR/CPS via best-alpha ----------
def cfr_block(sub, direction):
    # direction: 'exaggerate' uses neg sources & alpha>0 ; 'curtail' uses pos sources & alpha<0
    cls = "neg" if direction == "exaggerate" else "pos"
    sub = sub[sub.src_class == cls]
    if sub.empty: return None
    base = sub[sub.alpha == 0].set_index("source")["P"]
    shifts = []
    for s, sd in sub[sub.alpha != 0].groupby("source"):
        if s not in base.index: continue
        b = base.loc[s]
        if direction == "exaggerate":
            best = sd[sd.alpha > 0]["P"].max(); shift = best - b
        else:
            best = sd[sd.alpha < 0]["P"].min(); shift = best - b
        shifts.append(shift)
    shifts = np.array(shifts)
    if direction == "exaggerate":
        flip = (shifts > A.thr).mean() * 100
    else:
        flip = (shifts < -A.thr).mean() * 100
    return len(shifts), float(np.mean(shifts)), float(flip)
rows = []
keys = ["dataset", "backbone", "conv", "method", "pathology", "classifier", "calibration"]
for k, sub in L.groupby(keys):
    for direction in ("curtail", "exaggerate"):
        r = cfr_block(sub, direction)
        if r:
            rows.append(dict(zip(keys, k), direction=direction, N=r[0], CPS=round(r[1], 4), CFR=round(r[2], 1)))
cfr = pd.DataFrame(rows); T(cfr, "t_cfr_full.csv")
# summary: mean over pathologies
summ = cfr.groupby(["dataset", "backbone", "conv", "method", "classifier", "calibration", "direction"]).agg(
    CFR_mean=("CFR", "mean"), CPS_mean=("CPS", "mean"), N_path=("pathology", "nunique")).reset_index()
T(summ.round(3), "t_cfr_summary.csv")

# ---------- T_calibrated_agreement (Platt, per dataset/backbone/conv/method) ----------
rows = []
for k, sub in L[L.calibration == "platt"].groupby(["dataset", "backbone", "conv", "method", "pathology"]):
    d = {}
    for clf in ("XRV", "Ark+"):
        for direction, cls in (("curtail", "pos"), ("exaggerate", "neg")):
            s = sub[(sub.classifier == clf) & (sub.src_class == cls)]
            base = s[s.alpha == 0]["P"].mean()
            ext = s[s.alpha == s.alpha.min()]["P"].mean() if direction == "curtail" else s[s.alpha == s.alpha.max()]["P"].mean()
            d[f"{clf}_{direction}"] = round(ext - base, 3)
    rows.append(dict(zip(["dataset", "backbone", "conv", "method", "pathology"], k), **d,
                     agree_curtail=bool(d["XRV_curtail"] < -0.05 and d["Ark+_curtail"] < -0.05),
                     agree_exaggerate=bool(d["XRV_exaggerate"] > 0.05 and d["Ark+_exaggerate"] > 0.05)))
T(pd.DataFrame(rows), "t_calibrated_agreement.csv")

# ---------- real-image: T_calibration_params, T_realimage_sep + F double-sigmoid / nll ----------
try:
    rs = pd.read_csv(f"{A.sweep}/real_scores.csv")
    rows3, rows1 = [], []
    for ds in rs.dataset.unique():
        for p in [x for x in ORDER if x in rs[rs.dataset == ds].pathology.unique()]:
            sub = rs[(rs.dataset == ds) & (rs.pathology == p)]; y = sub.label.values
            xl, xo, al = sub.xrv_logit.values, sub.xrv_opnorm.values, sub.ark_logit.values
            cx, ca = calib["XRV"][ds][p], calib["Ark+"][ds][p]
            reps = {"xrv_buggy": sig(xo), "xrv_uncal": xo, "xrv_platt": sig(cx["a"] * xl + cx["b"]),
                    "ark_uncal": sig(al), "ark_platt": sig(ca["a"] * al + ca["b"])}
            rows1.append(dict(dataset=ds, pathology=p, xrv_auroc=round(roc_auc_score(y, xl), 3),
                              ark_auroc=round(roc_auc_score(y, al), 3),
                              **{f"gap_{k}": round(v[y == 1].mean() - v[y == 0].mean(), 4) for k, v in reps.items()}))
            for clf, lg in (("XRV", xl), ("Ark+", al)):
                c = calib[clf][ds][p]; pp = sig(c["a"] * lg + c["b"])
                rows3.append(dict(clf=clf, dataset=ds, pathology=p, a=round(c["a"], 3), b=round(c["b"], 3),
                                  T_only=round(c["T_only"], 2) if c["T_only"] else None,
                                  nll_raw=round(c["nll_raw"], 3), nll_platt=round(c["nll_platt"], 3),
                                  nll_temp=round(c["nll_temp"], 3),
                                  post_Ppos=round(pp[y == 1].mean(), 3), post_Pneg=round(pp[y == 0].mean(), 3)))
    t1 = pd.DataFrame(rows1); T(t1, "t_realimage_separability.csv"); T(pd.DataFrame(rows3), "t_calibration_params.csv")
    for ds in t1.dataset.unique():
        sub = t1[t1.dataset == ds].set_index("pathology").reindex([x for x in ORDER if x in t1.pathology.values])
        cols = ["gap_xrv_buggy", "gap_xrv_uncal", "gap_xrv_platt", "gap_ark_uncal", "gap_ark_platt"]
        fig, ax = plt.subplots(figsize=(11, 5)); x = np.arange(len(sub)); w = 0.16
        for i, c in enumerate(cols):
            ax.bar(x + (i - 2) * w, sub[c].values, w, label=c.replace("gap_", ""))
        ax.axhline(A.thr, ls="--", c="gray"); ax.set_xticks(x); ax.set_xticklabels(sub.index, rotation=15)
        ax.set_ylabel("real pos−neg gap"); ax.legend(fontsize=8, ncol=5); ax.set_title(f"{ds}: separability by scoring/calibration")
        Fsave(fig, f"f_doublesigmoid_gap_{ds}.png")
    t3 = pd.DataFrame(rows3); fig, ax = plt.subplots(figsize=(9, 5))
    grp = t3.groupby(["clf", "dataset"])[["nll_raw", "nll_platt", "nll_temp"]].mean(); x = np.arange(len(grp)); w = 0.25
    for i, c in enumerate(["nll_raw", "nll_platt", "nll_temp"]):
        ax.bar(x + (i - 1) * w, grp[c].values, w, label=c.replace("nll_", ""))
    ax.set_xticks(x); ax.set_xticklabels(["/".join(k) for k in grp.index], rotation=15); ax.legend(); ax.set_ylabel("mean NLL")
    ax.set_title("Calibration: raw vs Platt vs temperature"); Fsave(fig, "f_calibration_nll.png")
except Exception as e:
    print("real-image section skipped:", e)

# ---------- FRD / IoU / stability passthrough tables ----------
for src, name in (("frd_all.csv", "t_frd.csv"), ("iou_all.csv", "t_iou.csv"), ("cav_stability.csv", "t_cav_stability.csv")):
    try: T(pd.read_csv(f"{A.sweep}/{src}").round(4), name)
    except Exception as e: print(name, "skipped:", e)

# ---------- coverage matrix ----------
cov = sw.groupby(["dataset", "backbone", "conv", "method"]).size().reset_index(name="response_rows")
T(cov, "t_coverage.csv")

# ---------- key figures ----------
# F1: calibrated vs uncalibrated CFR effect (mean over pathologies), scaled+LR
try:
    s = summ[(summ.conv.isin(["scaled", "na"])) & (summ.method == "LR")]
    piv = s.pivot_table(index=["dataset", "backbone", "direction"], columns=["classifier", "calibration"], values="CFR_mean")
    fig, ax = plt.subplots(figsize=(13, 5)); idx = piv.index; x = np.arange(len(idx)); cols = [c for c in piv.columns if c[1] in ("uncal", "platt")]; w = 0.8 / len(cols)
    for i, c in enumerate(cols):
        ax.bar(x + (i - len(cols) / 2) * w, piv[c].values, w, label=f"{c[0]}-{c[1]}")
    ax.set_xticks(x); ax.set_xticklabels(["/".join(map(str, k)) for k in idx], rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("CFR (%)"); ax.legend(fontsize=8); ax.set_title("CFR: calibrated vs uncalibrated, both classifiers (scaled, LR)")
    Fsave(fig, "f_cfr_calibration_effect.png")
except Exception as e: print("f_cfr skipped:", e)

# F2/F3: raw-P curves per classifier (scaled, LR), 3 backbones x pathologies, add/remove uncal
def rawp_fig(clf, cal, fname):
    sub = curves[(curves.classifier == clf) & (curves.calibration == cal) & (curves.conv.isin(["scaled", "na"])) & (curves.method == "LR") & (curves.dataset == "CheXpert")]
    bks = ["AE", "SD", "SDXL"]; paths = [p for p in ORDER if p in sub.pathology.unique()]
    if sub.empty: return
    fig, axes = plt.subplots(len(bks), len(paths), figsize=(4 * len(paths), 3.2 * len(bks)), squeeze=False)
    for r, bk in enumerate(bks):
        for c, p in enumerate(paths):
            ax = axes[r][c]
            for srcc, color in (("neg", "tab:green"), ("pos", "tab:brown")):
                d = sub[(sub.backbone == bk) & (sub.pathology == p) & (sub.src_class == srcc)].sort_values("alpha")
                if not d.empty: ax.errorbar(d.alpha, d.P_mean, yerr=d.P_se, marker="o", ms=3, color=color, label=srcc)
            ax.grid(alpha=0.3)
            if r == 0: ax.set_title(p)
            if c == 0: ax.set_ylabel(f"{bk}")
    fig.suptitle(f"{clf} {cal} P(target) vs α (CheXpert, scaled, LR)  green=neg(+exag) brown=pos(−curtail)")
    Fsave(fig, fname)
try:
    rawp_fig("XRV", "uncal", "f_rawP_xrv_uncal.png"); rawp_fig("XRV", "platt", "f_rawP_xrv_platt.png")
    rawp_fig("Ark+", "uncal", "f_rawP_ark_uncal.png"); rawp_fig("Ark+", "platt", "f_rawP_ark_platt.png")
except Exception as e: print("rawp figs skipped:", e)

# F4: scaled vs unscaled convention effect on calibrated CFR (SD/SDXL)
try:
    s = summ[(summ.calibration == "platt") & (summ.classifier == "XRV") & (summ.backbone.isin(["SD", "SDXL"]))]
    piv = s.pivot_table(index=["dataset", "backbone", "method", "direction"], columns="conv", values="CFR_mean")
    if {"scaled", "unscaled"}.issubset(piv.columns):
        fig, ax = plt.subplots(figsize=(12, 5)); x = np.arange(len(piv)); w = 0.4
        ax.bar(x - w / 2, piv["scaled"].values, w, label="scaled"); ax.bar(x + w / 2, piv["unscaled"].values, w, label="unscaled")
        ax.set_xticks(x); ax.set_xticklabels(["/".join(map(str, k)) for k in piv.index], rotation=60, ha="right", fontsize=7)
        ax.set_ylabel("CFR (%) XRV-platt"); ax.legend(); ax.set_title("α-convention effect (scaled vs unscaled)")
        Fsave(fig, "f_convention_effect.png")
except Exception as e: print("convention fig skipped:", e)

# F5: FRD and CAV-stability bars
try:
    fr = pd.read_csv(f"{A.sweep}/frd_all.csv")
    fig, ax = plt.subplots(figsize=(12, 5)); fr["key"] = fr.dataset + "/" + fr.backbone + "/" + fr.conv + "/" + fr.method
    piv = fr.pivot_table(index="key", columns="direction", values="frd")
    piv.plot(kind="bar", ax=ax); ax.set_ylabel("FRD (lower=closer)"); ax.set_title("FRD across backbone/conv/method")
    plt.xticks(rotation=70, ha="right", fontsize=6); Fsave(fig, "f_frd.png")
except Exception as e: print("frd fig skipped:", e)
try:
    cs = pd.read_csv(f"{A.sweep}/cav_stability.csv")
    fig, ax = plt.subplots(figsize=(11, 5)); cs["key"] = cs.backbone + "/" + cs.conv + "/" + cs.method
    cs.pivot_table(index="pathology", columns="key", values="inter_chex_nih").reindex([p for p in ORDER if p in cs.pathology.values]).plot(kind="bar", ax=ax)
    ax.set_ylabel("CheX↔NIH CAV cosine"); ax.set_title("CAV cross-dataset stability"); plt.xticks(rotation=15)
    Fsave(fig, "f_cav_stability.png")
except Exception as e: print("stability fig skipped:", e)

print("\nANALYZE DONE ->", A.out)
