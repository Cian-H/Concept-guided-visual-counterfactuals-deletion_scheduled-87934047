#!/usr/bin/env python3
"""Total sweep harness for concept-guided CXR counterfactual evaluation (BATCHED).

Fully-crossed, checkpointed, config-driven. All GPU work (encode / decode /
both classifiers) is batched, so the eval sample size M can be large.

  backbones  : AE (TXRV ResNetAE) | SD-1.5 VAE | SDXL VAE
  conventions: VAE latent scaling  scaled (z*sf) | unscaled (raw mean)   (AE: n/a)
  CAV methods: LR (logistic coef)  | MD (class mean-difference)
  classifiers: XRV DenseNet        | Ark+ Swin-L   (logits saved -> all
               probability variants -- buggy/op_norm/raw/Platt -- derive offline)
  datasets   : CheXpert | NIH

Sampling is patient-deduplicated and split into DISJOINT slices per
(dataset, pathology, class):  eval=[0:M] | calib=[M:M+n_calib] | cav=[...]  -- no
image/patient leakage between CAV-training, calibration, and evaluation.

Outputs under --out (default sweep_out/): cavs/*.npz, calib.json, real_scores.csv,
sweep_scores.csv (core, per-source edit logits over the signed-alpha grid),
frd_all.csv, iou_all.csv, cav_stability.csv. analyze.py turns these into tables/figs.

NOTE on CFR (computed in analyze.py): it is the BEST shift over the alpha grid
(oracle-alpha) -> report as an upper-bound flip rate, not a fixed-strength effect.
"""
import os, json, argparse, itertools, tempfile, shutil
os.environ.setdefault("TORCH_HOME", "/tmp/torch_cache")
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
import torchxrayvision as xrv
import torchvision.transforms as TT
import timm
from diffusers import AutoencoderKL
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

# ---------------------------------------------------------------- config
# Dataset/weight locations are set from CLI args (env fallback) right after
# parse_args(); see below. This default points at the bundled NIH bbox file,
# resolved relative to the repo root so it works from any working directory.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_NIH_BBOX = os.path.join(_REPO_ROOT, "metadata", "BBox_List_2017.csv")
PATHOLOGY = {
    "Cardiomegaly":     dict(chex="Cardiomegaly",     nih="Cardiomegaly", txrv=10, ark="Cardiomegaly"),
    "Edema":            dict(chex="Edema",            nih="Edema",        txrv=4,  ark="Edema"),
    "Consolidation":    dict(chex="Consolidation",    nih="Consolidation",txrv=1,  ark="Consolidation"),
    "Atelectasis":      dict(chex="Atelectasis",      nih="Atelectasis",  txrv=0,  ark="Atelectasis"),
    "Pleural Effusion": dict(chex="Pleural Effusion", nih="Effusion",     txrv=7,  ark="Effusion"),
}
GRID = {
    ("AE", "na"):        [-15., -10., -5., 0., 5., 10., 15.],
    ("VAE", "scaled"):   [-61.118, -40.745, -20.373, 0., 20.373, 40.745, 61.118],
    ("VAE", "unscaled"): [-250., -166.7, -83.3, 0., 83.3, 166.7, 250.],
}
CONVS = {"AE": ["na"], "SD": ["scaled", "unscaled"], "SDXL": ["scaled", "unscaled"]}

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="sweep_out")
ap.add_argument("--phases", default="cavs,calib,response,frd,iou,stability")
ap.add_argument("--datasets", default="CheXpert,NIH")
ap.add_argument("--backbones", default="AE,SD,SDXL")
ap.add_argument("--convs", default="")
ap.add_argument("--methods", default="LR,MD")
ap.add_argument("--pathologies", default=",".join(PATHOLOGY))
ap.add_argument("--M", type=int, default=8, help="eval sources per (pathology, class)")
ap.add_argument("--n_cav", type=int, default=200)
ap.add_argument("--n_calib", type=int, default=120)
ap.add_argument("--bs_enc", type=int, default=16)
ap.add_argument("--bs_dec", type=int, default=16)
ap.add_argument("--bs_xrv", type=int, default=64)
ap.add_argument("--bs_ark", type=int, default=12)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--force", action="store_true")
ap.add_argument("--chexpert-root", default=os.environ.get("CXR_CHEXPERT_ROOT", "/data/chexpert"),
                help="dir CONTAINING CheXpert-v1.0-small/ (CSVs + image tree)")
ap.add_argument("--nih-root", default=os.environ.get("CXR_NIH_ROOT", "/data/nih_raw"),
                help="dir of NIH ChestX-ray14 images")
ap.add_argument("--nih-csv", default=os.environ.get("CXR_NIH_CSV", ""),
                help="NIH metadata CSV (default: {nih-root}/Data_Entry_2017_v2020.csv)")
ap.add_argument("--nih-bbox", default=os.environ.get("CXR_NIH_BBOX", _DEFAULT_NIH_BBOX),
                help="NIH BBox_List_2017.csv (default: bundled metadata/ copy)")
ap.add_argument("--ark-weights", default=os.environ.get("CXR_ARK_WEIGHTS",
                "/data/ark_weights/Ark+SwinL768_ChestX-ray14_ft.pth.tar"),
                help="Ark+ SwinL768 checkpoint (.pth.tar)")
ap.add_argument("--torch-home", default=os.environ.get("TORCH_HOME", "/tmp/torch_cache"),
                help="torch hub cache dir for auto-downloaded XRV/VAE weights")
A = ap.parse_args()
os.environ["TORCH_HOME"] = A.torch_home
CHEX_ROOT = A.chexpert_root
CHEX = os.path.join(CHEX_ROOT, "CheXpert-v1.0-small")
NIH_IMG = A.nih_root
NIH_CSV = A.nih_csv or os.path.join(NIH_IMG, "Data_Entry_2017_v2020.csv")
NIH_BBOX = A.nih_bbox
ARK_W = A.ark_weights
DATASETS = A.datasets.split(","); BACKBONES = A.backbones.split(","); METHODS = A.methods.split(",")
PATHS = [p for p in PATHOLOGY if p in A.pathologies.split(",")]; PHASES = A.phases.split(",")
dev = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(f"{A.out}/cavs", exist_ok=True)
def convs_for(bb): return [c for c in CONVS[bb] if (not A.convs or c in A.convs.split(","))]
def fam(bb): return "AE" if bb == "AE" else "VAE"
def grid(bb, conv): return GRID[(fam(bb), conv)]

# ---------------------------------------------------------------- models (lazy)
_M = {}
def xrv_model():
    if "xrv" not in _M: _M["xrv"] = xrv.models.DenseNet(weights="densenet121-res224-all").to(dev).eval()
    return _M["xrv"]
ARK = ["Atelectasis","Cardiomegaly","Effusion","Infiltration","Mass","Nodule","Pneumonia",
       "Pneumothorax","Consolidation","Edema","Emphysema","Fibrosis","Pleural_Thickening","Hernia"]
AI = {l: i for i, l in enumerate(ARK)}
def ark_model():
    if "ark" not in _M:
        m = timm.create_model("swin_large_patch4_window12_384", pretrained=False, num_classes=14, img_size=768, window_size=12)
        ck = torch.load(ARK_W, map_location="cpu", weights_only=False)
        src = {k.replace("module.", "", 1): v for k, v in ck["state_dict"].items()}
        mp = {}
        for k, v in src.items():
            if ".downsample." in k:
                p = k.split("."); p[1] = str(int(p[1]) + 1); mp[".".join(p)] = v
            elif k.startswith("head."): mp[k.replace("head.", "head.fc.")] = v
            else: mp[k] = v
        m.load_state_dict(mp, strict=False); _M["ark"] = m.eval().to(dev)
    return _M["ark"]
def vae(bb):
    key = "vae_" + bb
    if key not in _M:
        repo = "stable-diffusion-v1-5/stable-diffusion-v1-5" if bb == "SD" else "stabilityai/sdxl-vae"
        sub = {"subfolder": "vae"} if bb == "SD" else {}
        _M[key] = AutoencoderKL.from_pretrained(repo, torch_dtype=torch.float32, **sub).to(dev).eval()
    return _M[key]
def ae_model():
    if "ae" not in _M: _M["ae"] = xrv.autoencoders.ResNetAE(weights="101-elastic").to(dev).eval()
    return _M["ae"]

# ---------------------------------------------------------------- BATCHED classifier scoring (logits)
_xc, _xr = xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)
def _xrv_t(pil):
    n = xrv.datasets.normalize(np.array(pil.convert("L")), 255)
    return torch.from_numpy(_xr(_xc(n[None, ...])))  # [1,224,224]
def xrv_batch(pils, idx):
    m = xrv_model(); lg, op = [], []
    for i in range(0, len(pils), A.bs_xrv):
        x = torch.stack([_xrv_t(p) for p in pils[i:i + A.bs_xrv]]).to(dev).float()
        assert x.shape[-1] == 224 and x.shape[-2] == 224
        with torch.no_grad():
            f = F.adaptive_avg_pool2d(F.relu(m.features(x)), (1, 1)).flatten(1)
            lg += m.classifier(f)[:, idx].cpu().tolist()       # raw logit (feature path == DenseNet.forward)
            op += m(x)[:, idx].cpu().tolist()                  # op_norm-calibrated prob
    return lg, op
_ap = TT.Compose([TT.Resize(896), TT.CenterCrop(768), TT.ToTensor(),
                  TT.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
def ark_batch(pils, lab):
    m = ark_model(); out = []
    for i in range(0, len(pils), A.bs_ark):
        x = torch.stack([_ap(p.convert("RGB")) for p in pils[i:i + A.bs_ark]]).to(dev)
        with torch.no_grad():
            out += m(x)[:, AI[lab]].cpu().tolist()
    return out

# ---------------------------------------------------------------- BATCHED encode/decode
_sc, _sr = xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(512)
def make_codec(bb, conv):
    if bb == "AE":
        ae = ae_model()
        def enc_many(paths):
            outs = []
            for i in range(0, len(paths), A.bs_enc):
                xs = []
                for pa in paths[i:i + A.bs_enc]:
                    a = np.array(Image.open(pa).convert("L").resize((224, 224)))
                    xs.append(torch.from_numpy(xrv.datasets.normalize(a, 255)[None, :, :]))
                with torch.no_grad():
                    outs.append(ae.encode(torch.stack(xs).to(dev).float()))
            return torch.cat(outs, 0)
        def dec_many(Z):
            res = []
            for i in range(0, Z.shape[0], A.bs_dec):
                with torch.no_grad():
                    d = ae.decode(Z[i:i + A.bs_dec])
                for k in range(d.shape[0]):
                    o = np.clip((d[k, 0].cpu().numpy() + 1024) / 2048 * 255, 0, 255).astype(np.uint8)
                    res.append(Image.fromarray(o).filter(ImageFilter.UnsharpMask(1.5, 120, 3)))
            return res
        return enc_many, dec_many
    v = vae(bb); sf = float(v.config.scaling_factor) if conv == "scaled" else 1.0
    def enc_many(paths):
        outs = []
        for i in range(0, len(paths), A.bs_enc):
            xs = []
            for pa in paths[i:i + A.bs_enc]:
                a = np.array(Image.open(pa).convert("L")).astype(np.float32) / 255.0
                t = torch.from_numpy(_sr(_sc(a[None, ...])))
                if t.shape[0] == 1: t = t.repeat(3, 1, 1)
                xs.append((t - 0.5) / 0.5)
            with torch.no_grad():
                outs.append(v.encode(torch.stack(xs).to(dev)).latent_dist.mean * sf)
        return torch.cat(outs, 0)
    def dec_many(Z):
        res = []
        for i in range(0, Z.shape[0], A.bs_dec):
            with torch.no_grad():
                d = v.decode(Z[i:i + A.bs_dec] / sf).sample
            for k in range(d.shape[0]):
                o = d[k].cpu(); o = o.mean(0) if o.ndim == 3 else o
                res.append(Image.fromarray(np.clip((o.numpy() + 1) * 127.5, 0, 255).astype(np.uint8)))
        return res
    return enc_many, dec_many

# ---------------------------------------------------------------- data samplers (disjoint, patient-level)
_cache = {}
def chex_df(split):
    if ("chex", split) not in _cache:
        d = pd.read_csv(f"{CHEX}/{split}.csv"); d = d[d["Path"].str.contains("frontal")]; _cache[("chex", split)] = d
    return _cache[("chex", split)]
def nih_df():
    if "nih" not in _cache: _cache["nih"] = pd.read_csv(NIH_CSV)
    return _cache["nih"]
def _chex_patient(p):
    for t in p.split("/"):
        if t.startswith("patient"): return t
    return p
def build_pool(dataset, pathology, label):
    key = ("pool", dataset, pathology, label)
    if key in _cache: return _cache[key]
    c = PATHOLOGY[pathology]
    if dataset == "CheXpert":
        df = pd.concat([chex_df("valid"), chex_df("train")], ignore_index=True)
        sub = df[df[c["chex"]] == (1.0 if label else 0.0)].copy()
        sub["patient"] = sub["Path"].map(_chex_patient)
        sub = sub.drop_duplicates("patient").sample(frac=1, random_state=A.seed)
        pool = [os.path.join(CHEX_ROOT, p) for p in sub["Path"]]
    else:
        has = nih_df()["Finding Labels"].str.contains(c["nih"], regex=False)
        sub = (nih_df()[has] if label else nih_df()[~has]).copy()
        sub = sub.drop_duplicates("Patient ID").sample(frac=1, random_state=A.seed)
        pool = [os.path.join(NIH_IMG, f) for f in sub["Image Index"]]
    _cache[key] = pool; return pool
def paths_for(dataset, pathology, label, role):
    pool = build_pool(dataset, pathology, label)
    lo = {"eval": 0, "calib": A.M, "cav": A.M + A.n_calib}[role]
    hi = {"eval": A.M, "calib": A.M + A.n_calib, "cav": A.M + A.n_calib + A.n_cav}[role]
    return pool[lo:hi]
def unit(v): return v / (np.linalg.norm(v) + 1e-12)

# ---------------------------------------------------------------- PHASE cavs
def cav_path(ds, bb, conv, p): return f"{A.out}/cavs/{ds}_{bb}_{conv}_{p.replace(' ', '')}.npz"
def phase_cavs():
    for ds, bb in itertools.product(DATASETS, BACKBONES):
        for conv in convs_for(bb):
            enc_many, _ = make_codec(bb, conv)
            for p in PATHS:
                fp = cav_path(ds, bb, conv, p)
                if os.path.exists(fp) and not A.force: continue
                Zp = enc_many(paths_for(ds, p, 1, "cav")); Zn = enc_many(paths_for(ds, p, 0, "cav"))
                shape = (1,) + tuple(Zp.shape[1:])
                Zpf, Znf = Zp.flatten(1).cpu().numpy(), Zn.flatten(1).cpu().numpy()
                md = unit(Zpf.mean(0) - Znf.mean(0)).astype(np.float32)
                clf = LogisticRegression(C=1e6, max_iter=2000).fit(np.vstack([Zpf, Znf]), [1] * len(Zpf) + [0] * len(Znf))
                lr = unit(clf.coef_[0]).astype(np.float32)
                np.savez(fp, LR=lr, MD=md, shape=np.array(shape))
                print("cav", ds, bb, conv, p, "cos(LR,MD)=%.3f n=%d/%d" % (float(np.dot(lr, md)), len(Zpf), len(Znf)))
def load_cav(ds, bb, conv, p, method):
    d = np.load(cav_path(ds, bb, conv, p)); return torch.from_numpy(d[method]).float().to(dev).view(*tuple(d["shape"]))

# ---------------------------------------------------------------- PHASE calib (+ real logits)
def phase_calib():
    rows_real, calib = [], {"XRV": {ds: {} for ds in DATASETS}, "Ark+": {ds: {} for ds in DATASETS}}
    def fit(z, y):
        z = np.array(z).reshape(-1, 1); y = np.array(y)
        pl = LogisticRegression(C=1e6, max_iter=2000).fit(z, y)
        tp = LogisticRegression(C=1e6, max_iter=2000, fit_intercept=False).fit(z, y)
        a, b, ta = float(pl.coef_[0, 0]), float(pl.intercept_[0]), float(tp.coef_[0, 0])
        s = lambda v: 1 / (1 + np.exp(-v))
        return dict(a=a, b=b, T_only=(1 / ta if ta else None),
                    nll_raw=float(log_loss(y, s(z.ravel()), labels=[0, 1])),
                    nll_platt=float(log_loss(y, s(a * z.ravel() + b), labels=[0, 1])),
                    nll_temp=float(log_loss(y, s(ta * z.ravel()), labels=[0, 1])))
    for ds in DATASETS:
        for p in PATHS:
            c = PATHOLOGY[p]
            pos, neg = paths_for(ds, p, 1, "calib"), paths_for(ds, p, 0, "calib")
            pils = [Image.open(x) for x in pos + neg]; y = [1] * len(pos) + [0] * len(neg)
            xl, xo = xrv_batch(pils, c["txrv"]); al = ark_batch(pils, c["ark"])
            for i in range(len(pils)):
                rows_real.append(dict(dataset=ds, pathology=p, label=y[i], xrv_logit=xl[i], xrv_opnorm=xo[i], ark_logit=al[i]))
            calib["XRV"][ds][p] = fit(xl, y); calib["Ark+"][ds][p] = fit(al, y)
            print("calib", ds, p, "XRV b=%.2f Ark+ b=%.2f n=%d" % (calib["XRV"][ds][p]["b"], calib["Ark+"][ds][p]["b"], len(y)))
    json.dump(calib, open(f"{A.out}/calib.json", "w"), indent=2)
    pd.DataFrame(rows_real).to_csv(f"{A.out}/real_scores.csv", index=False)

# ---------------------------------------------------------------- PHASE response (batched)
def phase_response():
    fp = f"{A.out}/sweep_scores.csv"; done = set()
    if os.path.exists(fp) and not A.force:
        done = set(map(tuple, pd.read_csv(fp)[["dataset", "backbone", "conv", "method", "pathology"]].drop_duplicates().values))
    rows = []
    for ds, bb in itertools.product(DATASETS, BACKBONES):
        for conv in convs_for(bb):
            enc_many, dec_many = make_codec(bb, conv); g = grid(bb, conv)
            for p in PATHS:
                c = PATHOLOGY[p]
                z = {cls: enc_many(paths_for(ds, p, 1 if cls == "pos" else 0, "eval")) for cls in ("pos", "neg")}
                for method in METHODS:
                    if (ds, bb, conv, method, p) in done: continue
                    v = load_cav(ds, bb, conv, p, method)
                    for cls in ("pos", "neg"):
                        Zc = z[cls]; meta = []
                        edits = []
                        for i in range(Zc.shape[0]):
                            for al in g:
                                edits.append(Zc[i:i + 1] + al * v); meta.append((i, al))
                        imgs = dec_many(torch.cat(edits, 0))
                        xl, xo = xrv_batch(imgs, c["txrv"]); al_ = ark_batch(imgs, c["ark"])
                        for (i, alpha), a, b, cc in zip(meta, xl, xo, al_):
                            rows.append(dict(dataset=ds, backbone=bb, conv=conv, method=method, pathology=p,
                                             src_class=cls, source=i, alpha=alpha, xrv_logit=a, xrv_opnorm=b, ark_logit=cc))
                    print("response", ds, bb, conv, method, p, "rows=%d" % len(rows))
    if rows:
        new = pd.DataFrame(rows)
        if os.path.exists(fp) and not A.force: new = pd.concat([pd.read_csv(fp), new], ignore_index=True)
        new.to_csv(fp, index=False)

# ---------------------------------------------------------------- PHASE frd (batched)
def phase_frd():
    from frd_score import frd
    rows = []
    for ds, bb in itertools.product(DATASETS, BACKBONES):
        for conv in convs_for(bb):
            enc_many, dec_many = make_codec(bb, conv); amax = max(grid(bb, conv))
            for method in METHODS:
                for direction, cls, alpha in (("curtail", "pos", -amax), ("exaggerate", "neg", amax)):
                    rd, sd = tempfile.mkdtemp(prefix="frd_r_"), tempfile.mkdtemp(prefix="frd_s_"); n = 0
                    for p in PATHS:
                        v = load_cav(ds, bb, conv, p, method)
                        Z0 = enc_many(paths_for(ds, p, 1 if cls == "pos" else 0, "eval"))
                        for k, im in enumerate(dec_many(Z0)): im.save(f"{rd}/{p[:4]}_{k}.png")
                        for k, im in enumerate(dec_many(Z0 + alpha * v)): im.save(f"{sd}/{p[:4]}_{k}.png"); n += 1
                    try: score = float(frd.compute_frd(paths=[rd, sd], frd_version="v1"))
                    except Exception as e: score = float("nan"); print("FRD err", e)
                    rows.append(dict(dataset=ds, backbone=bb, conv=conv, method=method, direction=direction, n=n, frd=score))
                    print("frd", ds, bb, conv, method, direction, "=%.3f" % score)
                    shutil.rmtree(rd, ignore_errors=True); shutil.rmtree(sd, ignore_errors=True)
    pd.DataFrame(rows).to_csv(f"{A.out}/frd_all.csv", index=False)

# ---------------------------------------------------------------- PHASE iou (NIH bbox, batched)
def phase_iou():
    if not os.path.exists(NIH_BBOX): print("no bbox csv, skip iou"); return
    bb_df = pd.read_csv(NIH_BBOX); bb_df.columns = [c.strip() for c in bb_df.columns]
    fcol = [c for c in bb_df.columns if "Finding" in c][0]
    BB = {"Cardiomegaly": "Cardiomegaly", "Atelectasis": "Atelectasis", "Pleural Effusion": "Effusion"}
    rows = []
    for bb in BACKBONES:
        for conv in convs_for(bb):
            enc_many, dec_many = make_codec(bb, conv); amax = max(grid(bb, conv))
            for method in METHODS:
                for p, tok in BB.items():
                    if p not in PATHS: continue
                    cand = bb_df[bb_df[fcol] == tok].head(A.M)
                    fps = [os.path.join(NIH_IMG, r["Image Index"]) for _, r in cand.iterrows() if os.path.exists(os.path.join(NIH_IMG, r["Image Index"]))]
                    rrows = [r for _, r in cand.iterrows() if os.path.exists(os.path.join(NIH_IMG, r["Image Index"]))]
                    if not fps: continue
                    v = load_cav("NIH", bb, conv, p, method); Z0 = enc_many(fps)
                    for direction, alpha in (("curtail", -amax), ("exaggerate", amax)):
                        base = dec_many(Z0); cf = dec_many(Z0 + alpha * v); ious = []
                        for r, bimg, cimg in zip(rrows, base, cf):
                            b = np.array(bimg.convert("L"), float); cc = np.array(cimg.convert("L"), float)
                            diff = np.abs(cc - b); H, W = diff.shape
                            mask = diff >= np.quantile(diff, 0.90)
                            bx, by, bw, bh = [float(r[k]) for k in ["Bbox [x", "y", "w", "h]"]]
                            sx, sy = W / 1024.0, H / 1024.0; gt = np.zeros_like(mask)
                            gt[int(by * sy):int((by + bh) * sy), int(bx * sx):int((bx + bw) * sx)] = True
                            inter = (mask & gt).sum(); uni = (mask | gt).sum()
                            ious.append(inter / uni if uni else 0.0)
                        rows.append(dict(backbone=bb, conv=conv, method=method, pathology=p, direction=direction,
                                         n=len(ious), iou_mean=float(np.mean(ious)) if ious else float("nan")))
                        print("iou", bb, conv, method, p, direction, "=%.3f" % (np.mean(ious) if ious else float("nan")))
    pd.DataFrame(rows).to_csv(f"{A.out}/iou_all.csv", index=False)

# ---------------------------------------------------------------- PHASE stability
def phase_stability():
    rows = []
    for bb in BACKBONES:
        for conv in convs_for(bb):
            for method in METHODS:
                for p in PATHS:
                    fa, fb = cav_path("CheXpert", bb, conv, p), cav_path("NIH", bb, conv, p)
                    if not (os.path.exists(fa) and os.path.exists(fb)): continue
                    a = unit(np.load(fa)[method]); b = unit(np.load(fb)[method])
                    rand = unit(np.random.RandomState(0).randn(len(a)))
                    rows.append(dict(backbone=bb, conv=conv, method=method, pathology=p,
                                     inter_chex_nih=float(np.dot(a, b)), random_baseline=float(np.dot(a, rand))))
    pd.DataFrame(rows).to_csv(f"{A.out}/cav_stability.csv", index=False); print("stability rows:", len(rows))

# ---------------------------------------------------------------- run
if __name__ == "__main__":
    print("config:", dict(datasets=DATASETS, backbones=BACKBONES, methods=METHODS, pathologies=PATHS,
                          phases=PHASES, M=A.M, n_cav=A.n_cav, n_calib=A.n_calib,
                          bs=(A.bs_enc, A.bs_dec, A.bs_xrv, A.bs_ark), out=A.out))
    _needs_ark = {"calib", "response", "frd", "iou"}
    if _needs_ark.intersection(PHASES) and not os.path.exists(ARK_W):
        raise SystemExit(f"Ark+ weights not found: {ARK_W}\n"
                         f"Set --ark-weights / $CXR_ARK_WEIGHTS (see DATA.md).")
    fns = dict(cavs=phase_cavs, calib=phase_calib, response=phase_response,
               frd=phase_frd, iou=phase_iou, stability=phase_stability)
    for ph in PHASES:
        print(f"\n===== PHASE {ph} ====="); fns[ph]()
    print("\nSWEEP DONE ->", A.out)
