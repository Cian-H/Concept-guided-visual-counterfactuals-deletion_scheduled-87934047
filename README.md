# On the Evaluation of Concept-Guided Generation of Visual Counterfactual Explanations in Medical Imaging

Code and reference results for evaluating concept-guided visual counterfactuals
on chest X-rays. The harness fully crosses **3 backbones × 2 latent conventions
× 2 CAV methods × 2 classifiers × 2 datasets** and writes per-source edit logits
and image metrics to CSV; a second script derives summary tables and figures.

- Backbones: `AE` (TorchXRayVision ResNet autoencoder), `SD` (SD-1.5 VAE), `SDXL` (SDXL VAE)
- Conventions: latent `scaled` vs `unscaled` (AE: `na`)
- CAV methods: `LR` (logistic coefficient), `MD` (class mean-difference)
- Classifiers: XRV DenseNet, Ark+ SwinL
- Datasets: CheXpert, NIH ChestX-ray14
- Pathologies: Cardiomegaly, Edema, Consolidation, Atelectasis, Pleural Effusion

## Authors

Omitted for anonymous review.

## Install

```bash
pip install -r requirements.txt
```

`frd-score` and `pyradiomics` are required only for the FRD phase. GPU + CUDA
are required for `sweep.py`; `analyze.py` is CPU-only.

## Configure data & weights

Datasets and the Ark+ checkpoint are not redistributed. See **`DATA.md`** for
how to obtain and lay them out. Each location is a CLI flag with an environment
fallback:

| Flag | Env var | Default |
|------|---------|---------|
| `--chexpert-root` | `CXR_CHEXPERT_ROOT` | `/data/chexpert` |
| `--nih-root` | `CXR_NIH_ROOT` | `/data/nih_raw` |
| `--nih-csv` | `CXR_NIH_CSV` | `<nih-root>/Data_Entry_2017_v2020.csv` |
| `--nih-bbox` | `CXR_NIH_BBOX` | bundled `metadata/BBox_List_2017.csv` |
| `--ark-weights` | `CXR_ARK_WEIGHTS` | `/data/ark_weights/Ark+SwinL768_ChestX-ray14_ft.pth.tar` |
| `--torch-home` | `TORCH_HOME` | `/tmp/torch_cache` |

## Reproduce

```bash
# classifier core (M=50): CAVs, calibration, response sweep, CAV-stability
python scripts/sweep.py --M 50 --phases cavs,calib,response,stability \
  --chexpert-root /data/chexpert --nih-root /data/nih_raw \
  --ark-weights /path/to/Ark+SwinL768_ChestX-ray14_ft.pth.tar --out results/raw

# image metrics (M=20, reuses the M=50 CAVs): FRD (both datasets) + IoU (NIH)
python scripts/sweep.py --M 20 --phases frd,iou \
  --chexpert-root /data/chexpert --nih-root /data/nih_raw \
  --ark-weights /path/to/Ark+SwinL768_ChestX-ray14_ft.pth.tar --out results/raw

# tables + figures (CPU; reads results/raw/, writes results/tables + results/figs)
python scripts/analyze.py --sweep results/raw --out results
```

Subset any axis for quick runs, e.g.
`--backbones SD --convs scaled --pathologies Cardiomegaly --M 4`. Each phase
checkpoints to its output file, so reruns skip completed work.

The committed `results/` is the reference output of the commands above. Without
a GPU you can regenerate every table and figure from the committed `results/raw/`
using just the final `analyze.py` command above.

## Layout

```
scripts/sweep.py      batched, checkpointed GPU harness (writes results/raw/*)
scripts/analyze.py    CPU; derives results/tables/* and results/figs/* from results/raw/*
metadata/             BBox_List_2017.csv (bundled; NIH IoU phase)
results/raw/          per-source logits + calibration + image-metric CSVs
results/tables/       summary CSVs (one per file below)
results/figs/         PNG figures
```

## Outputs reference

### `results/raw/`

| File | Contents | Produced by |
|------|----------|-------------|
| `sweep_scores.csv` | per-source edit logits over the signed-α grid. Columns: `dataset, backbone, conv, method, pathology, src_class, source, alpha, xrv_logit, xrv_opnorm, ark_logit` | `sweep.py` phase `response` |
| `calib.json` | per-(classifier, dataset, pathology) Platt `a,b`, `T_only`, and `nll_raw/nll_platt/nll_temp` | `sweep.py` phase `calib` |
| `real_scores.csv` | real-image logits. Columns: `dataset, pathology, label, xrv_logit, xrv_opnorm, ark_logit` | `sweep.py` phase `calib` |
| `frd_all.csv` | Columns: `dataset, backbone, conv, method, direction, n, frd` | `sweep.py` phase `frd` |
| `iou_all.csv` | Columns: `backbone, conv, method, pathology, direction, n, iou_mean` | `sweep.py` phase `iou` |
| `cav_stability.csv` | Columns: `backbone, conv, method, pathology, inter_chex_nih, random_baseline` | `sweep.py` phase `stability` |

### `results/tables/`

| File | Contents |
|------|----------|
| `t_rawP_curves.csv` | mean/SE P(target) per α. Columns: `dataset, backbone, conv, method, pathology, src_class, classifier, calibration, alpha, P_mean, P_se, n` |
| `t_cfr_full.csv` | per-cell CFR/CPS. Columns: `dataset, backbone, conv, method, pathology, classifier, calibration, direction, N, CPS, CFR` (CFR — classifier flip rate %; CPS — classifier probability shift) |
| `t_cfr_summary.csv` | summary rows per dataset/backbone/conv/method/classifier/calibration/direction (pathologies collapsed). Columns: `dataset, backbone, conv, method, classifier, calibration, direction, CFR_mean, CPS_mean, N_path` |
| `t_calibrated_agreement.csv` | per-cell Platt ΔP by classifier/direction plus boolean `agree_curtail`/`agree_exaggerate` |
| `t_realimage_separability.csv` | real-image `xrv_auroc`, `ark_auroc`, and pos−neg `gap_*` per probability variant |
| `t_calibration_params.csv` | per-(classifier, dataset, pathology) `a, b, T_only, nll_raw, nll_platt, nll_temp, post_Ppos, post_Pneg` |
| `t_frd.csv` | passthrough of `frd_all.csv` |
| `t_iou.csv` | passthrough of `iou_all.csv` |
| `t_cav_stability.csv` | passthrough of `cav_stability.csv` |
| `t_coverage.csv` | response-row counts per `dataset, backbone, conv, method` |

### `results/figs/`

| File | Contents |
|------|----------|
| `f_doublesigmoid_gap_CheXpert.png`, `f_doublesigmoid_gap_NIH.png` | real pos−neg gap by scoring/calibration variant |
| `f_calibration_nll.png` | mean NLL: raw vs Platt vs temperature |
| `f_cfr_calibration_effect.png` | CFR by classifier × calibration (scaled, LR) |
| `f_rawP_xrv_uncal.png`, `f_rawP_xrv_platt.png`, `f_rawP_ark_uncal.png`, `f_rawP_ark_platt.png` | P(target) vs α grids (CheXpert, scaled, LR) |
| `f_convention_effect.png` | scaled vs unscaled CFR (SD/SDXL, XRV-platt) |
| `f_frd.png` | FRD across backbone/conv/method |
| `f_cav_stability.png` | CheX↔NIH CAV cosine per pathology |
