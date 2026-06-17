# Data & model weights

This pack does not redistribute datasets or classifier weights. Obtain each
below and point the scripts at them with the corresponding flag/env var.

## CheXpert

Source: Stanford AIMI (https://stanfordaimi.azurewebsites.net/) — the
`CheXpert-v1.0-small` release.

Expected layout (point `--chexpert-root` / `$CXR_CHEXPERT_ROOT` at the dir that
*contains* `CheXpert-v1.0-small/`):

```
<chexpert-root>/
  CheXpert-v1.0-small/
    train.csv
    valid.csv
    train/ ...   # patient image folders referenced by the CSV "Path" column
    valid/ ...
```

Default: `/data/chexpert`.

## NIH ChestX-ray14

Source: NIH Clinical Center (https://nihcc.app.box.com/v/ChestXray-NIHCC).

- `--nih-root` / `$CXR_NIH_ROOT`: directory holding the extracted PNG images
  (filenames match the `Image Index` column). Default `/data/nih_raw`.
- `--nih-csv` / `$CXR_NIH_CSV`: `Data_Entry_2017_v2020.csv`. Defaults to
  `<nih-root>/Data_Entry_2017_v2020.csv`.
- `--nih-bbox` / `$CXR_NIH_BBOX`: `BBox_List_2017.csv`. A copy is bundled at
  `metadata/BBox_List_2017.csv` and used by default (the IoU phase only).

## Ark+ classifier weights

The Ark+ SwinL768 ChestX-ray14 fine-tuned checkpoint
(`Ark+SwinL768_ChestX-ray14_ft.pth.tar`). Obtain from the Ark+ project
(https://github.com/jlianglab/Ark) and pass via `--ark-weights` /
`$CXR_ARK_WEIGHTS`. Required for the `calib`, `response`, `frd`, and `iou`
phases; the harness exits with a clear message if it is missing.

## Auto-downloaded weights

Fetched automatically into `--torch-home` / `$TORCH_HOME` (default
`/tmp/torch_cache`) on first use:

- XRV DenseNet classifier — `densenet121-res224-all` (torchxrayvision)
- XRV ResNet autoencoder (AE backbone) — `101-elastic` (torchxrayvision)
- SD-1.5 VAE — `stable-diffusion-v1-5/stable-diffusion-v1-5` (diffusers)
- SDXL VAE — `stabilityai/sdxl-vae` (diffusers)
