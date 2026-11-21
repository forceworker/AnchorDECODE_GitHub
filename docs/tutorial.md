# AnchorDECODE tutorial

This guide runs one complete SPOTS mouse spleen example, from public source files to spatial cell-type proportion maps. The [notebook](../tutorials/SPOTS_tutorial.ipynb) uses the same data reader and model. No data, pretrained checkpoints, or generated predictions are distributed with this repository.

## 1. Install

Open a terminal in the repository root. The recommended environment explicitly installs PyTorch 2.11.0 with CUDA 12.8 for NVIDIA GPUs on Linux or Windows:

```bash
conda env create -f environment.yml
conda activate anchordecode
```

Alternatively, install `requirements-gpu.txt` into a Python 3.11 environment using `python -m pip install -r requirements-gpu.txt`. Install JupyterLab separately (`python -m pip install jupyterlab`) if you want to use the notebook.

The core model requires NumPy, pandas, SciPy, scikit-learn, and PyTorch. The example additionally uses h5py to read deposited counts and matplotlib to draw maps. The CUDA build is selected from the [official PyTorch repository](https://pytorch.org/get-started/previous-versions/#v2110); GPU execution also requires an NVIDIA GPU and a compatible driver.

Verify the installation:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda); print('CUDA available:', torch.cuda.is_available())"
```

The expected build is `2.11.0+cu128`, with CUDA runtime `12.8` and `CUDA available: True`. A GPU package alone does not select GPU execution: use `device="cuda"` in Python or `--device cuda` in the script. If CUDA is unavailable, check the driver and GPU visibility before starting a GPU run.

For CPU-only use, create a separate Python 3.11 environment and install:

```bash
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip install jupyterlab
```

Then use `device="cpu"` in the Python examples and `--device cpu` in script commands; change `DEVICE` in the notebook to `"cpu"` as well.

## 2. Obtain the data

The data are from [GEO GSE198353](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE198353), associated with the [SPOTS paper](https://doi.org/10.1038/s41587-022-01536-3). Download:

1. [GSE198353_spleen_rep_1_filtered_feature_bc_matrix.h5](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE198nnn/GSE198353/suppl/GSE198353_spleen_rep_1_filtered_feature_bc_matrix.h5)
2. [GSE198353_spleen_replicate_1_spatial.tar.gz](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE198nnn/GSE198353/suppl/GSE198353_spleen_replicate_1_spatial.tar.gz)

Create `data/SPOTS/` and place both files there, without unpacking the archive:

```text
data/SPOTS/
├── GSE198353_spleen_rep_1_filtered_feature_bc_matrix.h5
└── GSE198353_spleen_replicate_1_spatial.tar.gz
```

For a command-line download, run from the repository root:

```bash
mkdir -p data/SPOTS
curl -fL --retry 3 -o data/SPOTS/GSE198353_spleen_rep_1_filtered_feature_bc_matrix.h5 https://ftp.ncbi.nlm.nih.gov/geo/series/GSE198nnn/GSE198353/suppl/GSE198353_spleen_rep_1_filtered_feature_bc_matrix.h5
curl -fL --retry 3 -o data/SPOTS/GSE198353_spleen_replicate_1_spatial.tar.gz https://ftp.ncbi.nlm.nih.gov/geo/series/GSE198nnn/GSE198353/suppl/GSE198353_spleen_replicate_1_spatial.tar.gz
```

## 3. Read counts and coordinates

```python
from pathlib import Path
import json
from examples.spots import load_spots

expression, coords = load_spots(Path("data/SPOTS"))
markers = json.loads(Path("examples/spots_markers.json").read_text())
print(expression.shape)  # (2653, 21)
print(coords.shape)      # (2653, 2)
```

The HDF5 file contains both RNA and antibody counts. The reader selects only `Antibody Capture` features and preserves their original counts and identifiers. It obtains coordinates from `spatial/tissue_positions_list.csv` inside the spatial archive and matches them to the expression matrix by spot barcode. Coordinates are full-resolution image pixels: `x` is the image column, and `y` is the image row.

The resulting expression matrix contains 2,653 tissue spots and 21 proteins. All 21 proteins enter the model; seven requested marker proteins are excluded from the neural input, leaving 14 non-marker features. No additional filtering, normalization, logarithm, or imputation is applied before training.

## Marker definitions

The example's [marker file](../examples/spots_markers.json) contains the literature-screened markers used in the SPOTS experiment. Protein identifiers must match the expression columns exactly.

| Cell type | Protein identifier(s) | Supporting literature (PMID) |
| --- | --- | --- |
| B cells | `CD19`, `CD20` | [36593397](https://pubmed.ncbi.nlm.nih.gov/36593397/) |
| T cells | `CD3` | [36593397](https://pubmed.ncbi.nlm.nih.gov/36593397/) |
| Red pulp macrophages | `F4_80` | [30078711](https://pubmed.ncbi.nlm.nih.gov/30078711/), [35145087](https://pubmed.ncbi.nlm.nih.gov/35145087/) |
| Red pulp macrophages | `CD163` | [35145087](https://pubmed.ncbi.nlm.nih.gov/35145087/) |
| CD169+ macrophages | `CD169_siglec` | [30078711](https://pubmed.ncbi.nlm.nih.gov/30078711/) |
| Neutrophils | `Ly6G` | [31932813](https://pubmed.ncbi.nlm.nih.gov/31932813/) |

These markers define the modeled populations in the context of mouse spleen. The returned proportions describe the relative composition among these five modeled populations; they do not establish that every cell population in the tissue has been represented.

## 4. Fit the model

```python
from model import AnchorDECODE

model = AnchorDECODE(device="cuda")  # Experimental hyperparameters; GPU execution
result = model.train(expression, coords, markers)
prediction = result["prediction"]
print(prediction.head())
```

AnchorDECODE first obtains a marker-based estimate with spatial smoothing. A neural ensemble then learns to reproduce this estimate using non-marker proteins and spatially separated training folds. The ensemble's out-of-fold estimates are combined with the marker estimate by centered log-ratio fusion. Ground-truth proportions and external reference profiles are not inputs.

Each tissue section is fitted independently. Spatial folds govern neural training and preprocessing; marker calibration uses the entire target section. They therefore do not represent validation on an independent unseen tissue section.

To check your GPU installation quickly, use `AnchorDECODE(device="cuda", epochs=2)`; this abbreviated run is only a smoke test, not the experimental training setting.

## 5. Save predictions and draw maps

```python
from examples.spots import plot_proportions

output = Path("results/SPOTS")
model.export(output)
coords.to_csv(output / "coordinates.csv")
plot_proportions(prediction, coords, output / "proportions.png")
```

Each row of `prediction.csv` corresponds to a spot barcode, and the five cell-type columns follow the order in the marker JSON. Values are nonnegative and sum to one within each spot. The maps use a common 0–1 color scale and the original image orientation.

| Output | Contents |
| --- | --- |
| `prediction.csv` | Final estimated cell-type proportions |
| `proportions.png` | Spatial maps for the five cell types |
| `coordinates.csv` | Spot coordinates aligned to prediction rows |
| `model.pt` | Fitted networks, preprocessing, marker calibration, and predictions |
| `before_spatial.csv`, `B_seed.csv` | Marker estimates before and after spatial smoothing |
| `student_oof.csv`, `student_seed*.csv` | Ensemble and per-seed out-of-fold neural estimates |
| `coverage.csv`, `reliability.csv`, `calibration.csv` | Marker coverage, retention, and calibration summaries |
| `history.csv`, `spatial_folds.csv`, `features.csv` | Training losses, fold assignments, and neural feature membership |
| `parameters.json`, `predictions_lock.json` | Run settings, provenance, and output checksums |

Reload a fitted model and replay its predictions:

```python
restored = AnchorDECODE.load(output / "model.pt", device="cuda")
replayed = restored.predict(expression)
```

Replay requires the original expression values, numeric dtype, identifiers, and ordering. For another tissue section, construct a new `AnchorDECODE` instance and call `train()` on that section.

## Run everything from the command line

The following command performs the same loading, training, export, and plotting steps:

```bash
python examples/spots.py --data-dir data/SPOTS --out results/SPOTS --device cuda
```

For a brief installation test, choose a separate output directory:

```bash
python examples/spots.py --data-dir data/SPOTS --out results/SPOTS_smoke --epochs 2 --device cuda
```

## Model settings

The example retains the hyperparameters used in the SPOTS experiment. The table lists the defaults in `AnchorDECODE()`; the GPU tutorial explicitly overrides `device` to `"cuda"`.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `marker_background_quantile` | 0.3 | Background quantile for marker calibration |
| `marker_anchor_quantile` | 1.0 | Upper anchor quantile for marker calibration |
| `marker_evidence_power` | 1.0 | Marker activation exponent |
| `marker_agreement_threshold` | 0.1 | Marker consistency threshold |
| `spatial_k` | 8 | Number of spatial neighbors |
| `spatial_strength` | 0.3 | Spatial smoothing strength |
| `spatial_distance_temperature` | 2.0 | Distance weighting temperature |
| `spatial_marker_temperature` | 0.5 | Marker similarity weighting temperature |
| `n_splits` | 5 | Spatial folds per seed |
| `hidden_dim` | 64 | Width of each of the two hidden layers |
| `dropout` | 0.1 | Dropout rate |
| `lr` | 0.001 | Learning rate |
| `weight_decay` | 0.0001 | AdamW weight decay |
| `epochs` | 180 | Training epochs per network |
| `seeds` | `(42, 43, 44)` | Three ensemble seeds; 15 networks across five folds |
| `fusion_lambda` | 0.5 | Neural contribution to centered log-ratio fusion |
| `device` | `"cpu"` | Computation device |
| `num_threads` | 2 | PyTorch CPU thread count |
| `verbose` | `True` | Print progress after each ensemble seed |

The network uses two hidden layers, GELU activations, LayerNorm after the first linear layer, and a softmax output. Fit settings and marker diagnostics are exported for each run.

## Use your own spatial proteomics data

Prepare three inputs:

1. **Expression:** a numeric pandas DataFrame with spots or sampled tissue regions as rows and protein identifiers as columns. Use raw nonnegative counts or intensities, unique identifiers, finite values, and positive total abundance for every row. Retain the full measured panel, including non-marker proteins.
2. **Coordinates:** a two-column DataFrame with finite `x` and `y` values and the same row identifiers and order as expression. At least five distinct spatial coordinates are needed with the default fold setting.
3. **Markers:** a dictionary mapping cell types to lists of protein identifiers. Each cell type needs a measurable marker that is not assigned to another modeled cell type. At least one non-marker protein must remain for neural training. Marker coverage and consistency determine which requested markers contribute to the marker estimate.

From the repository root:

```python
import json
from pathlib import Path
import pandas as pd
from model import AnchorDECODE

expression = pd.read_csv("data/my_sample/expression.csv", index_col=0)
coords = pd.read_csv("data/my_sample/coordinates.csv", index_col=0)
coords = coords.loc[expression.index, ["x", "y"]]
markers = json.loads(Path("data/my_sample/markers.json").read_text())

model = AnchorDECODE(device="cuda")
prediction = model.train(expression, coords, markers)["prediction"]
model.export("results/my_sample")
```

For example, a marker file has the form `{"B cells": ["CD19", "CD20"], "T cells": ["CD3"]}`. Replace this with markers appropriate to your tissue, species, cell types, and measured protein panel. Do not transfer the SPOTS marker set unchanged to unrelated tissues.

If starting from an AnnData object, construct the expression DataFrame from its raw measurement matrix and the appropriate protein identifiers, and obtain coordinates from `obsm["spatial"]`. Ensure that the selected matrix has not already been log-transformed. AnnData support is optional and requires a separate installation of `anndata`.

## Troubleshooting

| Message or symptom | Action |
| --- | --- |
| Missing GEO input file | Check the exact filenames and `--data-dir`; leave the spatial archive compressed. |
| Marker absent from the panel | Match marker identifiers to protein columns, including case and punctuation. |
| No non-marker proteins remain | Provide the full measured protein panel. |
| Not enough distinct coordinates | Check the coordinates and choose a valid `n_splits` for your data. |
| CUDA unavailable | Use CPU or install a compatible CUDA-enabled PyTorch build. |
| Replay rejects the expression matrix | Use the original matrix with identical dtype, values, identifiers, and order; fit anew for a different section. |
