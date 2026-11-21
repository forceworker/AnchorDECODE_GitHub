"""Run AnchorDECODE on the deposited SPOTS spleen replicate 1 files.

Download instructions: docs/tutorial.md. Run from the repository root:
    python examples/spots.py --data-dir data/SPOTS --out results/SPOTS
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tarfile

import h5py
import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from model import AnchorDECODE

MATRIX_FILE = "GSE198353_spleen_rep_1_filtered_feature_bc_matrix.h5"
SPATIAL_FILE = "GSE198353_spleen_replicate_1_spatial.tar.gz"
MARKER_FILE = Path(__file__).with_name("spots_markers.json")


def load_spots(data_dir):
    """Return raw spot-by-protein counts and aligned full-resolution (x, y).

    Select Antibody Capture features from the deposited 10x matrix; keep its
    barcode and feature order. Read coordinates directly from the tar archive.
    No normalization, logarithm, additional spot filtering or imputation.
    """
    data_dir = Path(data_dir)
    for name in (MATRIX_FILE, SPATIAL_FILE):
        if not (data_dir / name).is_file():
            raise FileNotFoundError(
                f"Missing {data_dir / name}. Download the two GEO files listed "
                "in docs/tutorial.md before running this example."
            )
    with h5py.File(data_dir / MATRIX_FILE, "r") as handle:
        group = handle["matrix"]
        matrix = sparse.csc_matrix(
            (group["data"][:], group["indices"][:], group["indptr"][:]),
            shape=tuple(group["shape"][:]),
        )
        barcodes = group["barcodes"].asstr()[:]
        features = group["features"]
        keep = features["feature_type"].asstr()[:] == "Antibody Capture"
        proteins = features["id"].asstr()[:][keep]
        if not keep.any():
            raise ValueError("The input has no Antibody Capture features")
        expression = pd.DataFrame(
            matrix[keep, :].T.toarray(), index=barcodes, columns=proteins
        )
    expression.index.name = "spot_id"
    if not expression.index.is_unique or not expression.columns.is_unique:
        raise ValueError("Spot and protein identifiers must be unique")

    columns = ["barcode", "in_tissue", "array_row", "array_col",
               "pxl_row_in_fullres", "pxl_col_in_fullres"]
    with tarfile.open(data_dir / SPATIAL_FILE, "r:gz") as archive:
        members = [m for m in archive.getmembers() if m.isfile()
                   and Path(m.name).name == "tissue_positions_list.csv"]
        if len(members) != 1:
            raise ValueError("Expected one tissue_positions_list.csv in spatial archive")
        with archive.extractfile(members[0]) as stream:
            positions = pd.read_csv(stream, header=None, names=columns,
                                    dtype={"barcode": str}).set_index("barcode")
    if not positions.index.is_unique:
        raise ValueError("Spatial metadata contains duplicate barcodes")
    missing = expression.index.difference(positions.index)
    if len(missing):
        raise ValueError(f"Spatial metadata is missing {len(missing)} matrix barcodes")
    positions = positions.loc[expression.index]
    if not positions["in_tissue"].eq(1).all():
        raise ValueError("The filtered matrix contains off-tissue spots")
    coords = positions[["pxl_col_in_fullres", "pxl_row_in_fullres"]].copy()
    coords.columns = ["x", "y"]
    coords.index = expression.index.copy()
    return expression, coords


def plot_proportions(prediction, coords, output_path):
    """Save maps with a common 0–1 color scale in image coordinate orientation."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(prediction.columns), figsize=(16, 4),
                             squeeze=False, layout="constrained")
    for ax, cell_type in zip(axes.flat, prediction.columns):
        points = ax.scatter(coords["x"], coords["y"], c=prediction[cell_type],
                            s=5, cmap="viridis", vmin=0, vmax=1, linewidths=0)
        ax.set_title(cell_type, fontsize=10)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.axis("off")
    fig.colorbar(points, ax=axes.ravel().tolist(), label="Estimated proportion",
                 shrink=0.65)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/SPOTS")
    parser.add_argument("--out", type=Path, default=ROOT / "results/SPOTS")
    parser.add_argument("--epochs", type=int, default=180,
                        help="Epochs per network; the study setting is 180")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    args = parser.parse_args()
    expression, coords = load_spots(args.data_dir)
    markers = json.loads(MARKER_FILE.read_text(encoding="utf-8"))
    missing = {p for entries in markers.values() for p in entries} - set(expression.columns)
    if missing:
        raise ValueError(f"Example markers absent from the protein panel: {sorted(missing)}")
    print(f"Input: {expression.shape[0]} spots × {expression.shape[1]} proteins", flush=True)
    model = AnchorDECODE(epochs=args.epochs, device=args.device)
    prediction = model.train(expression, coords, markers)["prediction"]
    values = prediction.to_numpy()
    if not np.isfinite(values).all() or (values < 0).any():
        raise RuntimeError("Invalid predicted proportions")
    np.testing.assert_allclose(values.sum(axis=1), 1.0, rtol=0, atol=1e-10)
    model.export(args.out)
    coords.to_csv(args.out / "coordinates.csv")
    plot_proportions(prediction, coords, args.out / "proportions.png")
    print(f"Saved {prediction.shape[0]} × {prediction.shape[1]} proportions to {args.out.resolve()}")


if __name__ == "__main__":
    main()
