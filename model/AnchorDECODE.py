"""AnchorDECODE: marker-guided estimation of spatial cell-type proportions.

Dependencies: NumPy, pandas, SciPy, scikit-learn and PyTorch.

Example
-------
>>> from AnchorDECODE import AnchorDECODE
>>> model = AnchorDECODE()
>>> result = model.train(expression, coords, marker_dict)
>>> proportions = result["prediction"]
>>> model.save("model.pt")
>>> restored = AnchorDECODE.load("model.pt")
>>> proportions = restored.predict(expression)

Expression must contain raw nonnegative protein intensities or counts, not
log-transformed values. Spatial calibration uses the whole target sample;
MLP preprocessing and training use spatial training folds only. Each sample
is fitted independently. predict/forward replay OOF predictions for the fitted
sample; call train on a new instance for a different sample.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import KMeans
import torch
from torch import nn

__all__ = [
    'AnchorDECODE', 'NonMarkerMLP', 'balanced_marker_spatial_seed',
    'preprocess_nonmarkers', 'train_mlp', 'clr_fusion', 'evaluate_proportions',
]


def _expression_frame(expression):
    """Validate raw expression and convert spot/protein identifiers to strings."""
    if not isinstance(expression, pd.DataFrame):
        raise TypeError('expression must be a spot × protein pandas DataFrame')
    x = expression.copy()
    x.index = x.index.map(str)
    x.columns = x.columns.map(str)
    if not x.index.is_unique or not x.columns.is_unique:
        raise ValueError('Spot and protein names must be unique')
    a = x.to_numpy()
    if x.empty or not np.issubdtype(a.dtype, np.number):
        raise ValueError('Expression must be a nonempty numeric matrix')
    if np.iscomplexobj(a) or not np.isfinite(a).all() or (a < 0).any():
        raise ValueError('Expression must be finite, real and nonnegative')
    if (a.sum(axis=1) <= 0).any():
        raise ValueError('Every spot must have a positive full-panel total')
    return x


def _coordinates_array(coords, index):
    if isinstance(coords, pd.DataFrame):
        if not coords.index.map(str).equals(index):
            raise ValueError('Expression and coordinate indices/order differ')
        coords = coords.to_numpy()
    xy = np.asarray(coords, dtype=np.float64)
    if xy.ndim != 2 or xy.shape != (len(index), 2) or not np.isfinite(xy).all():
        raise ValueError('coords must be finite [n_spots, 2] coordinates')
    return xy


def _normalize_rows(values, eps):
    a = np.maximum(np.nan_to_num(np.asarray(values, dtype=np.float64),
                                nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    total = a.sum(axis=1, keepdims=True)
    p = a / np.maximum(total, eps)
    p[total[:, 0] <= eps] = 1.0 / a.shape[1]
    return p


def _validate_seed_parameters(parameters):
    if not all(np.isfinite(v) for v in parameters.values()):
        raise ValueError('Seed parameters must be finite')
    if not 0 <= parameters['marker_background_quantile'] < parameters['marker_anchor_quantile'] <= 1:
        raise ValueError('Require 0 <= background quantile < anchor quantile <= 1')
    if parameters['marker_evidence_power'] <= 0:
        raise ValueError('marker_evidence_power must be positive')
    if not 0 <= parameters['marker_agreement_threshold'] <= 1:
        raise ValueError('marker_agreement_threshold must be in [0, 1]')
    if not isinstance(parameters['spatial_k'], (int, np.integer)) or parameters['spatial_k'] < 1:
        raise ValueError('spatial_k must be a positive integer')
    if not 0 <= parameters['spatial_strength'] <= 1:
        raise ValueError('spatial_strength must be in [0, 1]')
    if min(parameters['spatial_distance_temperature'], parameters['spatial_marker_temperature']) <= 0:
        raise ValueError('Spatial temperatures must be positive')


def balanced_marker_spatial_seed(expression, coords, marker_dict, *,
                                 marker_background_quantile=.3, marker_anchor_quantile=1.,
                                 marker_evidence_power=1., marker_agreement_threshold=.1,
                                 spatial_k=8, spatial_strength=.3,
                                 spatial_distance_temperature=2., spatial_marker_temperature=.5):
    """Compute balanced marker proportions with spatial smoothing.

    Defaults use full-panel totals, Q30 background, max-Q30 scaling, power=1,
    agreement threshold=0.1, and 8 neighbors with smoothing strength=0.3.
    Distance/marker temperatures are 2.0/0.5; epsilon is 1e-6.
    Shared markers are excluded; each cell type requires a unique panel marker.
    Marker confidence/weight annotations do not affect seed aggregation.
    Returns seed arrays, calibration, reliability, coverage and parameters.
    """
    _validate_seed_parameters(dict(
        marker_background_quantile=marker_background_quantile,
        marker_anchor_quantile=marker_anchor_quantile,
        marker_evidence_power=marker_evidence_power,
        marker_agreement_threshold=marker_agreement_threshold, spatial_k=spatial_k,
        spatial_strength=spatial_strength, spatial_distance_temperature=spatial_distance_temperature,
        spatial_marker_temperature=spatial_marker_temperature))
    frame = _expression_frame(expression)
    xy = _coordinates_array(coords, frame.index)
    x = frame.to_numpy()
    proteins = frame.columns.tolist()
    if not marker_dict or any(not isinstance(k, str) for k in marker_dict):
        raise ValueError('marker_dict must have nonempty string cell-type keys')
    cell_types = list(marker_dict)
    lookup = {p: i for i, p in enumerate(proteins)}
    mask = np.zeros((len(cell_types), len(proteins)), dtype=np.float32)
    requested, cleaned, coverage_rows = {}, {}, []
    for k, cell_type in enumerate(cell_types):
        entries = marker_dict[cell_type]
        if isinstance(entries, (str, dict)):
            raise TypeError('Each marker list must be a sequence of names or dictionaries')
        names = []
        for entry in entries:
            if isinstance(entry, str):
                name = entry
            elif isinstance(entry, dict) and 'protein' in entry:
                name = str(entry['protein'])
                default = 1.0 if str(entry.get('confidence', 'medium')).lower() == 'high' else .75
                weight = float(entry.get('weight', default))
                if not np.isfinite(weight) or weight <= 0:
                    raise ValueError(f'Invalid marker weight: {cell_type}/{name}')
            else:
                raise TypeError('Markers must be names or dictionaries with a protein field')
            if name not in names:
                names.append(name)
        requested[cell_type] = names
        cleaned[cell_type] = [p for p in names if p in lookup]
        mask[k, [lookup[p] for p in cleaned[cell_type]]] = 1
        coverage_rows.append(dict(cell_type=cell_type, n_requested=len(names),
                                  missing_markers=';'.join(p for p in names if p not in lookup)))
    shared = set(np.asarray(proteins)[mask.sum(axis=0) > 1])
    mask[:, mask.sum(axis=0) > 1] = 0
    for k, ct in enumerate(cell_types):
        excluded = [p for p in cleaned[ct] if p in shared]
        cleaned[ct] = [p for p in cleaned[ct] if p not in shared]
        coverage_rows[k].update(n_present=len(cleaned[ct]),
                                coverage=len(cleaned[ct]) / max(len(requested[ct]), 1),
                                present_markers=';'.join(cleaned[ct]),
                                shared_markers_excluded=';'.join(excluded))
    if np.any(mask.sum(axis=1) == 0):
        missing = [cell_types[k] for k in np.flatnonzero(mask.sum(axis=1) == 0)]
        raise ValueError(f'Cell types without a unique panel marker: {missing}')

    eps = 1e-6
    union = mask.any(axis=0)
    full_density = x.sum(axis=1)
    marker_density = x[:, union].sum(axis=1)
    positive = full_density[full_density > eps]
    fallback = float(np.median(positive)) if positive.size else 1.0
    density = np.where(full_density > eps, full_density, fallback)
    normalized = x / density[:, None]  # Full-panel normalization precedes Q30 calibration.
    activation = np.zeros_like(x, dtype=np.float64)
    calibration_rows = []
    for p in np.flatnonzero(union):
        values = normalized[:, p]
        floor = float(np.quantile(values, marker_background_quantile))
        anchor = float(np.quantile(values, marker_anchor_quantile))
        scale = max(anchor - floor, eps)
        calibrated = np.clip((values - floor) / scale, 0, 1)
        if marker_evidence_power != 1.:
            calibrated = calibrated ** marker_evidence_power
        activation[:, p] = calibrated
        calibration_rows.append(dict(protein=proteins[p], density_mode='balanced',
                                     background=floor, anchor=anchor, dynamic_range=scale,
                                     active_fraction=float(np.mean(calibrated > 0))))

    evidence = np.zeros((len(x), len(cell_types)), dtype=np.float64)
    reliability_rows = []
    for k, ct in enumerate(cell_types):
        indices = np.flatnonzero(mask[k] > 0)
        values = activation[:, indices]
        if len(indices) == 1:
            agreement = np.ones(1)
            retained = np.ones(1, dtype=bool)
        else:
            ranks = pd.DataFrame(values).rank(axis=0, method='average').to_numpy(dtype=np.float64)
            with np.errstate(invalid='ignore', divide='ignore'):
                correlation = np.corrcoef(ranks, rowvar=False)
            correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
            np.fill_diagonal(correlation, 1.0)
            agreement = np.maximum(correlation - np.eye(len(indices)), 0).sum(axis=1)
            agreement /= max(len(indices) - 1, 1)
            retained = agreement >= marker_agreement_threshold
            if not retained.any():
                retained[:] = True  # Retain all markers if none pass the agreement threshold.
        weights = retained.astype(np.float64)
        weights /= weights.sum()
        evidence[:, k] = values @ weights
        for j, p in enumerate(indices):
            reliability_rows.append(dict(cell_type=ct, protein=proteins[p],
                                         agreement=float(agreement[j]), retained=bool(retained[j]),
                                         weight=float(weights[j])))

    before = _normalize_rows(evidence + eps, eps)
    abundance = before.copy()
    spatial = dict(enabled=False, mean_l1_shift=0.0)
    if len(x) > 1 and spatial_strength > 0:
        k = min(spatial_k, len(x) - 1)
        distances, neighbors = cKDTree(xy).query(xy, k=k + 1)
        distances, neighbors = distances[:, 1:], neighbors[:, 1:]
        positive_distance = distances[distances > 0]
        distance_scale = float(np.median(positive_distance)) if positive_distance.size else 1.0
        difference = np.abs(before[:, None, :] - before[neighbors]).sum(axis=2)
        weights = np.exp(-distances / (spatial_distance_temperature * distance_scale + eps))
        weights *= np.exp(-difference / spatial_marker_temperature)
        weights /= np.maximum(weights.sum(axis=1, keepdims=True), eps)
        neighbor_abundance = np.sum(before[neighbors] * weights[:, :, None], axis=1)
        abundance = _normalize_rows((1 - spatial_strength) * before + spatial_strength * neighbor_abundance, eps)
        spatial = dict(enabled=True, k=k, strength=spatial_strength, distance_scale=distance_scale,
                       mean_l1_shift=float(np.mean(abs(abundance - before).sum(axis=1))))
    parameters = dict(density_mode='balanced', marker_background_quantile=marker_background_quantile,
                      marker_anchor_quantile=marker_anchor_quantile, marker_evidence_power=marker_evidence_power,
                      marker_agreement_threshold=marker_agreement_threshold, marker_epsilon=eps, spatial_k=spatial_k,
                      spatial_strength=spatial_strength, spatial_distance_temperature=spatial_distance_temperature,
                      spatial_marker_temperature=spatial_marker_temperature)
    diagnostics = dict(density_mode='balanced', background_quantile=marker_background_quantile,
                       marker_panel_fraction=float(union.mean()), n_markers=int(union.sum()),
                       full_library_quantiles=np.quantile(full_density, [0, .25, .5, .75, 1]).tolist(),
                       marker_library_quantiles=np.quantile(marker_density, [0, .25, .5, .75, 1]).tolist(),
                       spatial=spatial)
    return dict(B_seed=abundance, before_spatial=before, activation=activation, evidence=evidence,
                calibration=pd.DataFrame(calibration_rows), reliability=pd.DataFrame(reliability_rows),
                coverage=pd.DataFrame(coverage_rows), cell_types=cell_types,
                requested_markers=requested, cleaned_markers=cleaned,
                parameters=parameters, diagnostics=diagnostics)


def preprocess_nonmarkers(raw, train_indices=None, statistics=None):
    """Fit positive-median/asinh/standardization on training rows, or apply saved statistics."""
    a = np.asarray(raw, dtype=np.float32)
    if statistics is None:
        if train_indices is None or len(train_indices) == 0:
            raise ValueError('Nonempty training indices are required to fit preprocessing')
        scale = np.ones(a.shape[1], dtype=np.float32)
        for j in range(a.shape[1]):
            positive = a[train_indices, j][a[train_indices, j] > 0]
            scale[j] = np.median(positive) if len(positive) else 1
        z = np.arcsinh(a / np.maximum(scale, 1e-6))
        mean = z[train_indices].mean(axis=0)
        sd = np.maximum(z[train_indices].std(axis=0), 1e-4)
        statistics = dict(scale=scale, mean=mean, sd=sd)
    else:
        z = np.arcsinh(a / np.maximum(statistics['scale'], 1e-6))
    return ((z - statistics['mean']) / statistics['sd']).astype(np.float32), statistics


class NonMarkerMLP(nn.Module):
    """Map preprocessed non-marker features to cell-type proportions."""
    def __init__(self, n_features, n_cell_types, hidden_dim=64, dropout=.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(), nn.Linear(hidden_dim, n_cell_types),
        )

    def forward(self, x):
        return self.net(x).softmax(dim=1)


def train_mlp(x_train, seed_train, *, random_seed=42, epochs=180, lr=1e-3,
              weight_decay=1e-4, hidden_dim=64, dropout=.1, device='cpu'):
    """Train a full-batch MLP against training-row seed proportions using MSE."""
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    model = NonMarkerMLP(x_train.shape[1], seed_train.shape[1], hidden_dim, dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    x = torch.tensor(x_train, dtype=torch.float32, device=device)
    target = torch.tensor(seed_train, dtype=torch.float32, device=device)
    history = []
    for epoch in range(epochs):
        model.train()
        prediction = model(x)
        loss = (prediction - target).square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite weak-supervision loss')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach().cpu()))
    model.eval()
    return model, history


def clr_fusion(seed, student, weight=.5, eps=1e-8):
    """Compute CLR^{-1}((1-weight)*CLR(seed) + weight*CLR(student))."""
    if not 0 <= weight <= 1:
        raise ValueError('Fusion weight must be in [0, 1]')
    if np.shape(seed) != np.shape(student):
        raise ValueError('Seed and student shapes differ')
    log_seed = np.log(np.maximum(seed, eps))
    log_student = np.log(np.maximum(student, eps))
    z = ((1 - weight) * (log_seed - log_seed.mean(axis=1, keepdims=True))
         + weight * (log_student - log_student.mean(axis=1, keepdims=True)))
    z -= z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


class AnchorDECODE:
    """Estimate cell-type proportions using a marker seed and OOF MLP ensemble.

    Defaults: 5 spatial folds, seeds (42, 43, 44), 180 epochs, 64 hidden units,
    dropout=0.1, AdamW lr=1e-3/weight_decay=1e-4, CLR fusion weight=0.5.
    The default ensemble contains 15 networks. CPU is the default device.

    train/fit return a dictionary of DataFrames: before_spatial, B_seed,
    student_oof and prediction. Fitted networks, preprocessing statistics and
    seed diagnostics are available as models_, preprocessing_ and seed_details_.

    This estimator is not an nn.Module; NonMarkerMLP implements the trainable
    PyTorch network. Its forward accepts a tensor of preprocessed features.
    """
    def __init__(self, *, epochs=180, lr=1e-3, weight_decay=1e-4, hidden_dim=64,
                 dropout=.1, n_splits=5, seeds=(42, 43, 44), fusion_lambda=.5,
                 device='cpu', num_threads=2, verbose=True,
                 marker_background_quantile=.3, marker_anchor_quantile=1.,
                 marker_evidence_power=1., marker_agreement_threshold=.1,
                 spatial_k=8, spatial_strength=.3,
                 spatial_distance_temperature=2., spatial_marker_temperature=.5):
        if epochs < 1 or n_splits < 2 or hidden_dim < 1 or num_threads < 1:
            raise ValueError('epochs/hidden_dim/num_threads >= 1 and n_splits >= 2 are required')
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError('seeds must be nonempty and distinct')
        if not 0 <= fusion_lambda <= 1 or not 0 <= dropout < 1 or lr <= 0 or weight_decay < 0:
            raise ValueError('Invalid optimizer, dropout or fusion settings')
        self.config = dict(epochs=epochs, lr=lr, weight_decay=weight_decay, hidden_dim=hidden_dim,
                           dropout=dropout, n_splits=n_splits, seeds=list(seeds),
                           fusion_lambda=fusion_lambda, device=str(device),
                           num_threads=num_threads, verbose=verbose)
        seed_parameters = dict(marker_background_quantile=marker_background_quantile,
                               marker_anchor_quantile=marker_anchor_quantile,
                               marker_evidence_power=marker_evidence_power,
                               marker_agreement_threshold=marker_agreement_threshold,
                               spatial_k=spatial_k, spatial_strength=spatial_strength,
                               spatial_distance_temperature=spatial_distance_temperature,
                               spatial_marker_temperature=spatial_marker_temperature)
        _validate_seed_parameters(seed_parameters)
        self.config.update(seed_parameters)
        self.models_ = []

    def train(self, expression, coords, marker_dict):
        """Fit one target sample without ground-truth proportions.

        expression: spot-by-protein DataFrame of raw nonnegative measurements.
        coords: matching-index DataFrame or array with shape (n_spots, 2).
        marker_dict: {cell_type: [protein_name, ...]}; entries may also be
            dictionaries with a "protein" field. Key order defines output columns.

        Returns before_spatial, B_seed, student_oof and prediction DataFrames.
        Spot/protein identifiers are converted to strings. Repeated calls refit
        the ensemble; fit is an alias and returns the same result dictionary."""
        self.models_ = []  # Refit from scratch.
        cfg = self.config
        torch.set_num_threads(cfg['num_threads'])
        x = _expression_frame(expression)
        xy = _coordinates_array(coords, x.index)
        if len(np.unique(xy, axis=0)) < cfg['n_splits']:
            raise ValueError('Not enough distinct coordinates for spatial folds')
        seed_keys = ('marker_background_quantile', 'marker_anchor_quantile',
                     'marker_evidence_power', 'marker_agreement_threshold', 'spatial_k',
                     'spatial_strength', 'spatial_distance_temperature', 'spatial_marker_temperature')
        self.seed_details_ = balanced_marker_spatial_seed(x, xy, marker_dict,
                                                        **{key: cfg[key] for key in seed_keys})
        self.spot_names_ = x.index.tolist()
        self.protein_names_ = x.columns.tolist()
        self.cell_types_ = self.seed_details_['cell_types']
        self.expression_hash_ = hashlib.sha256(x.to_numpy().tobytes(order='C')).hexdigest()
        self.coordinates_ = xy
        # Exclude every requested panel marker, including shared or rejected markers.
        marker_union = {p for names in self.seed_details_['requested_markers'].values() for p in names}
        self.nonmarker_indices_ = np.array([j for j, p in enumerate(x.columns) if p not in marker_union])
        if len(self.nonmarker_indices_) == 0:
            raise ValueError('No non-marker proteins remain for the MLP')
        standardized_xy = (xy - xy.mean(axis=0)) / np.maximum(xy.std(axis=0), 1e-8)
        self.fold_ids_ = KMeans(cfg['n_splits'], random_state=1701, n_init=20).fit_predict(standardized_xy)
        if len(np.unique(self.fold_ids_)) != cfg['n_splits']:
            raise ValueError('KMeans did not produce the requested number of nonempty folds')
        raw = x.to_numpy(dtype=np.float32)[:, self.nonmarker_indices_]
        b = self.seed_details_['B_seed'].astype(np.float32)
        self.preprocessing_ = []
        cached = []
        for fold in range(cfg['n_splits']):
            tr = np.flatnonzero(self.fold_ids_ != fold)
            te = np.flatnonzero(self.fold_ids_ == fold)
            features, statistics = preprocess_nonmarkers(raw, train_indices=tr)
            self.preprocessing_.append(statistics)
            cached.append((tr, te, features))

        per_seed, history_rows = [], []
        for random_seed in cfg['seeds']:
            oof = np.zeros_like(b)
            models = []
            for fold, (tr, te, features) in enumerate(cached):
                model, losses = train_mlp(
                    features[tr], b[tr], random_seed=random_seed, epochs=cfg['epochs'],
                    lr=cfg['lr'], weight_decay=cfg['weight_decay'], hidden_dim=cfg['hidden_dim'],
                    dropout=cfg['dropout'], device=cfg['device'])
                with torch.no_grad():
                    xt = torch.tensor(features[te], device=cfg['device'])
                    oof[te] = model(xt).cpu().numpy()
                models.append(model)
                history_rows.extend(dict(seed=random_seed, fold=fold, epoch=i + 1, weak_mse=v)
                                    for i, v in enumerate(losses))
            self.models_.append(models)
            per_seed.append(oof)
            if cfg['verbose']:
                print(f'Completed seed {random_seed}: {cfg["n_splits"]} spatial folds', flush=True)
        self.student_per_seed_ = np.stack(per_seed)
        student = self.student_per_seed_.mean(axis=0)
        # Use float64 for log-ratio fusion.
        prediction = clr_fusion(b.astype(np.float64), student.astype(np.float64), cfg['fusion_lambda'])
        self.history_ = pd.DataFrame(history_rows)
        values = dict(before_spatial=self.seed_details_['before_spatial'], B_seed=b,
                      student_oof=student, prediction=prediction)
        self.result_ = {name: pd.DataFrame(a, index=x.index, columns=self.cell_types_)
                        for name, a in values.items()}
        return self.result_

    fit = train

    def forward(self, expression):
        """Replay OOF inference and CLR fusion for the fitted expression matrix.

        Requires the same values, numeric dtype, identifiers and order as train.
        Returns a spot-by-cell-type DataFrame. predict is an alias."""
        if len(self.models_) != len(self.config['seeds']):
            raise RuntimeError('Call train() or load() first')
        x = _expression_frame(expression)
        if x.columns.tolist() != self.protein_names_ or x.index.tolist() != self.spot_names_:
            raise ValueError('OOF replay requires the original spot/protein IDs and order')
        if hashlib.sha256(x.to_numpy().tobytes(order='C')).hexdigest() != self.expression_hash_:
            raise ValueError('Input differs from the fitted slice; train a new model for a new slice')
        raw = x.to_numpy(dtype=np.float32)[:, self.nonmarker_indices_]
        predictions = np.zeros((len(self.models_), len(x), len(self.cell_types_)), dtype=np.float32)
        for fold, statistics in enumerate(self.preprocessing_):
            te = np.flatnonzero(self.fold_ids_ == fold)
            features, _ = preprocess_nonmarkers(raw, statistics=statistics)
            xt = torch.tensor(features[te], device=self.config['device'])
            for s, models in enumerate(self.models_):
                models[fold].eval()
                with torch.no_grad():
                    predictions[s, te] = models[fold](xt).cpu().numpy()
        p = clr_fusion(self.result_['B_seed'].to_numpy(dtype=np.float64),
                       predictions.mean(axis=0).astype(np.float64), self.config['fusion_lambda'])
        return pd.DataFrame(p, index=x.index, columns=self.cell_types_)

    predict = forward

    def save(self, path):
        """Save all networks, preprocessing, seed calibration, folds and predictions in one .pt file."""
        if len(self.models_) != len(self.config['seeds']):
            raise RuntimeError('No complete trained ensemble to save')
        details = self.seed_details_
        state = dict(format_version=1, config=self.config, spot_names=self.spot_names_,
                     protein_names=self.protein_names_, cell_types=self.cell_types_,
                     expression_hash=self.expression_hash_,
                     coordinates=torch.from_numpy(self.coordinates_),
                     nonmarker_indices=torch.from_numpy(self.nonmarker_indices_),
                     fold_ids=torch.from_numpy(self.fold_ids_),
                     preprocessing=[{k: torch.from_numpy(v) for k, v in s.items()} for s in self.preprocessing_],
                     models=[[{k: v.detach().cpu() for k, v in m.state_dict().items()} for m in models]
                             for models in self.models_],
                     seed_arrays={k: torch.from_numpy(details[k])
                                  for k in ('B_seed', 'before_spatial', 'activation', 'evidence')},
                     seed_tables={k: details[k].to_dict(orient='list')
                                  for k in ('calibration', 'reliability', 'coverage')},
                     seed_metadata={k: details[k] for k in ('cell_types', 'requested_markers',
                                    'cleaned_markers', 'parameters', 'diagnostics')},
                     results={k: torch.from_numpy(v.to_numpy().copy()) for k, v in self.result_.items()},
                     student_per_seed=torch.from_numpy(self.student_per_seed_),
                     history=self.history_.to_dict(orient='list'))
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, path)

    @classmethod
    def load(cls, path, device='cpu'):
        """Load a saved ensemble onto the requested device."""
        state = torch.load(path, map_location='cpu', weights_only=True)
        if state['format_version'] != 1:
            raise ValueError('Unsupported checkpoint format')
        model = cls(**dict(state['config'], device=str(device)))
        torch.set_num_threads(model.config['num_threads'])
        model.spot_names_, model.protein_names_ = state['spot_names'], state['protein_names']
        model.cell_types_ = state['cell_types']
        model.expression_hash_ = state['expression_hash']
        model.coordinates_ = state['coordinates'].numpy()
        model.nonmarker_indices_ = state['nonmarker_indices'].numpy()
        model.fold_ids_ = state['fold_ids'].numpy()
        model.preprocessing_ = [{k: v.numpy() for k, v in s.items()} for s in state['preprocessing']]
        model.seed_details_ = dict(state['seed_metadata'])
        model.seed_details_.update({k: v.numpy() for k, v in state['seed_arrays'].items()})
        model.seed_details_.update({k: pd.DataFrame(v) for k, v in state['seed_tables'].items()})
        model.result_ = {k: pd.DataFrame(v.numpy(), index=model.spot_names_, columns=model.cell_types_)
                         for k, v in state['results'].items()}
        model.student_per_seed_ = state['student_per_seed'].numpy()
        model.history_ = pd.DataFrame(state['history'])
        for states in state['models']:
            models = []
            for weights in states:
                net = NonMarkerMLP(len(model.nonmarker_indices_), len(model.cell_types_),
                                   model.config['hidden_dim'], model.config['dropout']).to(device)
                net.load_state_dict(weights)
                net.eval()
                models.append(net)
            model.models_.append(models)
        return model

    def export(self, directory):
        """Export model.pt, prediction/audit CSVs, parameters and SHA256 hashes."""
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        files = []
        self.save(out / 'model.pt')
        files.append('model.pt')
        tables = dict(self.result_)
        tables.update({k: self.seed_details_[k] for k in ('calibration', 'reliability', 'coverage')})
        tables['history'] = self.history_
        for name, frame in tables.items():
            frame.to_csv(out / f'{name}.csv', index=name in self.result_)
            files.append(f'{name}.csv')
        for s, seed in enumerate(self.config['seeds']):
            name = f'student_seed{seed}.csv'
            pd.DataFrame(self.student_per_seed_[s], index=self.spot_names_, columns=self.cell_types_).to_csv(out / name)
            files.append(name)
        pd.DataFrame({'fold': self.fold_ids_}, index=self.spot_names_).to_csv(out / 'spatial_folds.csv')
        pd.DataFrame({'protein': self.protein_names_, 'is_nonmarker':
                      np.isin(np.arange(len(self.protein_names_)), self.nonmarker_indices_)}).to_csv(
                          out / 'features.csv', index=False)
        files.extend(['spatial_folds.csv', 'features.csv'])
        metadata = dict(config=self.config, seed=self.seed_details_['parameters'],
                        diagnostics=self.seed_details_['diagnostics'],
                        requested_markers=self.seed_details_['requested_markers'],
                        cleaned_markers=self.seed_details_['cleaned_markers'],
                        expression_hash=self.expression_hash_,
                        scope='within-slice spatial OOF; seed calibrated on the whole target slice',
                        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        (out / 'parameters.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + '\n')
        files.append('parameters.json')
        lock = dict(truth_used=False, n_models=sum(map(len, self.models_)),
                    files={name: hashlib.sha256((out / name).read_bytes()).hexdigest() for name in files})
        (out / 'predictions_lock.json').write_text(json.dumps(lock, indent=2) + '\n')


def evaluate_proportions(prediction, truth):
    """Compute proportion metrics; call explicitly after fixing predictions and settings."""
    if isinstance(prediction, pd.DataFrame) and isinstance(truth, pd.DataFrame):
        if not prediction.index.equals(truth.index) or not prediction.columns.equals(truth.columns):
            raise ValueError('Prediction and truth labels/order differ')
    p, y = np.asarray(prediction, dtype=float), np.asarray(truth, dtype=float)
    if p.ndim != 2 or p.shape != y.shape or p.size == 0:
        raise ValueError('Expected matching nonempty [spots, cell types] arrays')
    if not np.isfinite(p).all() or not np.isfinite(y).all() or (p < 0).any() or (y < 0).any():
        raise ValueError('Proportions must be finite and nonnegative')
    if not np.allclose(p.sum(1), 1, atol=1e-6) or not np.allclose(y.sum(1), 1, atol=1e-6):
        raise ValueError('Proportion rows must sum to one')
    denominator = p.var() + y.var() + (p.mean() - y.mean()) ** 2
    ccc = 2 * np.mean((p - p.mean()) * (y - y.mean())) / denominator if denominator > 0 else np.nan
    per_type = []
    for j in range(p.shape[1]):
        a, b = p[:, j], y[:, j]
        den = a.var() + b.var() + (a.mean() - b.mean()) ** 2
        per_type.append(2 * np.mean((a - a.mean()) * (b - b.mean())) / den if den > 0 else np.nan)
    variable = p.std() > 0 and y.std() > 0 and p.size >= 2
    return dict(pooled_CCC=float(ccc), macro_CCC=float(np.nanmean(per_type)),
                MAE=float(abs(p - y).mean()), RMSE=float(np.sqrt(((p - y) ** 2).mean())),
                Pearson=float(pearsonr(p.ravel(), y.ravel()).statistic) if variable else np.nan,
                Spearman=float(spearmanr(p.ravel(), y.ravel()).statistic) if variable else np.nan,
                Top1=float(np.mean(p.argmax(1) == y.argmax(1))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--expression', required=True, help='Raw spot × protein CSV, spot IDs in first column')
    parser.add_argument('--coords', required=True, help='Same-order spot × 2 coordinate CSV')
    parser.add_argument('--markers', required=True, help='JSON: {cell_type: [protein, ...]}')
    parser.add_argument('--out', required=True)
    parser.add_argument('--epochs', type=int, default=180)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    # CSV inputs use float32; the Python API preserves the input seed precision.
    expression = pd.read_csv(args.expression, index_col=0).astype(np.float32)
    coordinates = pd.read_csv(args.coords, index_col=0).astype(np.float32)
    markers = json.loads(Path(args.markers).read_text())
    estimator = AnchorDECODE(epochs=args.epochs, device=args.device)
    estimator.train(expression, coordinates, markers)
    estimator.export(args.out)
    print(f'Saved predictions and all fold models to {args.out}')
