#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.ppmi_ssl_dataset import select_3t_manifest

ENDPOINTS = ["putamen_sbr_min", "NHY", "putamen_sbr_mean", "NP3TOT", "MCATOT", "caudate_sbr_mean"]
REGRESSION_METRICS = ["spearman_rho", "pearson_r", "r2", "mae", "rmse"]
ENDPOINT_METRICS = {e: REGRESSION_METRICS for e in ENDPOINTS}
ENDPOINT_METRICS["NHY"] = ["quadratic_weighted_kappa", "balanced_accuracy", "macro_f1", "hy_stage_mae"]
MRI_ONLY, MRI_COV, COV_ONLY = "MRI-only", "MRI + Cov.", "Cov.-only"
OUTER_DEVELOPMENT_SPLITS = ("train", "val")
FORBIDDEN_PROBE_FEATURES = set(ENDPOINTS) | {
    "total_LEDD", "disease_duration_from_onset_years", "PDSTATE",
    "tremor_score", "pigd_score", "td_pigd_subtype",
}
# Preserve the suite's grid order: the first candidate wins a numerical tie.
DEFAULT_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1., 10., 100., 1000., 10000.)
_WORKER_RIDGE_GRID = DEFAULT_GRID
_WORKER_LOGISTIC_GRID = DEFAULT_GRID
_WORKER_INNER_FOLDS = 5
_WORKER_RANDOM_STATE = 3407
_WORKER_HY_METRIC = "quadratic_weighted_kappa"
_WORKER_REQUIRE_CONVERGENCE = True
_WORKER_LOGISTIC_RETRY_ITER = 100000
_WORKER_LOGISTIC_RETRY_SOLVER = "newton-cg"


def first_column(frame: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    lower = {str(column).lower(): str(column) for column in frame.columns}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return None


def canonicalize_manifest(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    candidates = {
        "subject_id": ["subject_id", "PATNO", "patno", "participant_id"],
        "fold": ["fold", "outer_fold", "cv_fold"],
        "split": ["split", "partition", "set"],
        "class_label": ["class_label", "diagnosis", "cohort", "group"],
        "age": ["age", "age_at_visit", "AGE"],
        "sex": ["sex", "gender", "SEX"],
        "scanner_id": ["scanner_id", "scanner", "site_scanner", "ModelName"],
        "field_strength": ["MagneticFieldStrength", "field_strength", "magnetic_field_strength"],
        "putamen_sbr_min": ["putamen_sbr_min"],
        "putamen_sbr_mean": ["putamen_sbr_mean"],
        "caudate_sbr_mean": ["caudate_sbr_mean"],
        "NP3TOT": ["NP3TOT", "np3tot", "MDS_UPDRS_III", "updrs_iii"],
        "MCATOT": ["MCATOT", "mcatot", "MoCA", "moca_total"],
        "NHY": ["NHY", "nhy", "HY_stage", "hy_stage"],
        "t1j_path": ["t1j_crop_path", "t1jac_crop_path", "T1JAC_path", "T1J"],
    }
    out = frame.copy()
    mapping: dict[str, str] = {}
    for canonical, names in candidates.items():
        source = first_column(out, names)
        if source is not None:
            mapping[canonical] = source
            if source != canonical:
                out[canonical] = out[source]
    missing = [column for column in ["subject_id", "fold", "split", "class_label"] if column not in out.columns]
    if missing:
        raise ValueError(f"Manifest lacks required columns after robust mapping: {missing}")
    out["subject_id"] = out["subject_id"].astype("string").str.strip()
    out["fold"] = pd.to_numeric(out["fold"], errors="raise").astype(int)
    out["split"] = out["split"].astype("string").str.lower().replace({"valid": "val", "validation": "val"})
    for endpoint in ENDPOINTS:
        if endpoint in out.columns:
            out[endpoint] = pd.to_numeric(out[endpoint], errors="coerce")
    return out, mapping


def is_pd(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return numeric.eq(1)
    text = series.astype("string").str.upper()
    return text.str.contains(r"\bPD\b|PARKINSON", regex=True, na=False) & ~text.str.contains(
        r"HC|CONTROL|\bCN\b", regex=True, na=False
    )


def split_numeric_categorical(train: pd.DataFrame, features: Sequence[str]) -> tuple[list[str], list[str]]:
    numeric: list[str] = []
    categorical: list[str] = []
    for column in features:
        converted = pd.to_numeric(train[column], errors="coerce")
        if converted.notna().mean() >= 0.5:
            numeric.append(column)
        else:
            categorical.append(column)
    return numeric, categorical


def make_preprocessor(train: pd.DataFrame, features: Sequence[str]):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric, categorical = split_numeric_categorical(train, features)
    transformers = []
    if numeric:
        transformers.append(
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric)
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            )
        )
    if not transformers:
        raise RuntimeError("No usable feature columns")
    return ColumnTransformer(transformers)


def covariate_columns(frame: pd.DataFrame) -> list[str]:
    required = ["age", "sex", "scanner_id", "field_strength"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise RuntimeError(f"Cov.-only requires age, sex, scanner, and field strength; missing {missing}")
    return required


def metric_bundle(endpoint: str, y_true: Iterable[Any], y_pred: Iterable[Any]) -> dict[str, float]:
    yt = pd.to_numeric(pd.Series(y_true), errors="coerce").to_numpy(dtype=float)
    yp = pd.to_numeric(pd.Series(y_pred), errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[keep], yp[keep]
    if endpoint == "NHY":
        from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score

        if len(yt) == 0:
            return {metric: np.nan for metric in ENDPOINT_METRICS[endpoint]}
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
            balanced = balanced_accuracy_score(yt, yp)
        return {
            "quadratic_weighted_kappa": float(cohen_kappa_score(yt, yp, labels=[1.0, 2.0, 3.0], weights="quadratic"))
            if len(np.unique(yt)) > 1
            else np.nan,
            "balanced_accuracy": float(balanced),
            "macro_f1": float(f1_score(yt, yp, labels=[1.0, 2.0, 3.0], average="macro", zero_division=0)),
            "hy_stage_mae": float(np.mean(np.abs(yt - yp))),
        }
    if len(yt) < 2:
        return {metric: np.nan for metric in ENDPOINT_METRICS[endpoint]}
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    variable = np.std(yt) > 0 and np.std(yp) > 0
    return {
        "spearman_rho": float(spearmanr(yt, yp).statistic) if variable else np.nan,
        "pearson_r": float(pearsonr(yt, yp).statistic) if variable else np.nan,
        "r2": float(r2_score(yt, yp)),
        "mae": float(mean_absolute_error(yt, yp)),
        "rmse": float(math.sqrt(mean_squared_error(yt, yp))),
    }


def _stable_seed(*values: Any) -> int:
    payload = "|".join(str(value) for value in values).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little") % (2**31 - 1)


def _inner_assignment(
    development: pd.DataFrame,
    endpoint: str,
    outer_fold: int,
    requested_splits: int,
    random_state: int,
) -> tuple[np.ndarray, int, str]:
    from sklearn.model_selection import KFold, StratifiedKFold

    ordered = development.sort_values("subject_id").reset_index()
    y = pd.to_numeric(ordered[endpoint], errors="raise").to_numpy(dtype=float)
    if endpoint == "NHY":
        counts = pd.Series(y).value_counts()
        n_splits = min(int(requested_splits), int(counts.min()))
        if n_splits < 2:
            raise RuntimeError(
                f"NHY outer fold {outer_fold} cannot support at least two stratified inner folds: "
                f"{counts.to_dict()}"
            )
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=_stable_seed(random_state, endpoint, outer_fold, "inner"),
        )
        raw_splits = splitter.split(np.zeros(len(ordered)), y)
        splitter_name = "StratifiedKFold"
    else:
        n_splits = min(int(requested_splits), len(ordered))
        if n_splits < 2:
            raise RuntimeError(f"{endpoint} outer fold {outer_fold} has fewer than two development participants")
        splitter = KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=_stable_seed(random_state, endpoint, outer_fold, "inner"),
        )
        raw_splits = splitter.split(np.zeros(len(ordered)))
        splitter_name = "KFold"
    assignment_ordered = np.full(len(ordered), -1, dtype=int)
    for inner_fold, (_, validation_positions) in enumerate(raw_splits, start=1):
        assignment_ordered[np.asarray(validation_positions, dtype=int)] = inner_fold
    if np.any(assignment_ordered < 1):
        raise RuntimeError("At least one outer-development participant lacks an inner-fold assignment")
    by_subject = dict(
        zip(ordered["subject_id"].astype(str), assignment_ordered, strict=True)
    )
    assignment = (
        development["subject_id"].astype(str).map(by_subject).to_numpy(dtype=int)
    )
    return assignment, n_splits, splitter_name


def _make_estimator(endpoint: str, value: float, random_state: int):
    from sklearn.linear_model import LogisticRegression, Ridge

    if endpoint == "NHY":
        return LogisticRegression(
            C=float(value),
            class_weight="balanced",
            max_iter=5000,
            solver="lbfgs",
            random_state=int(random_state),
        )
    return Ridge(alpha=float(value))


def _candidate_grid(endpoint: str) -> tuple[float, ...]:
    return _WORKER_LOGISTIC_GRID if endpoint == "NHY" else _WORKER_RIDGE_GRID


def _fit_with_convergence(pipeline, x, y, endpoint: str) -> dict[str, Any]:
    from sklearn.base import clone
    from sklearn.exceptions import ConvergenceWarning

    attempts = [("lbfgs", 5000)]
    if endpoint == "NHY" and _WORKER_REQUIRE_CONVERGENCE:
        attempts += [("lbfgs", _WORKER_LOGISTIC_RETRY_ITER),
                     (_WORKER_LOGISTIC_RETRY_SOLVER, _WORKER_LOGISTIC_RETRY_ITER)]
    history = []
    initial = clone(pipeline)
    for solver, budget in attempts:
        # Retry every fit from the same initial state and training subset.
        pipeline.set_params(**clone(initial).get_params(deep=False))
        if endpoint == "NHY":
            pipeline.set_params(model__solver=solver, model__max_iter=budget)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            pipeline.fit(x, y)
        messages = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
        for warning in caught:
            if not issubclass(warning.category, ConvergenceWarning):
                warnings.warn(str(warning.message), warning.category)
        model = pipeline.named_steps["model"]
        raw_iter = getattr(model, "n_iter_", None)
        n_iter = int(np.max(raw_iter)) if raw_iter is not None else 0
        hit_limit = endpoint == "NHY" and n_iter >= budget
        converged = not messages and not hit_limit
        history.append({"solver": solver if endpoint == "NHY" else "ridge",
                        "max_iter": budget if endpoint == "NHY" else None,
                        "n_iter": n_iter, "warnings": messages, "converged": converged})
        if converged or not _WORKER_REQUIRE_CONVERGENCE:
            return {"attempts": history, "retries": len(history) - 1,
                    "warning_count": sum(len(a["warnings"]) for a in history),
                    "unresolved": int(not converged)}
        print(f"[nested-tuning][convergence] endpoint={endpoint} C={getattr(model, 'C', None)} "
              f"solver={solver} n_iter={n_iter} max_iter={budget}; retrying if available", flush=True)
    raise RuntimeError(f"Unconverged {endpoint} candidate after all retries: {history}")


def _assert_probe_features(features: Sequence[str], endpoint: str) -> None:
    forbidden = sorted(set(features) & FORBIDDEN_PROBE_FEATURES)
    if forbidden:
        raise RuntimeError(f"Downstream labels entered the probe feature matrix: {forbidden}")
    if endpoint in features:
        raise RuntimeError(f"Target {endpoint} entered its own probe feature matrix")


def _fit_nested_probe(
    fold_frame: pd.DataFrame,
    endpoint: str,
    image_features: Sequence[str],
    feature_setting: str,
    outer_fold: int,
    model_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    from sklearn.pipeline import Pipeline

    eligible = is_pd(fold_frame["class_label"]) & fold_frame[endpoint].notna()
    group = fold_frame.loc[eligible].copy()
    group[endpoint] = pd.to_numeric(group[endpoint], errors="coerce")
    development = group[group["split"].isin(OUTER_DEVELOPMENT_SPLITS)].copy()
    test = group[group["split"].eq("test")].copy()
    if development.empty or test.empty:
        raise RuntimeError(f"No eligible outer-development/test data for {endpoint}, fold={outer_fold}")
    overlap = set(development["subject_id"].astype(str)) & set(test["subject_id"].astype(str))
    if overlap:
        raise RuntimeError(f"Outer-test leakage detected: {sorted(overlap)[:10]}")

    image_features = [
        str(column)
        for column in image_features
        if column in group.columns and development[column].notna().any()
    ]
    if feature_setting == MRI_ONLY:
        features = list(image_features)
    elif feature_setting == MRI_COV:
        features = list(image_features) + [
            column for column in covariate_columns(group) if column not in image_features
        ]
    elif feature_setting == COV_ONLY:
        features = covariate_columns(group)
    else:
        raise ValueError(f"Unsupported feature setting {feature_setting}")
    if not features:
        raise RuntimeError(f"No features for {feature_setting}/{endpoint}/fold={outer_fold}")
    _assert_probe_features(features, endpoint)
    if development[features].isna().all(axis=1).any():
        bad = development.loc[development[features].isna().all(axis=1), "subject_id"].astype(str).tolist()
        raise RuntimeError(f"Participants have no usable probe features: {bad[:10]}")

    assignment, n_inner_folds, splitter_name = _inner_assignment(
        development,
        endpoint,
        outer_fold,
        _WORKER_INNER_FOLDS,
        _WORKER_RANDOM_STATE,
    )
    development = development.reset_index(drop=True)
    if len(assignment) != len(development):
        raise RuntimeError("Inner-fold assignment length mismatch")
    development["_inner_fold"] = assignment
    primary_metric = _WORKER_HY_METRIC if endpoint == "NHY" else "r2"
    candidate_rows: list[dict[str, Any]] = []
    best_value: float | None = None
    best_score = -np.inf

    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        from contextlib import nullcontext

        def threadpool_limits(*args, **kwargs):  # type: ignore
            return nullcontext()

    for candidate in _candidate_grid(endpoint):
        inner_pred = np.full(len(development), np.nan, dtype=float)
        n_warnings = 0
        fit_history = []
        for inner_fold in range(1, n_inner_folds + 1):
            inner_train = development[development["_inner_fold"].ne(inner_fold)].copy()
            inner_val = development[development["_inner_fold"].eq(inner_fold)].copy()
            if set(inner_train["subject_id"].astype(str)) & set(inner_val["subject_id"].astype(str)):
                raise RuntimeError("Inner train/validation subject leakage")
            preprocessor = make_preprocessor(inner_train, features)
            estimator = _make_estimator(
                endpoint,
                float(candidate),
                _stable_seed(
                    _WORKER_RANDOM_STATE,
                    endpoint,
                    outer_fold,
                    model_seed,
                    candidate,
                    inner_fold,
                ),
            )
            pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
            with threadpool_limits(limits=1):
                try:
                    fit_info = _fit_with_convergence(pipeline, inner_train[features], inner_train[endpoint], endpoint)
                except RuntimeError as exc:
                    raise RuntimeError(f"{endpoint} outer_fold={outer_fold} inner_fold={inner_fold} "
                                       f"setting={feature_setting} candidate={candidate}: {exc}") from exc
                predicted = pipeline.predict(inner_val[features])
            n_warnings += fit_info["warning_count"]
            fit_history.append({"inner_fold": inner_fold, **fit_info})
            positions = inner_val.index.to_numpy(dtype=int)
            inner_pred[positions] = pd.to_numeric(
                pd.Series(predicted), errors="coerce"
            ).to_numpy(dtype=float)
        if not np.isfinite(inner_pred).all():
            raise RuntimeError(
                f"Non-finite inner OOF predictions for {endpoint}, fold={outer_fold}, candidate={candidate}"
            )
        metrics = metric_bundle(endpoint, development[endpoint], inner_pred)
        score = float(metrics[primary_metric])
        candidate_rows.append(
            {
                "endpoint": endpoint,
                "outer_fold": int(outer_fold),
                "feature_setting": feature_setting,
                "hyperparameter_name": "C" if endpoint == "NHY" else "alpha",
                "hyperparameter_value": float(candidate),
                "selection_metric": primary_metric,
                "selection_score": score,
                "n_inner_folds": int(n_inner_folds),
                "inner_splitter": splitter_name,
                "n_inner_oof": int(len(development)),
                "convergence_warning_count": int(n_warnings),
                "unresolved_convergence_count": sum(item["unresolved"] for item in fit_history),
                "convergence_retries": sum(item["retries"] for item in fit_history),
                "fit_attempts": json.dumps(fit_history),
                **{f"inner_{name}": value for name, value in metrics.items()},
            }
        )
        if np.isfinite(score) and score > best_score + 1e-12:
            best_score = score
            best_value = float(candidate)
    if best_value is None:
        raise RuntimeError(
            f"No finite {primary_metric} candidate for {endpoint}, fold={outer_fold}, "
            f"setting={feature_setting}"
        )

    final_preprocessor = make_preprocessor(development, features)
    final_estimator = _make_estimator(
        endpoint,
        best_value,
        _stable_seed(_WORKER_RANDOM_STATE, endpoint, outer_fold, model_seed, "final"),
    )
    final_pipeline = Pipeline(
        [("preprocessor", final_preprocessor), ("model", final_estimator)]
    )
    with threadpool_limits(limits=1):
        final_fit = _fit_with_convergence(final_pipeline, development[features], development[endpoint], endpoint)
        test_pred = final_pipeline.predict(test[features])
    test_pred = pd.to_numeric(pd.Series(test_pred), errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(test_pred).all():
        raise RuntimeError("Final outer-test predictions contain non-finite values")
    if endpoint == "NHY" and not set(np.unique(test_pred)).issubset({1.0, 2.0, 3.0}):
        raise RuntimeError(f"H&Y predictions are outside original stages: {np.unique(test_pred)}")
    n_encoded = int(
        final_pipeline.named_steps["preprocessor"].transform(
            development.iloc[:1][features]
        ).shape[1]
    )
    predictions = pd.DataFrame(
        {
            "subject_id": test["subject_id"].astype(str).to_numpy(),
            "endpoint": endpoint,
            "fold": int(outer_fold),
            "feature_setting": feature_setting,
            "y_true": test[endpoint].to_numpy(dtype=float),
            "y_pred": test_pred,
            "split": "test",
        }
    )
    selected = {
        "endpoint": endpoint,
        "outer_fold": int(outer_fold),
        "feature_setting": feature_setting,
        "hyperparameter_name": "C" if endpoint == "NHY" else "alpha",
        "selected_hyperparameter": best_value,
        "selection_metric": primary_metric,
        "selection_score": best_score,
        "n_inner_folds": int(n_inner_folds),
        "inner_splitter": splitter_name,
        "inner_oof_aggregation": "pooled_participant_level",
        "n_outer_development": int(len(development)),
        "n_outer_test": int(len(test)),
        "n_features_before_encoding": int(len(features)),
        "n_image_features": int(len(image_features)),
        "n_features_after_encoding": n_encoded,
        "final_convergence_warning_count": int(final_fit["warning_count"]),
        "final_unresolved_convergence_count": final_fit["unresolved"],
        "final_fit_attempts": json.dumps(final_fit["attempts"]),
    }
    if endpoint == "NHY":
        majority = float(development[endpoint].mode().sort_values().iloc[0])
        predictions["development_majority_stage"] = majority
        selected["development_majority_stage"] = majority
    audit = {
        "endpoint": endpoint,
        "outer_fold": int(outer_fold),
        "feature_setting": feature_setting,
        "outer_development_splits": "+".join(OUTER_DEVELOPMENT_SPLITS),
        "n_manifest_train": int(group["split"].eq("train").sum()),
        "n_manifest_val": int(group["split"].eq("val").sum()),
        "n_outer_development": int(len(development)),
        "n_outer_test": int(len(test)),
        "outer_development_test_overlap_n": int(len(overlap)),
        "inner_train_validation_overlap_max": 0,
        "outer_test_seen_during_candidate_selection": False,
        "target_in_features": bool(endpoint in features),
        "forbidden_label_features": ",".join(sorted(set(features) & FORBIDDEN_PROBE_FEATURES)),
        "status": "pass",
    }
    return predictions, pd.DataFrame(candidate_rows), selected, audit


def read_embeddings(root, checkpoint_name, experiments, seeds, folds):
    configs = {}
    import yaml

    for path in (ROOT / "configs").glob("*.yml"):
        cfg = yaml.safe_load(path.read_text())
        configs[cfg["experiment_name"]] = cfg["experiment"]
    groups = {}
    for path in sorted(Path(root).glob(f"**/{checkpoint_name}/embeddings.csv")):
        seed = int(path.parents[2].name.removeprefix("seed_"))
        fold = int(path.parents[1].name.removeprefix("fold_"))
        name = path.parents[3].name
        experiment = configs.get(name)
        if experiment is None:
            raise ValueError(f"Unknown experiment directory: {path}")
        if (experiments and experiment not in experiments) or seed not in seeds or fold not in folds:
            continue
        frame = pd.read_csv(path, low_memory=False, dtype={"subject_id": str})
        for column, expected in (("seed", seed), ("fold", fold)):
            if column in frame and not pd.to_numeric(frame[column], errors="raise").eq(expected).all():
                raise ValueError(f"{column} does not agree with the embedding path: {path}")
        frame["subject_id"] = frame["subject_id"].astype("string").str.strip()
        if frame.subject_id.isna().any() or frame.duplicated("subject_id").any():
            raise ValueError(f"Missing or duplicate participant identifiers: {path}")
        if "split" not in frame:
            raise ValueError(f"Embedding split labels missing: {path}")
        features = [f"z_global_{i}" for i in range(256)]
        if not set(features).issubset(frame):
            raise ValueError(f"Expected 256 embedding columns: {path}")
        if not np.isfinite(frame[features].to_numpy(dtype=float)).all():
            raise ValueError(f"Non-finite embeddings: {path}")
        key = (experiment, seed, fold)
        if key in groups:
            raise ValueError(f"Duplicate embedding export: {key}")
        groups[key] = frame[["subject_id", "split", *features]]
    if not groups:
        raise FileNotFoundError(f"No requested embeddings under {root}")
    experiments = experiments or sorted({key[0] for key in groups})
    missing = [(e, s, f) for e in experiments for s in seeds for f in folds if (e, s, f) not in groups]
    if missing:
        raise FileNotFoundError(f"Missing experiment/seed/fold embeddings: {missing}")
    return groups, experiments


def run(args):
    global _WORKER_RIDGE_GRID, _WORKER_LOGISTIC_GRID, _WORKER_INNER_FOLDS, _WORKER_RANDOM_STATE
    if args.inner_folds < 2 or any(x <= 0 or not np.isfinite(x) for x in args.ridge_grid + args.logistic_grid):
        raise ValueError("Use at least two inner folds and finite positive regularization candidates")
    _WORKER_RIDGE_GRID = tuple(args.ridge_grid)
    _WORKER_LOGISTIC_GRID = tuple(args.logistic_grid)
    _WORKER_INNER_FOLDS = args.inner_folds
    _WORKER_RANDOM_STATE = args.random_state
    manifest = pd.read_csv(args.manifest, low_memory=False, dtype={"subject_id": str, "PATNO": str, "patno": str})
    manifest, mapping = canonicalize_manifest(manifest)
    manifest = select_3t_manifest(manifest, ["t1_crop_path", "t2_crop_path"])
    if manifest.subject_id.isna().any() or not manifest.split.isin(["train", "val", "test"]).all():
        raise ValueError("Manifest contains missing subject IDs or invalid split labels")
    missing = set(args.endpoints) - set(manifest)
    if missing:
        raise ValueError(f"Missing requested endpoints: {sorted(missing)}")
    covariate_columns(manifest)
    if "NHY" in args.endpoints:
        stages = manifest.loc[is_pd(manifest.class_label), "NHY"].dropna().unique()
        if not set(stages).issubset({1., 2., 3.}):
            raise ValueError(f"Expected original PD H&Y stages 1, 2, 3; found {stages}")
    groups, experiments = read_embeddings(args.embedding_root, args.checkpoint_name, args.experiments, args.seeds, args.folds)
    subjects = set(manifest.subject_id)
    for fold in args.folds:
        partition = manifest[manifest.fold.eq(fold)]
        if set(partition.subject_id) != subjects or set(partition.split) != {"train", "val", "test"}:
            raise ValueError(f"Inconsistent participant coverage or empty partitions in fold {fold}")
    if set(args.folds) == {1, 2, 3, 4, 5}:
        counts = manifest[manifest.split.eq("test")].groupby("subject_id").size()
        if set(counts.index) != subjects or not counts.eq(1).all():
            raise ValueError("Each participant must enter exactly one outer test fold")
    for endpoint in args.endpoints:
        values = manifest.loc[is_pd(manifest.class_label), ["subject_id", endpoint]]
        if values.groupby("subject_id")[endpoint].nunique(dropna=False).gt(1).any():
            raise ValueError(f"Endpoint {endpoint} differs across a participant's folds")

    predictions, candidates, selected, audits = [], [], [], []
    assignments = []
    for fold in args.folds:
        partition = manifest[manifest.fold.eq(fold)].copy()
        for endpoint in args.endpoints:
            dev = partition[is_pd(partition.class_label) & partition[endpoint].notna()
                            & partition.split.isin(OUTER_DEVELOPMENT_SPLITS)]
            assignment, _, _ = _inner_assignment(dev, endpoint, fold, args.inner_folds, args.random_state)
            assignments.append(pd.DataFrame({"subject_id": dev.subject_id.to_numpy(),
                                            "endpoint": endpoint, "fold": fold, "inner_fold": assignment}))

        # The covariate model is fitted once per fold, then repeated across seeds.
        jobs = [("COV_ONLY", args.random_state, partition, [], (COV_ONLY,), args.seeds)]
        for experiment in experiments:
            for seed in args.seeds:
                embedding = groups[(experiment, seed, fold)]
                if set(embedding.subject_id) != set(partition.subject_id):
                    raise ValueError(f"Embedding/manifest cohort mismatch: {experiment}/{seed}/{fold}")
                data = partition.merge(embedding, on="subject_id", how="left", validate="one_to_one",
                                       suffixes=("", "_embedding"))
                saved_split = data.split_embedding.astype("string").str.lower().replace({"valid": "val", "validation": "val"})
                if not data.split.eq(saved_split).all():
                    raise ValueError(f"Manifest and embedding partitions disagree: {experiment}/{seed}/{fold}")
                features = [f"z_global_{i}" for i in range(256)]
                jobs.append((experiment, seed, data, features, (MRI_ONLY, MRI_COV), [seed]))
        for experiment, fit_seed, data, features, settings, output_seeds in jobs:
            for endpoint in args.endpoints:
                for setting in settings:
                    pred, candidate, choice, audit = _fit_nested_probe(data, endpoint, features, setting, fold, fit_seed)
                    for seed in output_seeds:
                        extra = {"experiment": experiment, "seed": seed, "protocol": "r2",
                                 "checkpoint_name": args.checkpoint_name, "cohort": "pd_only"}
                        predictions.append(pred.assign(**extra))
                        candidates.append(candidate.assign(**extra))
                        selected.append({**choice, **extra})
                        audits.append({**audit, **extra})
            print(f"[frozen probe] {experiment} seed={fit_seed} fold={fold}", flush=True)

    predictions = pd.concat(predictions, ignore_index=True)
    keys = ["experiment", "seed", "endpoint", "feature_setting"]
    if predictions.duplicated([*keys, "subject_id"]).any():
        raise ValueError("A participant has multiple held-out predictions in one seed")
    pooled, by_fold = [], []
    for key, group in predictions.groupby(keys):
        extra = dict(zip(keys, key))
        expected = manifest[is_pd(manifest.class_label) & manifest[extra["endpoint"]].notna()
                            & manifest.split.eq("test") & manifest.fold.isin(args.folds)]
        if set(group.subject_id) != set(expected.subject_id):
            raise ValueError(f"Incomplete held-out predictions: {key}")
        pooled.append({**extra, "n_test": len(group), "n_folds": group.fold.nunique(),
                       **metric_bundle(extra["endpoint"], group.y_true, group.y_pred)})
        for fold, part in group.groupby("fold"):
            by_fold.append({**extra, "fold": fold, "n_test": len(part),
                            **metric_bundle(extra["endpoint"], part.y_true, part.y_pred)})
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output / "oof_predictions.csv", index=False)
    pd.DataFrame(pooled).to_csv(output / "pooled_oof_metrics.csv", index=False)
    pd.DataFrame(by_fold).to_csv(output / "metrics_by_fold.csv", index=False)
    pd.concat(candidates, ignore_index=True).to_csv(output / "inner_cv_candidates.csv", index=False)
    pd.DataFrame(selected).to_csv(output / "selected_hyperparameters.csv", index=False)
    pd.DataFrame(audits).to_csv(output / "split_audit.csv", index=False)
    pd.concat(assignments, ignore_index=True).to_csv(output / "inner_fold_assignments.csv", index=False)
    full = set(args.seeds) == set(range(3407, 3412)) and set(args.folds) == set(range(1, 6))
    summary = {"status": "pass" if full else "partial_seed_or_fold_run", "protocol": "r2",
               "field_strength": "3T", "cohort": "pd_only", "experiments": experiments,
               "seeds": args.seeds, "folds": args.folds, "endpoints": args.endpoints,
               "ridge_grid": args.ridge_grid, "logistic_grid": args.logistic_grid,
               "inner_folds_requested": args.inner_folds, "random_state": args.random_state,
               "selection_metrics": {e: "quadratic_weighted_kappa" if e == "NHY" else "r2" for e in args.endpoints},
               "outer_development_splits": list(OUTER_DEVELOPMENT_SPLITS),
               "manifest_column_mapping": mapping, "n_prediction_rows": len(predictions),
               "test_predictions_only": True, "formal_inference_included": False}
    (output / "frozen_probe_summary.json").write_text(json.dumps(summary, indent=2))
    print(summary)


def get_parser():
    parser = argparse.ArgumentParser(description="Nested frozen probing on the common 3T cohort")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--embedding-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint-name", choices=["checkpoint_best"], default="checkpoint_best")
    parser.add_argument("--experiments", nargs="+", choices=[f"E{i}" for i in range(1, 10)])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(3407, 3412)))
    parser.add_argument("--folds", nargs="+", type=int, choices=range(1, 6), default=list(range(1, 6)))
    parser.add_argument("--endpoints", nargs="+", choices=["putamen_sbr_min", "NHY"], default=["putamen_sbr_min"])
    parser.add_argument("--cohort", choices=["pd_only"], default="pd_only")
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--ridge-grid", nargs="+", type=float, default=list(DEFAULT_GRID))
    parser.add_argument("--logistic-grid", nargs="+", type=float, default=list(DEFAULT_GRID))
    parser.add_argument("--random-state", type=int, default=3407)
    return parser


if __name__ == "__main__":
    run(get_parser().parse_args())
