from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from datasets.field_strength import filter_3t


MRI_COLUMNS = {
    "t1_crop_path": ("t1_crop_path", "t1_path", "T1_path", "t1w_path", "T1"),
    "t1j_crop_path": (
        "t1j_crop_path",
        "t1jac_crop_path",
        "T1JAC_path",
        "T1J",
        "t1_jacobian_path",
    ),
    "ratio_crop_path": (
        "ratio_crop_path",
        "ratio_path",
        "T1T2Ratio_path",
        "T1_T2_ratio_path",
        "Ratio",
    ),
    "t2_crop_path": ("t2_crop_path", "t2_path", "T2_path", "t2w_path", "T2"),
}


def select_3t_manifest(df, inputs):
    filtered = filter_3t(df, inputs)
    common = filter_3t(df, ["t1_crop_path", "t2_crop_path"])
    subject = next((c for c in ("subject_id", "PATNO", "patno") if c in df), None)
    if subject is None:
        raise ValueError("Manifest must contain subject_id or PATNO")
    keys = [subject, "fold", "split"]
    if set(map(tuple, filtered[keys].to_numpy())) != set(map(tuple, common[keys].to_numpy())):
        raise ValueError("Input-specific 3T cohort differs from the common T1/T2 3T cohort; no intersection was taken")
    if filtered.duplicated([subject, "fold"]).any():
        raise ValueError("Duplicate subject-fold records or overlapping partitions")
    if filtered.empty:
        raise ValueError("No participants remain after 3T filtering")
    splits = filtered["split"].astype("string").str.lower().replace({"valid": "val", "validation": "val"})
    if not splits.isin(["train", "val", "test"]).all():
        raise ValueError("Unknown train/val/test partition")
    folds = pd.to_numeric(filtered["fold"], errors="raise")
    if set(folds) == {1, 2, 3, 4, 5}:
        subjects = set(filtered[subject])
        for fold in range(1, 6):
            if set(filtered.loc[folds.eq(fold), subject]) != subjects:
                raise ValueError(f"Participant cohort differs in fold {fold}")
        counts = filtered.loc[splits.eq("test")].groupby(subject).size()
        if set(counts.index) != subjects or not counts.eq(1).all():
            raise ValueError("Each participant must enter exactly one outer test fold")
    return filtered


def resolve_modality_columns(df, requested):
    columns = {column.lower(): column for column in df.columns}
    resolved = []
    mapping = {}

    for name in requested:
        aliases = MRI_COLUMNS.get(name, (name,))
        column = next((columns[a.lower()] for a in aliases if a.lower() in columns), None)
        mapping[name] = column
        if column is None:
            raise ValueError(f"Missing MRI path column: {name}")
        resolved.append(column)

    return resolved, mapping


def resolve_image_path(value, base_dir=None):
    if pd.isna(value) or not str(value).strip():
        raise FileNotFoundError("Empty MRI path")

    path = Path(str(value).strip()).expanduser()
    if path.is_absolute():
        candidates = [path]
    else:
        candidates = []
        if base_dir is not None:
            candidates.append(Path(base_dir).expanduser() / path)
        candidates.append(path)

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(path)


def robust_normalize(x):
    x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
    foreground = x[x.abs() > 1e-6]
    values = foreground if foreground.numel() >= 16 else x.flatten()
    median = values.median()
    iqr = torch.quantile(values, 0.75) - torch.quantile(values, 0.25)
    scale = (iqr / 1.349).clamp_min(1e-4)
    return ((x - median) / scale).clamp(-8.0, 8.0)


def load_volume(path, target_shape=None):
    path = Path(path)
    suffix = "".join(path.suffixes).lower()

    if suffix.endswith(".npy"):
        data = np.load(path, allow_pickle=False)
    elif suffix.endswith(".npz"):
        with np.load(path, allow_pickle=False) as archive:
            data = archive[archive.files[0]]
    else:
        data = np.asanyarray(nib.load(str(path)).dataobj)

    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 4:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got {data.shape}: {path}")

    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    x = torch.from_numpy(data).unsqueeze(0)
    if target_shape and tuple(x.shape[-3:]) != tuple(target_shape):
        x = F.interpolate(
            x.unsqueeze(0),
            size=target_shape,
            mode="trilinear",
            align_corners=False,
        ).squeeze(0)
    return robust_normalize(x)


def augment_mri(x, noise_std, intensity_scale, gamma_range, mask_prob, modality_dropout):
    x = x.clone()
    channels = x.shape[0]

    if intensity_scale:
        scale = 1 + (torch.rand(channels, 1, 1, 1) * 2 - 1) * intensity_scale
        x *= scale
    if gamma_range:
        gamma = 1 + float((torch.rand(()) * 2 - 1) * gamma_range)
        x = torch.sign(x) * x.abs().clamp_min(1e-4).pow(gamma)
    if noise_std:
        x += torch.randn_like(x) * noise_std

    if mask_prob and torch.rand(()) < mask_prob:
        shape = x.shape[-3:]
        block = [max(2, size // 8) for size in shape]
        start = [
            int(torch.randint(0, max(1, size - width + 1), (1,)))
            for size, width in zip(shape, block)
        ]
        d, h, w = start
        bd, bh, bw = block
        x[:, d : d + bd, h : h + bh, w : w + bw] = 0

    if modality_dropout and channels > 1:
        keep = torch.rand(channels) > modality_dropout
        if not keep.any():
            keep[int(torch.randint(channels, (1,)))] = True
        x *= keep[:, None, None, None].to(x.dtype)

    return x


class PPMISSLDataset(Dataset):
    def __init__(
        self,
        manifest,
        global_inputs,
        base_dir=None,
        target_shape=None,
        augment=True,
        noise_std=0.02,
        intensity_scale=0.08,
        gamma_range=0.05,
        mask_probability=0.05,
        modality_dropout=0.05,
    ):
        self.df = manifest.reset_index(drop=True).copy()
        self.global_inputs, self.column_mapping = resolve_modality_columns(
            self.df, global_inputs
        )
        self.base_dir = Path(base_dir) if base_dir else None
        self.target_shape = target_shape
        self.augment = augment
        self.augmentation = {
            "noise_std": noise_std,
            "intensity_scale": intensity_scale,
            "gamma_range": gamma_range,
            "mask_prob": mask_probability,
            "modality_dropout": modality_dropout if len(self.global_inputs) > 1 else 0.0,
        }

    def __len__(self):
        return len(self.df)

    def _load(self, row):
        images = [
            load_volume(
                resolve_image_path(row[column], self.base_dir),
                self.target_shape,
            )
            for column in self.global_inputs
        ]
        shapes = {tuple(image.shape[-3:]) for image in images}
        if len(shapes) != 1:
            raise ValueError(f"Input modalities have different shapes: {sorted(shapes)}")
        return torch.cat(images)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image = self._load(row)
        if self.augment:
            view1 = augment_mri(image, **self.augmentation)
            view2 = augment_mri(image, **self.augmentation)
        else:
            view1 = view2 = image

        return {
            "view1": view1,
            "view2": view2,
            "subject_id": str(row.get("subject_id", row.get("PATNO", ""))),
            "fold": row.get("fold", np.nan),
            "split": str(row.get("split", "")),
        }
