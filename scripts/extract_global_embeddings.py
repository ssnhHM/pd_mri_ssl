#!/usr/bin/env python3
import argparse
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.ppmi_ssl_dataset import PPMISSLDataset, select_3t_manifest
from models.ssl_global import GlobalSSL


def read_config(path):
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def read_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def parse_shape(value):
    if value is None or str(value).lower() in {"none", "null", "native", "noresize"}:
        return None
    return tuple(int(v) for v in value)


def subject_column(df):
    column = next(
        (name for name in ("subject_id", "PATNO", "patno") if name in df),
        None,
    )
    if column is None:
        raise ValueError("Manifest must contain subject_id or PATNO")
    return column


def checkpoint_path(root, experiment, seed, fold, name):
    filename = name if name.endswith(".pt") else f"{name}.pt"
    return root / experiment / f"seed_{seed}" / f"fold_{fold}" / filename


@torch.no_grad()
def extract_fold(args, base_config, fold, checkpoint_name, device):
    experiment = base_config["experiment_name"]
    ckpt_path = checkpoint_path(
        Path(args.checkpoint_root),
        experiment,
        args.seed,
        fold,
        checkpoint_name,
    )
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)

    checkpoint = read_checkpoint(ckpt_path, device)
    cfg = {**base_config, **checkpoint.get("config", {})}
    if cfg.get("field_strength_mode") != "3T":
        raise ValueError("Checkpoint configuration is not the requested 3T analysis")
    if cfg.get("global_inputs") != base_config["global_inputs"]:
        raise ValueError("Checkpoint and requested configuration disagree on input columns/order")
    modalities = cfg["global_inputs"]
    shape = parse_shape(cfg.get("global_shape"))

    manifest_path = Path(args.manifest)
    manifest = pd.read_csv(
        manifest_path,
        low_memory=False,
        dtype={"subject_id": str, "PATNO": str, "patno": str},
    )
    id_column = subject_column(manifest)
    manifest = select_3t_manifest(manifest, modalities)
    fold_df = (
        manifest[manifest["fold"].astype(str) == str(fold)]
        .drop_duplicates(id_column)
        .reset_index(drop=True)
    )
    if fold_df.empty:
        raise RuntimeError(f"No rows found for fold {fold}")

    dataset = PPMISSLDataset(
        fold_df,
        modalities,
        base_dir=manifest_path.parent,
        target_shape=shape,
        augment=False,
    )
    loader_args = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers:
        loader_args["prefetch_factor"] = args.prefetch_factor
    loader = DataLoader(dataset, **loader_args)

    model = GlobalSSL(
        in_channels=len(modalities),
        embedding_dim=cfg.get("embedding_dim", 256),
        projection_dim=cfg.get("projection_dim", 128),
        base_channels=cfg.get("base_channels", 24),
        dropout=cfg.get("dropout", 0.1),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    rows = []
    offset = 0
    start = time.time()
    for batch in loader:
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        images = batch["view1"].to(device=device, dtype=dtype, non_blocking=True)
        amp = torch.amp.autocast("cuda") if device.type == "cuda" else nullcontext()
        with amp:
            embeddings = model.encode(images)
        embeddings = embeddings.float().cpu().numpy()

        for i, embedding in enumerate(embeddings):
            source = fold_df.iloc[offset + i]
            row = {
                "subject_id": str(source[id_column]),
                "fold": fold,
                "split": str(source["split"]),
                "seed": args.seed,
                "experiment": base_config["experiment"],
                "experiment_name": experiment,
                "checkpoint_name": checkpoint_name.removesuffix(".pt"),
            }
            row.update(
                {f"z_global_{j}": float(value) for j, value in enumerate(embedding)}
            )
            rows.append(row)

        offset += len(embeddings)
        if offset == len(dataset) or offset % args.progress_every < len(embeddings):
            print(
                f"{experiment} seed={args.seed} fold={fold}: "
                f"{offset}/{len(dataset)} ({(time.time() - start) / 60:.1f}m)"
            )

    output = pd.DataFrame(rows)
    keys = ["subject_id", "fold", "split"]
    if output.duplicated(keys).any() or len(output) != len(dataset):
        raise RuntimeError("Embedding export failed its row-level uniqueness check")

    checkpoint_name = checkpoint_name.removesuffix(".pt")
    output_dir = (
        Path(args.output_root)
        / experiment
        / f"seed_{args.seed}"
        / f"fold_{fold}"
        / checkpoint_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "embeddings.csv"
    output.to_csv(output_path, index=False)

    audit = {
        "status": "pass",
        "experiment_name": experiment,
        "seed": args.seed,
        "fold": fold,
        "checkpoint_name": checkpoint_name,
        "rows": len(output),
        "unique_subjects": output["subject_id"].nunique(),
        "augmentation_enabled": False,
        "global_inputs": modalities,
    }
    (output_dir / "embedding_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    return output_path


def main(args):
    cfg = read_config(args.config)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    device = torch.device(args.device)

    outputs = []
    for checkpoint_name in args.checkpoint_names:
        for fold in args.folds:
            outputs.append(
                extract_fold(args, cfg, fold, checkpoint_name, device)
            )
    print(f"Wrote {len(outputs)} embedding files to {args.output_root}")


def get_parser():
    parser = argparse.ArgumentParser(description="Extract frozen MRI embeddings")
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument(
        "--checkpoint-names", nargs="+", default=["checkpoint_best"]
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    return parser


if __name__ == "__main__":
    main(get_parser().parse_args())
