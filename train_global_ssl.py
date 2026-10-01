#!/usr/bin/env python3
import argparse
import json
import random
import time
from collections import deque
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from datasets.ppmi_ssl_dataset import MRI_COLUMNS, PPMISSLDataset, select_3t_manifest
from models.ssl_global import GlobalSSL
from models.ssl_losses import vicreg_loss


def read_config(path):
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_shape(value):
    if value is None or str(value).lower() in {"none", "null", "native", "noresize"}:
        return None
    shape = tuple(int(v) for v in value)
    if len(shape) != 3:
        raise ValueError("global_shape must contain three integers")
    return shape


def split_manifest(df, fold):
    subject_col = next(
        (column for column in ("subject_id", "PATNO", "patno") if column in df),
        None,
    )
    if subject_col is None:
        raise ValueError("Manifest must contain subject_id or PATNO")
    if not {"fold", "split"}.issubset(df.columns):
        raise ValueError("Manifest must contain fold and split")

    fold_df = df[df["fold"].astype(str) == str(fold)].copy()
    fold_df["split"] = (
        fold_df["split"]
        .astype("string")
        .str.lower()
        .replace({"valid": "val", "validation": "val"})
    )
    memberships = fold_df.groupby(subject_col)["split"].nunique()
    if memberships.gt(1).any():
        raise RuntimeError(f"Subject leakage detected in fold {fold}")

    parts = {
        split: fold_df[fold_df["split"] == split]
        .drop_duplicates(subject_col)
        .copy()
        for split in ("train", "val", "test")
    }
    if parts["train"].empty or parts["val"].empty:
        raise RuntimeError(f"Fold {fold} has no training or validation subjects")
    return parts, subject_col


def show_progress(label, done, total, start):
    width = 24
    fraction = done / max(total, 1)
    filled = round(width * fraction)
    elapsed = time.time() - start
    rate = done / max(elapsed, 1e-6)
    eta = (total - done) / max(rate, 1e-6)
    bar = "#" * filled + "." * (width - filled)
    end = "\n" if done == total else ""
    print(
        f"\r{label} [{bar}] {done}/{total} "
        f"elapsed={elapsed / 60:.1f}m eta={eta / 60:.1f}m",
        end=end,
        flush=True,
    )


def preload(df, split, cfg, manifest_dir, shape, device):
    dataset = PPMISSLDataset(
        df,
        cfg["global_inputs"],
        base_dir=manifest_dir,
        target_shape=shape,
        augment=False,
    )
    workers = cfg.get("preload_num_workers", 8)
    batch_size = cfg.get("preload_batch_size", 4)
    loader_args = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
    }
    if workers:
        loader_args["prefetch_factor"] = cfg.get("preload_prefetch_factor", 2)

    loader = DataLoader(dataset, **loader_args)
    chunks = []
    done = 0
    start = time.time()
    for batch in loader:
        chunks.append(batch["view1"].float())
        done += len(batch["view1"])
        if done == len(dataset) or done % cfg.get("preload_progress_every", 20) < batch_size:
            show_progress(f"preload {split}", done, len(dataset), start)

    x = torch.cat(chunks)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    size_gb = x.numel() * torch.empty((), dtype=dtype).element_size() / 1024**3
    print(f"{split}: {tuple(x.shape)}, {size_gb:.2f} GiB -> {device}")
    return x.to(device=device, dtype=dtype)


def augment_batch(x, cfg):
    x = x.clone()
    batch_size, channels = x.shape[:2]

    scale = cfg.get("intensity_scale", 0.08)
    if scale:
        factor = 1 + (
            torch.rand(
                batch_size, channels, 1, 1, 1, device=x.device, dtype=x.dtype
            )
            * 2
            - 1
        ) * scale
        x *= factor

    gamma_range = cfg.get("gamma_range", 0.05)
    if gamma_range:
        gamma = 1 + (
            torch.rand(batch_size, 1, 1, 1, 1, device=x.device, dtype=x.dtype) * 2
            - 1
        ) * gamma_range
        x = torch.sign(x) * x.abs().clamp_min(1e-4).pow(gamma)

    noise = cfg.get("noise_std", 0.02)
    if noise:
        x += torch.randn_like(x) * noise

    mask_prob = cfg.get("mask_prob", 0.05)
    if mask_prob:
        shape = x.shape[-3:]
        block = [max(2, size // 8) for size in shape]
        selected = torch.where(torch.rand(batch_size, device=x.device) < mask_prob)[0]
        for i in selected.tolist():
            start = [
                int(
                    torch.randint(
                        0, max(1, size - width + 1), (1,), device=x.device
                    )
                )
                for size, width in zip(shape, block)
            ]
            d, h, w = start
            bd, bh, bw = block
            x[i, :, d : d + bd, h : h + bh, w : w + bw] = 0

    drop_prob = cfg.get("modality_dropout", 0.05) if channels > 1 else 0
    if drop_prob:
        keep = torch.rand(batch_size, channels, device=x.device) > drop_prob
        empty = keep.sum(1) == 0
        if empty.any():
            replacement = torch.randint(
                channels, (int(empty.sum()),), device=x.device
            )
            keep[empty] = False
            keep[empty, replacement] = True
        x *= keep[:, :, None, None, None].to(x.dtype)

    return x


def autocast(device, enabled):
    if enabled and device.type == "cuda":
        return torch.amp.autocast("cuda")
    return nullcontext()


def make_scaler(device, enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled and device.type == "cuda")
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=enabled and device.type == "cuda")


def ssl_loss(model, x, cfg, device, amp, training):
    view1, view2 = (
        (augment_batch(x, cfg), augment_batch(x, cfg)) if training else (x, x)
    )
    with autocast(device, amp):
        p1 = model(view1)["p_global"]
        p2 = model(view2)["p_global"]
        loss, parts = vicreg_loss(
            p1,
            p2,
            sim_coeff=cfg.get("vicreg_sim_coeff", 25.0),
            std_coeff=cfg.get("vicreg_std_coeff", 25.0),
            cov_coeff=cfg.get("vicreg_cov_coeff", 1.0),
        )
    return loss, {name: value.item() for name, value in parts.items()}


def run_epoch(model, images, cfg, device, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    batch_size = cfg["batch_size"]
    accumulation = cfg.get("gradient_accumulation_steps", 4)
    amp = cfg.get("amp", True)
    order = (
        torch.randperm(len(images), device=device)
        if training
        else torch.arange(len(images), device=device)
    )
    totals = dict.fromkeys(("loss", "invariance", "variance", "covariance"), 0.0)
    seen = 0

    if training:
        optimizer.zero_grad(set_to_none=True)

    for step, start in enumerate(range(0, len(order), batch_size), 1):
        index = order[start : start + batch_size]
        batch = images.index_select(0, index)
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            loss, parts = ssl_loss(model, batch, cfg, device, amp, training)

        if training:
            scaler.scale(loss / accumulation).backward()
            if step % accumulation == 0 or start + batch_size >= len(order):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        n = len(index)
        totals["loss"] += loss.item() * n
        for name, value in parts.items():
            totals[name] += value * n
        seen += n

    return {name: value / seen for name, value in totals.items()}


def checkpoint(model, cfg, epoch, val_loss):
    return {
        "model_state_dict": model.state_dict(),
        "config": cfg,
        "epoch": epoch,
        "validation_loss": val_loss,
    }


def average_state_dicts(states):
    averaged = {}
    for key in states[0]:
        values = [state[key] for state in states]
        if torch.is_floating_point(values[0]):
            averaged[key] = torch.stack([v.float() for v in values]).mean(0).to(
                values[0].dtype
            )
        else:
            averaged[key] = values[-1]
    return averaged


def train(args):
    cfg = read_config(args.config)
    cfg["seed"] = args.seed if args.seed is not None else cfg.get("seed", 3407)
    cfg["fold"] = args.fold if args.fold is not None else cfg.get("fold", 1)
    cfg["epochs"] = args.epochs or cfg.get("epochs", 200)
    cfg["batch_size"] = args.batch_size or cfg.get("batch_size", 4)
    if cfg.get("field_strength_mode") != "3T":
        raise ValueError("This release uses the current-paper 3T configuration")

    if cfg.get("ssl_loss", "vicreg").lower() != "vicreg":
        raise ValueError("This release supports VICReg only")
    unknown_inputs = set(cfg["global_inputs"]) - set(MRI_COLUMNS)
    if unknown_inputs:
        raise ValueError(f"Unsupported SSL input columns: {sorted(unknown_inputs)}")

    seed, fold = cfg["seed"], cfg["fold"]
    set_seed(seed)

    manifest_path = Path(args.manifest or cfg["manifest_path"])
    manifest = pd.read_csv(
        manifest_path,
        low_memory=False,
        dtype={"subject_id": str, "PATNO": str, "patno": str},
    )
    manifest = select_3t_manifest(manifest, cfg["global_inputs"])
    parts, subject_col = split_manifest(manifest, fold)

    experiment = cfg["experiment_name"]
    output_root = Path(args.output_root or cfg["output_root"])
    output_dir = output_root / experiment / f"seed_{seed}" / f"fold_{fold}"
    summary_path = output_dir / "ssl_training_summary.json"
    ready = all((output_dir / name).is_file() for name in (
        "ssl_training_summary.json", "checkpoint_best.pt", "checkpoint_last.pt", "checkpoint_avg_last10.pt"
    ))
    if ready:
        try:
            ready = json.loads(summary_path.read_text()).get("status", "").lower() in {"", "ok", "completed"}
        except (OSError, ValueError):
            ready = False
    if ready and not args.force:
        print(f"Reusing completed run: {output_dir}")
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    device_name = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    device = torch.device(device_name)
    shape = parse_shape(cfg.get("global_shape"))

    cfg.update(
        {
            "manifest_path": str(manifest_path),
            "global_shape": list(shape) if shape else None,
            "clinical_labels_used_for_ssl": False,
            "dat_spect_used_for_ssl": False,
            "validation_augmentation": False,
        }
    )

    counts = {
        split: int(frame[subject_col].nunique()) for split, frame in parts.items()
    }
    if args.dry_run:
        print(
            {
                "experiment": experiment,
                "seed": seed,
                "fold": fold,
                "global_inputs": cfg["global_inputs"],
                "subjects": counts,
                "output_dir": str(output_dir),
            }
        )
        return

    if cfg["batch_size"] < 4:
        print("Warning: VICReg is unstable with a physical batch size below 4")

    train_images = preload(
        parts["train"], "train", cfg, manifest_path.parent, shape, device
    )
    val_images = preload(parts["val"], "val", cfg, manifest_path.parent, shape, device)

    model = GlobalSSL(
        in_channels=len(cfg["global_inputs"]),
        embedding_dim=cfg.get("embedding_dim", 256),
        projection_dim=cfg.get("projection_dim", 128),
        base_channels=cfg.get("base_channels", 24),
        dropout=cfg.get("dropout", 0.1),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.get("lr", 1e-4),
        weight_decay=cfg.get("weight_decay", 1e-4),
    )
    scaler = make_scaler(device, cfg.get("amp", True))

    with open(output_dir / "config_used.yml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)

    history = []
    last_states = deque(maxlen=cfg.get("average_last_n", 10))
    best_epoch = 0
    best_loss = float("inf")
    raw_best_epoch = 0
    raw_best_loss = float("inf")
    min_epoch = cfg.get("min_checkpoint_epoch", 50)

    for epoch in range(1, cfg["epochs"] + 1):
        train_metrics = run_epoch(
            model, train_images, cfg, device, optimizer=optimizer, scaler=scaler
        )
        val_metrics = run_epoch(model, val_images, cfg, device)
        val_loss = val_metrics["loss"]

        if val_loss < raw_best_loss:
            raw_best_epoch, raw_best_loss = epoch, val_loss

        eligible = epoch >= min_epoch or cfg["epochs"] < min_epoch
        if eligible and val_loss < best_loss:
            best_epoch, best_loss = epoch, val_loss
            torch.save(
                checkpoint(model, cfg, epoch, val_loss),
                output_dir / "checkpoint_best.pt",
            )

        torch.save(
            checkpoint(model, cfg, epoch, val_loss),
            output_dir / "checkpoint_last.pt",
        )
        last_states.append(
            {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        )

        row = {
            "epoch": epoch,
            "checkpoint_eligible": eligible,
            **{f"train_{name}": value for name, value in train_metrics.items()},
            **{f"val_{name}": value for name, value in val_metrics.items()},
        }
        history.append(row)
        pd.DataFrame(history).to_csv(output_dir / "train_log.csv", index=False)
        print(
            f"{experiment} seed={seed} fold={fold} epoch={epoch:03d} "
            f"train={train_metrics['loss']:.5f} val={val_loss:.5f}"
        )

    if best_epoch == 0:
        best_epoch, best_loss = cfg["epochs"], history[-1]["val_loss"]
        torch.save(
            checkpoint(model, cfg, best_epoch, best_loss),
            output_dir / "checkpoint_best.pt",
        )

    torch.save(
        {
            "model_state_dict": average_state_dicts(list(last_states)),
            "config": cfg,
            "epoch": cfg["epochs"],
            "averaged_last_n": len(last_states),
        },
        output_dir / "checkpoint_avg_last10.pt",
    )

    summary = {
        "status": "completed",
        "experiment_name": experiment,
        "seed": seed,
        "fold": fold,
        "global_inputs": cfg["global_inputs"],
        "global_input_shape": list(train_images.shape[-3:]),
        "resize_mode": "resize" if shape else "noresize",
        "ssl_loss": "vicreg",
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "unconstrained_best_epoch": raw_best_epoch,
        "unconstrained_best_validation_loss": raw_best_loss,
        "best_epoch_constrained_by_min_checkpoint_epoch": bool(
            cfg["epochs"] >= min_epoch and raw_best_epoch < min_epoch
        ),
        "min_checkpoint_epoch": min_epoch,
        "final_train_loss": history[-1]["train_loss"],
        "final_validation_loss": history[-1]["val_loss"],
        "physical_batch_size": cfg["batch_size"],
        "gradient_accumulation_steps": cfg.get("gradient_accumulation_steps", 4),
        "effective_batch_size": cfg["batch_size"]
        * cfg.get("gradient_accumulation_steps", 4),
        "train_subjects": counts["train"],
        "val_subjects": counts["val"],
        "test_subjects": counts["test"],
        "clinical_labels_used_for_ssl": False,
        "dat_spect_used_for_ssl": False,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary)


def get_parser():
    parser = argparse.ArgumentParser(
        description="Train one fold of the global 3D CNN with VICReg"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--fold", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device")
    parser.add_argument("--output-root")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


if __name__ == "__main__":
    train(get_parser().parse_args())
