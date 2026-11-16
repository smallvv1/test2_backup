#!/usr/local/bin/python3
"""Train-only audit of the current GPC selector versus Batch-Hard Triplet.

This diagnostic imports the current repository wrapper and loss implementation,
but never calls an optimizer, scheduler, evaluator, or checkpoint writer.
It builds train metadata directly from the audited dataset parsers because the
summary-caption JSON is not present on this server.  Caption features are not
needed for the relation/gradient experiments.
"""

from __future__ import annotations

import csv
import glob
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path("/home/DIFFER/github_latest_test2_lXaBlM")
ENGINE = Path("/data/DIFFER")
OUT = ROOT / "diagnostics/gpc_triplet_complementarity_20261116"
SEED = 1234
EPOCH = 60
DEVICE = torch.device("cuda:0")
MAX_BATCHES = 8

sys.path.insert(0, str(ENGINE))
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("current_train_final_gpc", ROOT / "scripts/train_final_gpc.py")
tfg = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(tfg)
from data.dataset_loader import ImageDataset  # noqa: E402
from loss.triplet_loss import euclidean_dist, hard_example_mining  # noqa: E402
from model import build_model  # noqa: E402


CHECKPOINTS = {
    "prcc_gpc": Path("/home/DIFFER/log_NDP/ndp_nteach_current_b0_pair_20261031_125146/PRCC_B0_full/PRCC/model_final.pth"),
    "prcc_bio": Path("/data/DIFFER/outputs/new_method_track/phase7a_prcc_relation_validation/P0_prcc_bio/model_final.pth"),
    "ltcc_gpc": Path("/home/DIFFER/log_NDP/ndp_nteach_b0_recovery_20261030_210612/LTCC_B0_full/LTCC/model_final.pth"),
}


def safe(v: Any) -> Any:
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, torch.Tensor):
        return v.detach().float().cpu().item() if v.ndim == 0 else v.detach().float().cpu().tolist()
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, dict):
        return {str(k): safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [safe(x) for x in v]
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(safe(payload), indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: safe(row.get(key, "")) for key in fields})


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_prcc_items() -> list[dict[str, Any]]:
    root = Path("/home/data/prcc/rgb/train")
    pdirs = sorted(glob.glob(str(root / "*")))
    raw_pids = sorted(int(Path(p).name) for p in pdirs)
    pid2label = {pid: idx for idx, pid in enumerate(raw_pids)}
    clothes_names = sorted(
        {
            Path(p).name if Path(f).name[0] in "AB" else Path(p).name + Path(f).name[0]
            for p in pdirs
            for f in glob.glob(str(Path(p) / "*.jpg"))
        }
    )
    clothes2label = {name: idx for idx, name in enumerate(clothes_names)}
    result = []
    for pdir in pdirs:
        raw_pid = int(Path(pdir).name)
        for image_path in glob.glob(str(Path(pdir) / "*.jpg")):
            cam = Path(image_path).name[0]
            clothes_name = Path(pdir).name if cam in "AB" else Path(pdir).name + cam
            result.append(
                {
                    "image_path": image_path,
                    "pid": pid2label[raw_pid],
                    "camid": {"A": 0, "B": 1, "C": 2}[cam],
                    "clothes_id": clothes2label[clothes_name],
                }
            )
    return result


def make_ltcc_items() -> list[dict[str, Any]]:
    root = Path("/home/data/LTCC_ReID/train")
    paths = sorted(glob.glob(str(root / "*.png")))
    pattern_pid_cam = re.compile(r"(\d+)_(\d+)_c(\d+)")
    pattern_clothes = re.compile(r"(\w+)_c")
    raw = []
    for image_path in paths:
        pid, _, cam = map(int, pattern_pid_cam.search(image_path).groups())
        clothes = pattern_clothes.search(Path(image_path).name).group(1)
        raw.append((image_path, pid, cam - 1, clothes))
    pid2label = {pid: idx for idx, pid in enumerate(sorted({x[1] for x in raw}))}
    clothes2label = {name: idx for idx, name in enumerate(sorted({x[3] for x in raw}))}
    return [
        {
            "image_path": image_path,
            "pid": pid2label[pid],
            "camid": cam,
            "clothes_id": clothes2label[clothes],
        }
        for image_path, pid, cam, clothes in raw
    ]


def make_bank(items: list[dict[str, Any]], dataset_name: str) -> dict[str, Any]:
    """Reproduce the current train_final_gpc bank, with deployed LTCC source."""
    if dataset_name == "prcc":
        holder = SimpleNamespace(train=items)
        return tfg.build_train_only_pair_bank(holder, dataset_name)
    exact: set[tuple[int, int]] = set()
    source = ENGINE / "outputs/new_method_track/phase6b_soft_relation_screen/artifacts/relation_source_rows.csv"
    with source.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            left, right = sorted((int(row["index_a"]), int(row["index_b"])))
            if left == right or right >= len(items):
                raise RuntimeError(f"bad LTCC relation row {row}")
            if items[left]["pid"] != items[right]["pid"] or items[left]["clothes_id"] == items[right]["clothes_id"]:
                raise RuntimeError(f"LTCC relation row is not cross-clothes same-PID: {row}")
            exact.add((left, right))
    return {
        "exact_lookup": exact,
        "path_to_index": {str(item["image_path"]): idx for idx, item in enumerate(items)},
        "dataset": dataset_name,
        "train_image_count": len(items),
        "cross_key_count": len(exact),
        "source": "deployed Phase6B exact binary candidate keys; relation scores ignored",
        "source_path": str(source),
        "source_row_count": len(exact),
        "all_keys_train_only": True,
    }


def sampler_batches(items: list[dict[str, Any]], dataset_name: str, rank: int) -> tuple[Any, list[list[int]]]:
    cfg = tfg.cfg_for(dataset_name)
    spec = tfg.SPEC[dataset_name]
    sampler = tfg.FinalBatchSampler(
        items,
        int(spec["batch_size"]),
        int(spec["num_instances"]),
        int(spec["steps_per_epoch"]),
        SEED,
        rank,
    )
    sampler.prepare(EPOCH)
    return cfg, list(iter(sampler))


def selector_scan(items: list[dict[str, Any]], bank: dict[str, Any], dataset_name: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cfg = tfg.cfg_for(dataset_name)
    spec = tfg.SPEC[dataset_name]
    all_rows: list[dict[str, Any]] = []
    per_pid = Counter()
    per_state = Counter()
    total_batches = 0
    zero_batches = 0
    candidate_cross: list[int] = []
    candidate_same: list[int] = []
    candidate_id: list[int] = []
    selected_counts: list[int] = []
    selected_cross = 0
    selected_same = 0
    for rank in (0, 1):
        _, batches = sampler_batches(items, dataset_name, rank)
        for iteration, batch in enumerate(batches, start=1):
            entries = [items[index] for index in batch]
            selected, info = tfg.select_gpc_pairs(entries, bank, EPOCH, rank, iteration, SEED)
            total_batches += 1
            zero_batches += int(len(selected) == 0)
            candidate_cross.append(int(info["candidate_counts"]["cross"]))
            candidate_same.append(int(info["candidate_counts"]["same"]))
            candidate_id.append(int(info["candidate_counts"]["id"]))
            selected_counts.append(len(selected))
            selected_cross += int(info["cross_clothes_selected"])
            selected_same += int(info["same_clothes_selected"])
            for i, j, image_key, state_key in selected:
                pid = int(entries[i]["pid"])
                per_pid[pid] += 1
                per_state[(pid, int(entries[i]["clothes_id"]))] += 1
                per_state[(pid, int(entries[j]["clothes_id"]))] += 1
                all_rows.append(
                    {
                        "dataset": dataset_name,
                        "rank": rank,
                        "epoch": EPOCH,
                        "iteration": iteration,
                        "image_i": str(entries[i]["image_path"]),
                        "image_j": str(entries[j]["image_path"]),
                        "local_i": i,
                        "local_j": j,
                        "pid": pid,
                        "clothes_i": int(entries[i]["clothes_id"]),
                        "clothes_j": int(entries[j]["clothes_id"]),
                        "relation": "cross_clothes" if entries[i]["clothes_id"] != entries[j]["clothes_id"] else "same_clothes",
                        "image_key": str(image_key),
                        "state_key": str(state_key),
                        "candidate_cross": info["candidate_counts"]["cross"],
                        "candidate_same": info["candidate_counts"]["same"],
                        "candidate_id": info["candidate_counts"]["id"],
                        "budget": info["matched_count"],
                        "selected_count": info["selected_count"],
                    }
                )
    summary = {
        "dataset": dataset_name,
        "train_images": len(items),
        "train_pids": len({int(x["pid"]) for x in items}),
        "train_clothing_states": len({(int(x["pid"]), int(x["clothes_id"])) for x in items}),
        "world_size": 2,
        "seed": SEED,
        "epoch": EPOCH,
        "batches": total_batches,
        "batch_size": int(spec["batch_size"]),
        "num_instances": int(spec["num_instances"]),
        "steps_per_epoch_per_rank": int(spec["steps_per_epoch"]),
        "budget_zero_batches": zero_batches,
        "budget_zero_rate": zero_batches / max(total_batches, 1),
        "candidate_cross_mean": float(np.mean(candidate_cross)),
        "candidate_same_mean": float(np.mean(candidate_same)),
        "candidate_id_mean": float(np.mean(candidate_id)),
        "selected_mean": float(np.mean(selected_counts)),
        "selected_total": len(all_rows),
        "selected_cross": selected_cross,
        "selected_same": selected_same,
        "selected_cross_rate": selected_cross / max(len(all_rows), 1),
        "selected_same_rate": selected_same / max(len(all_rows), 1),
        "covered_pid_count": len(per_pid),
        "covered_state_count": len(per_state),
        "pid_count_min": min(per_pid.values()) if per_pid else 0,
        "pid_count_max": max(per_pid.values()) if per_pid else 0,
        "pid_count_median": median(per_pid.values()) if per_pid else 0,
        "bank_source": bank["source"],
        "train_only": True,
        "query_gallery_used": False,
        "candidate_zero_semantics": "budget=min(cross,same,id); selected id candidates; exact current hash selector",
    }
    return summary, all_rows


def load_model(dataset_name: str, checkpoint: Path, device: torch.device, train: bool) -> torch.nn.Module:
    cfg = tfg.cfg_for(dataset_name)
    spec = tfg.SPEC[dataset_name]
    num_classes = len({item["pid"] for item in (make_prcc_items() if dataset_name == "prcc" else make_ltcc_items())})
    camera_num = 3 if dataset_name == "prcc" else 12
    model = build_model(cfg, num_classes=num_classes, camera_num=camera_num, pretrained=False)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"checkpoint mismatch {checkpoint}: {result}")
    del state
    model.to(device)
    model.train(train)
    return model


def prepare_batches(items: list[dict[str, Any]], dataset_name: str, max_batches: int) -> list[dict[str, Any]]:
    cfg = tfg.cfg_for(dataset_name)
    transform, _ = tfg.build_img_transforms(cfg)
    result = []
    for rank in (0, 1):
        set_seed(SEED)
        sampler = tfg.FinalBatchSampler(
            items,
            int(tfg.SPEC[dataset_name]["batch_size"]),
            int(tfg.SPEC[dataset_name]["num_instances"]),
            int(tfg.SPEC[dataset_name]["steps_per_epoch"]),
            SEED,
            rank,
        )
        sampler.prepare(EPOCH)
        loader = DataLoader(
            ImageDataset(items, transform=transform),
            batch_sampler=sampler,
            num_workers=0,
            pin_memory=False,
        )
        for iteration, batch in enumerate(loader, start=1):
            if iteration > max_batches:
                break
            result.append(
                {
                    "dataset": dataset_name,
                    "rank": rank,
                    "iteration": iteration,
                    "image": batch["image"].clone(),
                    "pid": batch["pid"].clone().long(),
                    "camid": batch["camid"].clone().long(),
                    "clothes_id": batch["clothes_id"].clone().long(),
                    "image_path": list(batch["image_path"]),
                }
            )
        del loader
    return result


def triplet_details(feat0: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor]:
    dist = euclidean_dist(feat0, feat0)
    dist_ap, dist_an, p_inds, n_inds = hard_example_mining(dist, labels, return_inds=True)
    per_anchor = F.softplus(dist_ap - dist_an)
    return {
        "dist_ap": dist_ap,
        "dist_an": dist_an,
        "p_inds": p_inds,
        "n_inds": n_inds,
        "per_anchor": per_anchor,
        "soft_total": per_anchor.mean(),
        "satisfied": dist_an >= dist_ap,
        "hinge_active": dist_ap > dist_an,
    }


def feature_from_output(model: torch.nn.Module, images: torch.Tensor, cams: torch.Tensor, with_grad: bool) -> torch.Tensor:
    if with_grad:
        output = model(images, cam_label=cams)
    else:
        with torch.no_grad():
            output = model(images, cam_label=cams)
    if isinstance(output, tuple):
        _, feat = output
        return feat[0]
    return output


def gradient_audit(
    dataset_name: str,
    items: list[dict[str, Any]],
    bank: dict[str, Any],
    batches: list[dict[str, Any]],
    checkpoint: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    model = load_model(dataset_name, checkpoint, DEVICE, train=True)
    pair_rows: list[dict[str, Any]] = []
    grad_rows: list[dict[str, Any]] = []
    case: dict[str, Any] | None = None
    for batch_data in batches:
        entries = [
            {
                "image_path": batch_data["image_path"][i],
                "pid": int(batch_data["pid"][i]),
                "clothes_id": int(batch_data["clothes_id"][i]),
            }
            for i in range(len(batch_data["pid"]))
        ]
        selected, info = tfg.select_gpc_pairs(
            entries,
            bank,
            EPOCH,
            int(batch_data["rank"]),
            int(batch_data["iteration"]),
            SEED,
        )
        images = batch_data["image"].to(DEVICE)
        labels = batch_data["pid"].to(DEVICE)
        cams = batch_data["camid"].to(DEVICE)
        feat0 = feature_from_output(model, images, cams, with_grad=True)
        tri = triplet_details(feat0, labels)
        g_tri = torch.autograd.grad(tri["soft_total"], feat0, retain_graph=True)[0]
        gpc, pair_values = tfg.gpc_relation_loss(feat0, selected)
        if selected:
            g_gpc = torch.autograd.grad(gpc, feat0, retain_graph=True)[0]
        else:
            g_gpc = torch.zeros_like(feat0)
        flat_tri = g_tri.detach().float().reshape(-1)
        flat_gpc = g_gpc.detach().float().reshape(-1)
        global_cos = F.cosine_similarity(flat_tri, flat_gpc, dim=0).item() if torch.linalg.vector_norm(flat_gpc) > 0 else 0.0
        grad_rows.append(
            {
                "dataset": dataset_name,
                "checkpoint": str(checkpoint),
                "rank": batch_data["rank"],
                "iteration": batch_data["iteration"],
                "selected_count": len(selected),
                "gpc_loss_raw": float(gpc.detach().cpu()),
                "gpc_loss_weighted_0p1": float((0.1 * gpc).detach().cpu()),
                "triplet_loss_softmargin": float(tri["soft_total"].detach().cpu()),
                "triplet_hinge_active_anchor_rate": float(tri["hinge_active"].float().mean().cpu()),
                "triplet_operational_satisfied_anchor_rate": float(tri["satisfied"].float().mean().cpu()),
                "triplet_grad_norm": float(torch.linalg.vector_norm(g_tri.detach().float()).cpu()),
                "gpc_grad_norm_raw": float(torch.linalg.vector_norm(g_gpc.detach().float()).cpu()),
                "gpc_grad_norm_weighted_0p1": float(torch.linalg.vector_norm(0.1 * g_gpc.detach().float()).cpu()),
                "grad_cosine_gpc_vs_triplet": global_cos,
                "softmargin_exact_zero": False,
                "autograd_mode": "FP32; no autocast; no optimizer step",
            }
        )
        for pair_index, (i, j, image_key, state_key) in enumerate(selected):
            vi = int(i)
            vj = int(j)
            pair_value = float(pair_values[pair_index].detach().cpu())
            pair_grad = torch.autograd.grad(pair_values[pair_index], feat0, retain_graph=True)[0]
            hp_i = int(tri["p_inds"][vi]) == vj
            hp_j = int(tri["p_inds"][vj]) == vi
            sat_i = bool(tri["satisfied"][vi].item())
            sat_j = bool(tri["satisfied"][vj].item())
            row = {
                "dataset": dataset_name,
                "checkpoint": str(checkpoint),
                "rank": batch_data["rank"],
                "iteration": batch_data["iteration"],
                "local_i": vi,
                "local_j": vj,
                "image_i": entries[vi]["image_path"],
                "image_j": entries[vj]["image_path"],
                "pid": entries[vi]["pid"],
                "clothes_i": entries[vi]["clothes_id"],
                "clothes_j": entries[vj]["clothes_id"],
                "relation": "cross_clothes" if entries[vi]["clothes_id"] != entries[vj]["clothes_id"] else "same_clothes",
                "gpc_pair_cosine_similarity": 1.0 - pair_value,
                "gpc_pair_loss": pair_value,
                "gpc_pair_loss_active": pair_value > 1e-12,
                "triplet_hard_positive_i": int(tri["p_inds"][vi]),
                "triplet_hard_positive_j": int(tri["p_inds"][vj]),
                "triplet_hard_negative_i": int(tri["n_inds"][vi]),
                "triplet_hard_negative_j": int(tri["n_inds"][vj]),
                "hard_positive_direct_i": hp_i,
                "hard_positive_direct_j": hp_j,
                "A_hard_positive_overlap": hp_i or hp_j,
                "B_not_hard_positive_either_direction": not (hp_i or hp_j),
                "dist_ap_i": float(tri["dist_ap"][vi].detach().cpu()),
                "dist_an_i": float(tri["dist_an"][vi].detach().cpu()),
                "dist_ap_j": float(tri["dist_ap"][vj].detach().cpu()),
                "dist_an_j": float(tri["dist_an"][vj].detach().cpu()),
                "triplet_soft_loss_i": float(tri["per_anchor"][vi].detach().cpu()),
                "triplet_soft_loss_j": float(tri["per_anchor"][vj].detach().cpu()),
                "triplet_operational_satisfied_i": sat_i,
                "triplet_operational_satisfied_j": sat_j,
                "C_both_endpoint_relative_constraints_satisfied": sat_i and sat_j,
                "D_at_least_one_endpoint_hinge_active": not (sat_i and sat_j),
                "triplet_exact_zero_i": False,
                "triplet_exact_zero_j": False,
                "gpc_pair_grad_norm_raw": float(torch.linalg.vector_norm(pair_grad.detach().float()).cpu()),
                "gpc_pair_grad_norm_weighted_0p1": float(torch.linalg.vector_norm(0.1 * pair_grad.detach().float()).cpu()),
            }
            pair_rows.append(row)
            if case is None and sat_i and sat_j and pair_value > 1e-8:
                case = {
                    "dataset": dataset_name,
                    "checkpoint": str(checkpoint),
                    "rank": batch_data["rank"],
                    "iteration": batch_data["iteration"],
                    "local_i": vi,
                    "local_j": vj,
                    "image_i": entries[vi]["image_path"],
                    "image_j": entries[vj]["image_path"],
                    "pid": entries[vi]["pid"],
                    "clothes_relation": row["relation"],
                    "gpc_cosine": 1.0 - pair_value,
                    "gpc_loss": pair_value,
                    "dist_ap_i": row["dist_ap_i"],
                    "dist_an_i": row["dist_an_i"],
                    "dist_ap_j": row["dist_ap_j"],
                    "dist_an_j": row["dist_an_j"],
                    "triplet_soft_loss_i": row["triplet_soft_loss_i"],
                    "triplet_soft_loss_j": row["triplet_soft_loss_j"],
                    "triplet_relative_constraint_satisfied_both": True,
                    "triplet_exact_zero_both": False,
                    "gpc_pair_grad_norm_raw": row["gpc_pair_grad_norm_raw"],
                    "gpc_pair_grad_norm_weighted_0p1": row["gpc_pair_grad_norm_weighted_0p1"],
                    "interpretation": "SoftMargin Triplet remains mathematically nonzero; the pair satisfies the operational relative ordering while GPC has nonzero direct gradient.",
                }
        del feat0, tri, gpc, pair_values, g_tri, g_gpc
        torch.cuda.empty_cache()
    del model
    torch.cuda.empty_cache()
    return pair_rows, grad_rows, case or {"status": "NO_CASE_IN_PROBED_BATCHES"}


def collect_eval_features(dataset_name: str, checkpoint: Path, batches: list[dict[str, Any]]) -> list[torch.Tensor]:
    model = load_model(dataset_name, checkpoint, DEVICE, train=False)
    out = []
    for batch in batches:
        feat = feature_from_output(
            model,
            batch["image"].to(DEVICE),
            batch["camid"].to(DEVICE),
            with_grad=False,
        )
        out.append(feat.detach().float().cpu())
    del model
    torch.cuda.empty_cache()
    return out


def bootstrap_ci(values_by_pid: dict[int, float], seed: int = SEED, draws: int = 2000) -> tuple[float | None, float | None]:
    if not values_by_pid:
        return None, None
    values = np.asarray(list(values_by_pid.values()), dtype=float)
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def fixed_pair_comparison(
    dataset_name: str,
    items: list[dict[str, Any]],
    bank: dict[str, Any],
    batches: list[dict[str, Any]],
    bio_checkpoint: Path,
    gpc_checkpoint: Path,
) -> list[dict[str, Any]]:
    bio_features = collect_eval_features(dataset_name, bio_checkpoint, batches)
    gpc_features = collect_eval_features(dataset_name, gpc_checkpoint, batches)
    rows = []
    for batch, bio, gpc in zip(batches, bio_features, gpc_features):
        entries = [
            {"image_path": batch["image_path"][i], "pid": int(batch["pid"][i]), "clothes_id": int(batch["clothes_id"][i])}
            for i in range(len(batch["pid"]))
        ]
        labels = batch["pid"].long()
        bio_dist = euclidean_dist(bio, bio)
        bap, ban, bpi, bni = hard_example_mining(bio_dist, labels, return_inds=True)
        sat = ban >= bap
        bio_norm = F.normalize(bio, dim=-1)
        gpc_norm = F.normalize(gpc, dim=-1)
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                if entries[i]["pid"] != entries[j]["pid"]:
                    continue
                hp = int(bpi[i]) == j or int(bpi[j]) == i
                groups = ["G1_hard_positive" if hp else "G2_not_direct_hard_positive"]
                if bool(sat[i]) and bool(sat[j]):
                    groups.append("G3_bio_relative_constraint_satisfied")
                groups.append("G4_cross_clothes" if entries[i]["clothes_id"] != entries[j]["clothes_id"] else "G5_same_clothes")
                s_bio = float(torch.sum(bio_norm[i] * bio_norm[j]))
                s_gpc = float(torch.sum(gpc_norm[i] * gpc_norm[j]))
                for group in groups:
                    rows.append(
                        {
                            "dataset": dataset_name,
                            "rank": batch["rank"],
                            "iteration": batch["iteration"],
                            "pid": entries[i]["pid"],
                            "image_i": entries[i]["image_path"],
                            "image_j": entries[j]["image_path"],
                            "group": group,
                            "bio_similarity": s_bio,
                            "bio_gpc_similarity": s_gpc,
                            "delta_similarity": s_gpc - s_bio,
                            "bio_cosine_distance": 1.0 - s_bio,
                            "bio_gpc_cosine_distance": 1.0 - s_gpc,
                            "bio_hard_positive_i": int(bpi[i]),
                            "bio_hard_positive_j": int(bpi[j]),
                            "bio_relative_satisfied_i": bool(sat[i]),
                            "bio_relative_satisfied_j": bool(sat[j]),
                            "strict_pair_status": "NOT_STRICT_SEPARATE_HISTORICAL_RUNNERS",
                        }
                    )
            for i in range(len(entries)):
                pos = int(bpi[i])
                neg = int(bni[i])
                pos_bio = float(torch.sum(bio_norm[i] * bio_norm[pos]))
                pos_gpc = float(torch.sum(gpc_norm[i] * gpc_norm[pos]))
                neg_bio = float(torch.sum(bio_norm[i] * bio_norm[neg]))
                neg_gpc = float(torch.sum(gpc_norm[i] * gpc_norm[neg]))
                rows.append(
                    {
                        "dataset": dataset_name,
                        "rank": batch["rank"],
                        "iteration": batch["iteration"],
                        "pid": entries[i]["pid"],
                        "image_i": entries[i]["image_path"],
                        "image_j": entries[neg]["image_path"],
                        "group": "G6_bio_batch_hard_negative",
                        "bio_similarity": neg_bio,
                        "bio_gpc_similarity": neg_gpc,
                        "delta_similarity": neg_gpc - neg_bio,
                        "bio_cosine_distance": 1.0 - neg_bio,
                        "bio_gpc_cosine_distance": 1.0 - neg_gpc,
                        "bio_hard_positive_i": pos,
                        "bio_hard_positive_j": neg,
                        "bio_hard_negative_i": neg,
                        "bio_relative_satisfied_i": bool(sat[i]),
                        "bio_identity_margin": pos_bio - neg_bio,
                        "bio_gpc_identity_margin_same_baseline_pair": pos_gpc - neg_gpc,
                        "strict_pair_status": "EXPLORATORY_ONLY_COMMON_INIT_NOT_STRICT_PAIR",
                    }
                )
    return rows


def summarize_fixed(rows: list[dict[str, Any]], dataset_name: str) -> list[dict[str, Any]]:
    out = []
    groups = sorted({row["group"] for row in rows})
    for group in groups:
        selected = [row for row in rows if row["group"] == group]
        by_pid: dict[int, list[float]] = defaultdict(list)
        for row in selected:
            by_pid[int(row["pid"])].append(float(row["delta_similarity"]))
        pid_delta = {pid: float(np.mean(values)) for pid, values in by_pid.items()}
        ci_low, ci_high = bootstrap_ci(pid_delta)
        bio = np.asarray([float(row["bio_similarity"]) for row in selected])
        gpc = np.asarray([float(row["bio_gpc_similarity"]) for row in selected])
        delta = np.asarray([float(row["delta_similarity"]) for row in selected])
        out.append(
            {
                "dataset": dataset_name,
                "group": group,
                "pids": len(by_pid),
                "pairs": len(selected),
                "bio_similarity_mean": float(np.mean(bio)),
                "bio_similarity_median": float(np.median(bio)),
                "bio_similarity_p10": float(np.quantile(bio, 0.10)),
                "bio_gpc_similarity_mean": float(np.mean(gpc)),
                "delta_similarity_mean": float(np.mean(delta)),
                "delta_similarity_median": float(np.median(delta)),
                "improvement_rate_delta_gt_0": float(np.mean(delta > 0)),
                "pid_bootstrap_95ci_low": ci_low,
                "pid_bootstrap_95ci_high": ci_high,
                "bio_identity_margin_mean": (
                    float(np.mean([float(row["bio_identity_margin"]) for row in selected if "bio_identity_margin" in row]))
                    if any("bio_identity_margin" in row for row in selected)
                    else None
                ),
                "bio_gpc_identity_margin_mean": (
                    float(np.mean([float(row["bio_gpc_identity_margin_same_baseline_pair"]) for row in selected if "bio_gpc_identity_margin_same_baseline_pair" in row]))
                    if any("bio_gpc_identity_margin_same_baseline_pair" in row for row in selected)
                    else None
                ),
                "status": "EXPLORATORY_ONLY_COMMON_INIT_NOT_STRICT_PAIR",
            }
        )
    return out


def provenance() -> dict[str, Any]:
    cfgs = {}
    for dataset_name in ("prcc", "ltcc"):
        cfg = tfg.cfg_for(dataset_name)
        cfgs[dataset_name] = {
            "loss_type": list(cfg.MODEL.LOSS_TYPE),
            "no_margin": bool(cfg.MODEL.NO_MARGIN),
            "sampler": str(cfg.DATA.SAMPLER),
            "batch_size": int(cfg.DATA.BATCH_SIZE),
            "num_instances": int(cfg.DATA.NUM_INSTANCES),
            "seed": int(cfg.SOLVER.SEED),
            "base_lr": float(cfg.SOLVER.BASE_LR),
            "warmup_epochs": int(cfg.SOLVER.WARMUP_EPOCHS),
            "max_epochs": int(cfg.SOLVER.MAX_EPOCHS),
            "clip_dim": int(cfg.MODEL.CLIP_DIM),
            "bio_index": str(cfg.DATA.BIO_INDEX),
            "nonbio_index": list(cfg.DATA.NOBIO_INDEX),
            "nonbio_head": bool(cfg.MODEL.NONBIO_HEAD),
            "last_layer": str(cfg.MODEL.LAST_LAYER),
            "gpc_lambda": 0.1,
            "gpc_namespace": tfg.PAIR_HASH_NAMESPACE,
        }
    return {
        "current_wrapper": str(ROOT / "scripts/train_final_gpc.py"),
        "current_wrapper_sha256": sha256_file(ROOT / "scripts/train_final_gpc.py"),
        "current_gpc_sha256": sha256_file(ROOT / "loss/gpc_relation.py"),
        "current_triplet_sha256": sha256_file(ROOT / "loss/triplet_loss.py"),
        "deployed_triplet_sha256": sha256_file(ENGINE / "loss/triplet_loss.py"),
        "deployed_make_loss_sha256": sha256_file(ENGINE / "loss/make_loss.py"),
        "configs": cfgs,
        "checkpoints": {
            name: {"path": str(path), "exists": path.is_file(), "sha256": sha256_file(path)}
            for name, path in CHECKPOINTS.items()
        },
        "common_init_sha256_from_manifest": {
            "prcc_bio": "db5c71389b37a859ec82b3e529c1cd7149c21a85df4a1b34f9650c9f6bd9378f",
            "prcc_gpc_manifest_source": "/data/DIFFER/outputs/new_method_track/phase7a_prcc_relation_validation/PRCC_COMMON_INIT.pth",
        },
        "strict_pair": {
            "prcc": False,
            "ltcc": False,
            "reason": "Bio and Bio+GPC are separate historical runs; shared seed/common-init evidence exists for PRCC, but there is no same-process bitwise paired continuation and runner/source snapshots differ.",
        },
        "formal_training_started": False,
        "optimizer_step_called": False,
        "scheduler_step_called": False,
        "official_query_gallery_used": False,
        "checkpoint_written_by_diagnostic": False,
        "rank_state_written_by_diagnostic": False,
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/PPU unavailable in actual PyTorch environment")
    set_seed()
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "runtime_provenance.json", provenance())
    summary_rows = []
    coverage_rows = []
    pair_rows_all = []
    grad_rows_all = []
    cases = {}
    fixed_summary = []
    for dataset_name, make_items in (("prcc", make_prcc_items), ("ltcc", make_ltcc_items)):
        print(f"[audit] metadata {dataset_name}", flush=True)
        items = make_items()
        bank = make_bank(items, dataset_name)
        summary, coverage_rows_local = selector_scan(items, bank, dataset_name)
        summary_rows.append(summary)
        coverage_rows.extend(coverage_rows_local)
        batches = prepare_batches(items, dataset_name, MAX_BATCHES)
        print(f"[audit] gradient {dataset_name} batches={len(batches)}", flush=True)
        checkpoint = CHECKPOINTS[f"{dataset_name}_gpc"]
        pair_rows, grad_rows, case = gradient_audit(dataset_name, items, bank, batches, checkpoint)
        pair_rows_all.extend(pair_rows)
        grad_rows_all.extend(grad_rows)
        cases[dataset_name] = case
        if dataset_name == "prcc" and CHECKPOINTS["prcc_bio"].is_file():
            print("[audit] exploratory fixed-pair feature comparison PRCC", flush=True)
            fixed_rows = fixed_pair_comparison(
                dataset_name,
                items,
                bank,
                batches,
                CHECKPOINTS["prcc_bio"],
                CHECKPOINTS["prcc_gpc"],
            )
            fixed_summary.extend(summarize_fixed(fixed_rows, dataset_name))
            write_csv(OUT / "gpc_triplet_fixed_pair_observations_prcc.csv", fixed_rows)
        else:
            for group in (
                "G1_hard_positive",
                "G2_not_direct_hard_positive",
                "G3_bio_relative_constraint_satisfied",
                "G4_cross_clothes",
                "G5_same_clothes",
            ):
                fixed_summary.append(
                    {
                        "dataset": dataset_name,
                        "group": group,
                        "status": "N/A_NO_STRICT_BIO_BIO_GPC_PAIR",
                    }
                )
    write_json(OUT / "gpc_triplet_summary.json", {"coverage": summary_rows, "cases": cases, "fixed_pair_summary": fixed_summary})
    write_csv(OUT / "gpc_triplet_pair_coverage.csv", coverage_rows)
    write_csv(OUT / "gpc_triplet_gradient_audit.csv", grad_rows_all)
    write_csv(OUT / "gpc_triplet_pair_categories.csv", pair_rows_all)
    write_csv(OUT / "gpc_triplet_fixed_pair_comparison.csv", fixed_summary)
    print(json.dumps({"output": str(OUT), "coverage": summary_rows, "cases": cases, "fixed_pair_summary": fixed_summary}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
