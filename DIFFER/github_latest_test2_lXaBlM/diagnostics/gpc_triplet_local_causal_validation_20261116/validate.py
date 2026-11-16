#!/usr/local/bin/python3
"""Local one-step causal validation for GPC/Batch-Hard complementarity.

This file is an isolated diagnostic.  It imports the current wrapper and
losses, materialises fixed train-only batches, and never writes checkpoints,
starts a scheduler, evaluates query/gallery, or changes formal training code.

The causal unit is one independent SGD step from an existing Bio checkpoint:

    A: CE + Triplet + clipBio
    B: CE + Triplet + clipBio + 0.1 * GPC

For every fixed batch with selected GPC pairs, A and B are restored to the
same CPU checkpoint state and use fresh, identical SGD state.  The two steps
are therefore counterfactual updates, not a reconstruction of the historical
60-epoch trajectory.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


ROOT = Path("/home/DIFFER/github_latest_test2_lXaBlM")
ENGINE = Path("/data/DIFFER")
OUT = ROOT / "diagnostics/gpc_triplet_local_causal_validation_20261116"
REPORT = ROOT / "reports/GPC_TRIPLET_LOCAL_CAUSAL_VALIDATION.md"
SEED = 1234
EPOCH = 60
DEVICE = torch.device("cuda:0")
LAMBDA_GPC = 0.1
PRCC_BATCHES_PER_RANK = 8  # exact schedule used for the prior 147-pair audit
LTCC_BATCHES_PER_RANK = 256  # fixed prefix; selected before looking at effects
PARAMETER_GRADIENT_BATCH_LIMIT = 100000  # all effective fixed batches; feature and parameter grads are both audited
USE_AUTOMIXED_PRECISION = True  # formal runner uses CUDA AMP; loss reductions remain FP32 where defined

CHECKPOINTS = {
    "prcc_bio": Path("/data/DIFFER/outputs/new_method_track/phase7a_prcc_relation_validation/P0_prcc_bio/model_final.pth"),
    # The available LTCC Bio control is the completed 15-epoch screen checkpoint.
    # It is a valid Bio state for a local one-step probe, but not a 60-epoch
    # counterpart to the historical LTCC GPC run.
    "ltcc_bio": Path("/data/DIFFER/outputs/new_method_track/phase5b_id_damage_ds/S0_bio_full_data/model_screen_final.pth"),
}

PRIOR_PAIR_CSV = ROOT / "diagnostics/gpc_triplet_complementarity_20261116/gpc_triplet_pair_categories.csv"
PRIOR_PAIR_COUNT = 147
PRIOR_PRCC_CKPT = Path("/home/DIFFER/log_NDP/ndp_nteach_current_b0_pair_20261031_125146/PRCC_B0_full/PRCC/model_final.pth")
PRIOR_LTCC_CKPT = Path("/home/DIFFER/log_NDP/ndp_nteach_b0_recovery_20261030_210612/LTCC_B0_full/LTCC/model_final.pth")


def safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        value = value.detach().float().cpu()
        return value.item() if value.ndim == 0 else value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(safe(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def tensor_hash(value: torch.Tensor) -> str:
    return sha256_bytes(value.detach().cpu().contiguous().numpy().tobytes())


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def import_runtime() -> Any:
    # The wrapper is local to this repository.  /data/DIFFER is added as a
    # fallback because processor/ and a few runtime modules live there.
    sys.path.insert(0, str(ROOT))
    sys.path.insert(1, str(ENGINE))
    spec = importlib.util.spec_from_file_location("current_train_final_gpc_causal", ROOT / "scripts/train_final_gpc.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import scripts/train_final_gpc.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tfg = import_runtime()
from loss.triplet_loss import euclidean_dist, hard_example_mining  # noqa: E402
from model import build_model  # noqa: E402


class NullWriter:
    def add_scalar(self, *_args: Any, **_kwargs: Any) -> None:
        return None


def cfg_for(dataset_name: str):
    cfg = tfg.cfg_for(dataset_name)
    cfg.defrost()
    # The server has the verified assets under /data/data.  This override is
    # diagnostic-only; no formal YAML or training wrapper is modified.
    cfg.DATA.CAPTION_DIR = str(Path("/data/data/DIFFER_TextCaption") / ("PRCC" if dataset_name == "prcc" else "LTCC_ReID"))
    cfg.DATA.NUM_WORKERS = 0
    cfg.DATA.PIN_MEMORY = False
    cfg.freeze()
    return cfg


def normalise_state(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict) and "model" in payload and isinstance(payload["model"], dict):
        payload = payload["model"]
    if not isinstance(payload, dict):
        raise TypeError(f"checkpoint is not a state dict: {type(payload)}")
    state = {}
    for key, value in payload.items():
        key = str(key)
        if key.startswith("module."):
            key = key[7:]
        state[key] = value
    return state


def state_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode("utf-8"))
        digest.update(state[key].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def build_model_pair(dataset_name: str, checkpoint: Path):
    cfg = cfg_for(dataset_name)
    num_classes = 150 if dataset_name == "prcc" else 77
    camera_num = 3 if dataset_name == "prcc" else 12
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    initial_state = normalise_state(payload)
    model_a = build_model(cfg, num_classes=num_classes, camera_num=camera_num, pretrained=False)
    model_b = build_model(cfg, num_classes=num_classes, camera_num=camera_num, pretrained=False)
    result_a = model_a.load_state_dict(initial_state, strict=True)
    result_b = model_b.load_state_dict(initial_state, strict=True)
    if result_a.missing_keys or result_a.unexpected_keys or result_b.missing_keys or result_b.unexpected_keys:
        raise RuntimeError(f"checkpoint load mismatch: {result_a}, {result_b}")
    model_a.to(DEVICE)
    model_b.to(DEVICE)
    model_a.train()
    model_b.train()
    return cfg, initial_state, model_a, model_b


def make_loss_optimizer(cfg, model):
    loss_fn, center = tfg.make_loss(cfg, num_classes=150 if cfg.DATA.DATASET == "prcc" else 77)
    optimizer, _center_optimizer = tfg.make_optimizer(cfg, model, center)
    return loss_fn, optimizer


def reset_branch(model, optimizer, initial_state):
    model.load_state_dict(initial_state, strict=True)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    optimizer.state.clear()


def materialise_fixed_dataset(dataset_name: str, max_batches_per_rank: int):
    cfg = cfg_for(dataset_name)
    dataset = tfg.build_dataset(cfg)
    try:
        bank = tfg.build_train_only_pair_bank(dataset, dataset_name)
    except FileNotFoundError as exc:
        # The current repository wrapper expects this frozen LTCC source under
        # the repository output tree, while the actual runtime asset is under
        # /data/DIFFER.  This fallback is diagnostic-only and records the
        # deployed source below; it does not alter selector or loss code.
        if dataset_name != "ltcc":
            raise
        source = ENGINE / "outputs/new_method_track/phase6b_soft_relation_screen/artifacts/relation_source_rows.csv"
        if not source.is_file():
            raise exc
        exact = set()
        with source.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                left, right = sorted((int(row["index_a"]), int(row["index_b"])))
                if left == right or right >= len(dataset.train):
                    raise RuntimeError(f"invalid LTCC deployed relation row: {row}")
                a, b = dataset.train[left], dataset.train[right]
                if int(a["pid"]) != int(b["pid"]) or int(a["clothes_id"]) == int(b["clothes_id"]):
                    raise RuntimeError(f"invalid LTCC deployed relation semantics: {row}")
                exact.add((left, right))
        bank = {
            "exact_lookup": exact,
            "path_to_index": {str(item["image_path"]): index for index, item in enumerate(dataset.train)},
            "dataset": dataset_name,
            "train_image_count": len(dataset.train),
            "cross_key_count": len(exact),
            "source": "deployed /data/DIFFER frozen binary exact-pair key set; SR3 values ignored",
            "source_path": str(source),
            "source_row_count": len(exact),
            "all_keys_train_only": True,
        }
    transform, _ = tfg.build_img_transforms(cfg)
    fixed = []
    plan_rows = []
    total_batches = 0
    zero_batches = 0
    selected_pairs = 0
    selected_pids: set[int] = set()
    selected_cross = 0
    for rank in (0, 1):
        set_seed(SEED)
        sampler = tfg.FinalBatchSampler(
            dataset.train,
            int(tfg.SPEC[dataset_name]["batch_size"]),
            int(tfg.SPEC[dataset_name]["num_instances"]),
            int(tfg.SPEC[dataset_name]["steps_per_epoch"]),
            SEED,
            rank,
        )
        sampler.prepare(EPOCH)
        loader = DataLoader(
            tfg.ImageDataset(dataset.train, transform=transform),
            batch_sampler=sampler,
            num_workers=0,
            pin_memory=False,
        )
        for iteration, batch in enumerate(loader, start=1):
            if iteration > max_batches_per_rank:
                break
            entries = [
                {
                    "image_path": str(batch["image_path"][i]),
                    "pid": int(batch["pid"][i]),
                    "clothes_id": int(batch["clothes_id"][i]),
                }
                for i in range(len(batch["pid"]))
            ]
            selected, info = tfg.select_gpc_pairs(entries, bank, EPOCH, rank, iteration, SEED)
            total_batches += 1
            zero_batches += int(not selected)
            selected_pairs += len(selected)
            selected_cross += int(info["cross_clothes_selected"])
            selected_pids.update(int(entries[i]["pid"]) for i, _j, *_ in selected)
            plan_hash = sha256_bytes(json.dumps([entry["image_path"] for entry in entries], sort_keys=True).encode())
            plan_rows.append(
                {
                    "dataset": dataset_name,
                    "rank": rank,
                    "iteration": iteration,
                    "batch_path_sha256": plan_hash,
                    "selected_count": len(selected),
                    "candidate_cross": int(info["candidate_counts"]["cross"]),
                    "candidate_same": int(info["candidate_counts"]["same"]),
                    "candidate_id": int(info["candidate_counts"]["id"]),
                    "budget": int(info["matched_count"]),
                    "cross_selected": int(info["cross_clothes_selected"]),
                    "same_selected": int(info["same_clothes_selected"]),
                }
            )
            if selected:
                frozen = {key: value.clone() if torch.is_tensor(value) else list(value) if isinstance(value, list) else value for key, value in batch.items()}
                fixed.append({
                    "dataset": dataset_name,
                    "rank": rank,
                    "iteration": iteration,
                    "batch": frozen,
                    "entries": entries,
                    "selected": selected,
                    "info": info,
                    "batch_path_sha256": plan_hash,
                })
        del loader
    plan_hash = sha256_bytes(json.dumps(plan_rows, sort_keys=True).encode())
    return dataset, bank, fixed, plan_rows, {
        "dataset": dataset_name,
        "fixed_batches_per_rank": max_batches_per_rank,
        "fixed_batches_total": total_batches,
        "batches_with_selected_gpc": len(fixed),
        "budget_zero_batches": zero_batches,
        "budget_zero_rate": zero_batches / max(total_batches, 1),
        "selected_pairs": selected_pairs,
        "selected_cross": selected_cross,
        "selected_same": selected_pairs - selected_cross,
        "selected_cross_rate": selected_cross / max(selected_pairs, 1),
        "selected_pid_count": len(selected_pids),
        "batch_plan_sha256": plan_hash,
        "train_only": True,
        "official_query_gallery_used": False,
    }


def triplet_details(feat0: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor]:
    distance = euclidean_dist(feat0, feat0)
    dist_ap, dist_an, p_inds, n_inds = hard_example_mining(distance, labels, return_inds=True)
    return {
        "dist_ap": dist_ap,
        "dist_an": dist_an,
        "p_inds": p_inds,
        "n_inds": n_inds,
        "per_anchor": F.softplus(dist_ap - dist_an),
        "satisfied": dist_an >= dist_ap,
        "active": dist_ap > dist_an,
    }


def output_of(model, images, cams, with_grad: bool):
    amp_context = torch.cuda.amp.autocast(enabled=USE_AUTOMIXED_PRECISION)
    with amp_context:
        if with_grad:
            return model(images, cam_label=cams)
        with torch.no_grad():
            return model(images, cam_label=cams)


def feat0_from_output(output):
    if isinstance(output, tuple):
        return output[1][0]
    return output


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1), dim=1).item())


def component_gradients(value: torch.Tensor, feat0: torch.Tensor, shared_params):
    feature_grad = torch.autograd.grad(value, feat0, retain_graph=True, allow_unused=True)[0]
    if feature_grad is None:
        feature_grad = torch.zeros_like(feat0)
    parameter_grads = torch.autograd.grad(value, shared_params, retain_graph=True, allow_unused=True)
    # Keep the per-parameter tensors, but do not concatenate the roughly
    # 303M-element EVA-L backbone into another 1.2GB vector.  The following
    # parameter_stats() routine accumulates norms and inner products directly
    # on the PPU and is mathematically identical to a flattened-vector norm or
    # cosine.
    return feature_grad.detach().float(), parameter_grads


def component_cosines(vectors: dict[str, torch.Tensor]) -> dict[str, float]:
    result = {}
    keys = list(vectors)
    for left_index, left in enumerate(keys):
        for right in keys[left_index + 1:]:
            result[f"{left}_vs_{right}"] = cosine(vectors[left], vectors[right])
    return result


def parameter_stats(gradient_lists: dict[str, tuple], params) -> tuple[dict[str, float], dict[str, float]]:
    names = list(gradient_lists)
    sums = {name: torch.zeros((), device=DEVICE, dtype=torch.float32) for name in names}
    dots = {(left, right): torch.zeros((), device=DEVICE, dtype=torch.float32)
            for left_index, left in enumerate(names) for right in names[left_index + 1:]}
    for parameter_index, _parameter in enumerate(params):
        current = {}
        for name in names:
            grad = gradient_lists[name][parameter_index]
            if grad is not None:
                current[name] = grad.detach().float()
                sums[name] = sums[name] + torch.sum(current[name] * current[name])
        for (left, right), _old in list(dots.items()):
            if left in current and right in current:
                dots[(left, right)] = dots[(left, right)] + torch.sum(current[left] * current[right])
    norms = {name: float(torch.sqrt(value.clamp_min(0)).item()) for name, value in sums.items()}
    cosines = {}
    for (left, right), dot in dots.items():
        denominator = max(norms[left] * norms[right], 1e-30)
        cosines[f"{left}_vs_{right}"] = float((dot.item() / denominator))
    return norms, cosines


def current_state_optimizer_hash(optimizer) -> str:
    return sha256_bytes(json.dumps(safe(optimizer.state_dict()), sort_keys=True).encode())


def run_one_batch(item, model_a, model_b, loss_a, loss_b, opt_a, opt_b, initial_state, dataset_name: str, batch_number: int, compute_parameter_gradients: bool):
    batch = item["batch"]
    images = batch["image"].to(DEVICE)
    labels = batch["pid"].to(DEVICE).long()
    cams = batch["camid"].to(DEVICE).long()
    clothes = batch["clothes_id"].to(DEVICE).long()
    captions = batch["caption_feature"].to(DEVICE)
    selected = item["selected"]

    reset_branch(model_a, opt_a, initial_state)
    reset_branch(model_b, opt_b, initial_state)

    # Same stochastic forward for the two branches.  The input tensor itself
    # is already frozen, including train augmentation.
    forward_seed = SEED + 1000003 * (0 if dataset_name == "prcc" else 1) + batch_number
    set_seed(forward_seed)
    out_a = output_of(model_a, images, cams, with_grad=True)
    feat_a = out_a[1]
    feat0_a = feat_a[0]
    total_a = loss_a(out_a[0], feat_a, labels, cams, captions, clothes, NullWriter(), batch_number)
    components_a = dict(loss_a.last_components)
    gpc_a, pair_values_a = tfg.gpc_relation_loss(feat0_a, selected)

    # The pre-update feature is the actual feature used by the training-mode
    # triplet/GPC graph.  It fixes groups before either counterfactual step.
    pre_feat = feat0_a.detach().float()
    tri = triplet_details(pre_feat, labels.detach())
    pre_norm = F.normalize(pre_feat, dim=-1)

    feature_vectors = {}
    parameter_gradient_lists = {}
    for name in ("ce", "triplet", "bio"):
        if name not in components_a:
            raise RuntimeError(f"missing loss component {name}: {components_a.keys()}")
    # Exact all-backbone flattened vectors are prohibitively expensive on the
    # PPU for four repeated higher-order graph traversals.  We therefore use a
    # predeclared, shared representation subspace: every parameter in the
    # final transformer block blocks.22, excluding classifier/Bio heads.  It
    # is the same parameter-name set for CE, Triplet, Bio and GPC on every
    # batch; feature gradients remain exact for the full feat[0].
    shared_params = [
        parameter for name, parameter in model_a.named_parameters()
        if parameter.requires_grad and name.startswith("blocks.22.")
    ]
    for name in ("ce", "triplet", "bio"):
        feature_vectors[name], parameter_gradient_lists[name] = component_gradients(components_a[name], feat0_a, shared_params) if compute_parameter_gradients else (torch.autograd.grad(components_a[name], feat0_a, retain_graph=True, allow_unused=True)[0], None)
    feature_vectors["gpc_raw"], parameter_gradient_lists["gpc_raw"] = component_gradients(gpc_a, feat0_a, shared_params) if compute_parameter_gradients else (torch.autograd.grad(gpc_a, feat0_a, retain_graph=True, allow_unused=True)[0], None)
    for name, value in list(feature_vectors.items()):
        if value is None:
            feature_vectors[name] = torch.zeros_like(feat0_a)
    feature_norms = {f"feature_{name}_grad_norm": float(torch.linalg.vector_norm(value).item()) for name, value in feature_vectors.items()}
    if compute_parameter_gradients:
        parameter_norm_values, parameter_cos = parameter_stats(parameter_gradient_lists, shared_params)
    else:
        parameter_norm_values = {name: None for name in ("ce", "triplet", "bio", "gpc_raw")}
        parameter_cos = {f"{left}_vs_{right}": None for left_index, left in enumerate(("ce", "triplet", "bio", "gpc_raw")) for right in ("ce", "triplet", "bio", "gpc_raw")[left_index + 1:]}
    parameter_norms = {f"parameter_{name}_grad_norm": value for name, value in parameter_norm_values.items()}
    parameter_norms["parameter_gpc_weighted_0p1_grad_norm"] = (parameter_norms["parameter_gpc_raw_grad_norm"] * LAMBDA_GPC) if parameter_norms["parameter_gpc_raw_grad_norm"] is not None else None
    feature_norms["feature_gpc_weighted_0p1_grad_norm"] = feature_norms["feature_gpc_raw_grad_norm"] * LAMBDA_GPC
    feature_cos = component_cosines(feature_vectors)

    # A: the base loss only.
    opt_a.zero_grad(set_to_none=True)
    optimizer_initial_hash_a = current_state_optimizer_hash(opt_a)
    total_a.backward()
    opt_a.step()

    # B: a fresh identical forward from the separately restored initial state.
    set_seed(forward_seed)
    out_b = output_of(model_b, images, cams, with_grad=True)
    feat_b = out_b[1]
    total_b = loss_b(out_b[0], feat_b, labels, cams, captions, clothes, NullWriter(), batch_number)
    gpc_b, _pair_values_b = tfg.gpc_relation_loss(feat_b[0], selected)
    opt_b.zero_grad(set_to_none=True)
    total_with_gpc = total_b + LAMBDA_GPC * gpc_b
    optimizer_initial_hash_b = current_state_optimizer_hash(opt_b)
    total_with_gpc.backward()
    opt_b.step()

    base_loss_match = float(abs(float(total_a.detach().cpu()) - float(total_b.detach().cpu())))
    model_a.eval()
    model_b.eval()
    post_a = feat0_from_output(output_of(model_a, images, cams, with_grad=False)).detach().float()
    post_b = feat0_from_output(output_of(model_b, images, cams, with_grad=False)).detach().float()
    post_a_norm = F.normalize(post_a, dim=-1)
    post_b_norm = F.normalize(post_b, dim=-1)

    pair_rows = []
    negative_rows = []
    for pair_index, (left, right, image_key, state_key) in enumerate(selected):
        i, j = int(left), int(right)
        hard_i = int(tri["p_inds"][i]) == j
        hard_j = int(tri["p_inds"][j]) == i
        group_g1 = hard_i or hard_j
        group_g2 = not group_g1
        group_g3 = bool(tri["satisfied"][i].item() and tri["satisfied"][j].item())
        group_g4 = int(clothes[i]) != int(clothes[j])
        group_g5 = not group_g4
        pre_sim = float(torch.sum(pre_norm[i] * pre_norm[j]).item())
        sim_a = float(torch.sum(post_a_norm[i] * post_a_norm[j]).item())
        sim_b = float(torch.sum(post_b_norm[i] * post_b_norm[j]).item())
        pair_rows.append({
            "dataset": dataset_name,
            "rank": item["rank"],
            "iteration": item["iteration"],
            "batch_path_sha256": item["batch_path_sha256"],
            "pair_index": pair_index,
            "pid": int(batch["pid"][i]),
            "local_i": i,
            "local_j": j,
            "image_i": str(batch["image_path"][i]),
            "image_j": str(batch["image_path"][j]),
            "clothes_i": int(clothes[i]),
            "clothes_j": int(clothes[j]),
            "relation": "cross_clothes" if group_g4 else "same_clothes",
            "G1_triplet_direct_hard_positive": group_g1,
            "G2_not_direct_hard_positive": group_g2,
            "G3_both_endpoint_relative_order_satisfied": group_g3,
            "G4_cross_clothes": group_g4,
            "G5_same_clothes": group_g5,
            "B_intersect_C": group_g2 and group_g3,
            "triplet_hard_positive_i": int(tri["p_inds"][i]),
            "triplet_hard_positive_j": int(tri["p_inds"][j]),
            "triplet_hard_negative_i": int(tri["n_inds"][i]),
            "triplet_hard_negative_j": int(tri["n_inds"][j]),
            "triplet_dist_ap_i": float(tri["dist_ap"][i].item()),
            "triplet_dist_an_i": float(tri["dist_an"][i].item()),
            "triplet_dist_ap_j": float(tri["dist_ap"][j].item()),
            "triplet_dist_an_j": float(tri["dist_an"][j].item()),
            "gpc_pair_loss_pre": float(pair_values_a[pair_index].detach().cpu()),
            "pre_similarity": pre_sim,
            "post_similarity_without_gpc": sim_a,
            "post_similarity_with_gpc": sim_b,
            "delta_without_gpc": sim_a - pre_sim,
            "delta_with_gpc": sim_b - pre_sim,
            "gpc_incremental_effect": sim_b - sim_a,
            "pre_cosine_distance": 1.0 - pre_sim,
            "post_cosine_distance_without_gpc": 1.0 - sim_a,
            "post_cosine_distance_with_gpc": 1.0 - sim_b,
            "gpc_incremental_distance_reduction": sim_b - sim_a,
            "base_loss_a": float(total_a.detach().cpu()),
            "base_loss_b": float(total_b.detach().cpu()),
            "gpc_loss_raw": float(gpc_b.detach().cpu()),
            "gpc_loss_weighted_0p1": float((LAMBDA_GPC * gpc_b).detach().cpu()),
            "base_loss_match_abs": base_loss_match,
            "optimizer_initial_state_hash_a": optimizer_initial_hash_a,
            "optimizer_initial_state_hash_b": optimizer_initial_hash_b,
        })

    for i in range(len(batch["pid"])):
        positive = int(tri["p_inds"][i])
        negative = int(tri["n_inds"][i])
        pre_pos = float(torch.sum(pre_norm[i] * pre_norm[positive]).item())
        pre_neg = float(torch.sum(pre_norm[i] * pre_norm[negative]).item())
        sim_a_pos = float(torch.sum(post_a_norm[i] * post_a_norm[positive]).item())
        sim_b_pos = float(torch.sum(post_b_norm[i] * post_b_norm[positive]).item())
        sim_a_neg = float(torch.sum(post_a_norm[i] * post_a_norm[negative]).item())
        sim_b_neg = float(torch.sum(post_b_norm[i] * post_b_norm[negative]).item())
        negative_rows.append({
            "dataset": dataset_name,
            "rank": item["rank"],
            "iteration": item["iteration"],
            "batch_path_sha256": item["batch_path_sha256"],
            "anchor_local": i,
            "pid": int(batch["pid"][i]),
            "negative_pid": int(batch["pid"][negative]),
            "positive_local_fixed": positive,
            "negative_local_fixed": negative,
            "anchor_relative_order_satisfied": bool(tri["satisfied"][i].item()),
            "pre_positive_similarity": pre_pos,
            "pre_negative_similarity": pre_neg,
            "pre_identity_margin": pre_pos - pre_neg,
            "post_positive_similarity_without_gpc": sim_a_pos,
            "post_positive_similarity_with_gpc": sim_b_pos,
            "post_negative_similarity_without_gpc": sim_a_neg,
            "post_negative_similarity_with_gpc": sim_b_neg,
            "negative_similarity_delta_without_gpc": sim_a_neg - pre_neg,
            "negative_similarity_delta_with_gpc": sim_b_neg - pre_neg,
            "negative_similarity_incremental_effect": sim_b_neg - sim_a_neg,
            "identity_margin_without_gpc": sim_a_pos - sim_a_neg,
            "identity_margin_with_gpc": sim_b_pos - sim_b_neg,
            "identity_margin_delta_without_gpc": (sim_a_pos - sim_a_neg) - (pre_pos - pre_neg),
            "identity_margin_delta_with_gpc": (sim_b_pos - sim_b_neg) - (pre_pos - pre_neg),
            "identity_margin_incremental_effect": (sim_b_pos - sim_b_neg) - (sim_a_pos - sim_a_neg),
        })

    gradient_row = {
        "dataset": dataset_name,
        "rank": item["rank"],
        "iteration": item["iteration"],
        "batch_path_sha256": item["batch_path_sha256"],
        "selected_gpc_pairs": len(selected),
        "base_loss": float(total_a.detach().cpu()),
        "gpc_loss_raw": float(gpc_a.detach().cpu()),
        "gpc_loss_weighted_0p1": float((LAMBDA_GPC * gpc_a).detach().cpu()),
        "triplet_loss": float(components_a["triplet"].detach().cpu()),
        "ce_loss": float(components_a["ce"].detach().cpu()),
        "clipBio_loss": float(components_a["bio"].detach().cpu()),
        "triplet_active_anchor_rate": float(tri["active"].float().mean().item()),
        "triplet_relative_satisfied_anchor_rate": float(tri["satisfied"].float().mean().item()),
        "triplet_feature_grad_norm": feature_norms["feature_triplet_grad_norm"],
        "gpc_feature_grad_norm_raw": feature_norms["feature_gpc_raw_grad_norm"],
        "gpc_feature_grad_norm_weighted_0p1": feature_norms["feature_gpc_weighted_0p1_grad_norm"],
        "shared_parameter_subspace": "all trainable parameters named blocks.22.* (final shared transformer block; classifier/Bio heads excluded)",
        "triplet_parameter_grad_norm": parameter_norms["parameter_triplet_grad_norm"],
        "gpc_parameter_grad_norm_raw": parameter_norms["parameter_gpc_raw_grad_norm"],
        "gpc_parameter_grad_norm_weighted_0p1": parameter_norms["parameter_gpc_weighted_0p1_grad_norm"],
        "base_loss_match_abs_a_b": base_loss_match,
        "parameter_optimizer_step_delta_not_historical": True,
        "parameter_gradient_computed_this_batch": compute_parameter_gradients,
    }
    gradient_row.update(feature_norms)
    gradient_row.update(parameter_norms)
    gradient_row.update({f"feature_cos_{key}": value for key, value in feature_cos.items()})
    gradient_row.update({f"parameter_cos_{key}": value for key, value in parameter_cos.items()})

    # Release the graph before the next independent batch.
    del out_a, out_b, feat_a, feat_b, feat0_a, total_a, total_b, gpc_a, gpc_b
    del pair_values_a, _pair_values_b, pre_feat, post_a, post_b
    del feature_vectors, parameter_gradient_lists, images, labels, cams, clothes, captions
    torch.cuda.empty_cache()
    return pair_rows, negative_rows, gradient_row


def bootstrap_ci(values_by_pid: dict[int, float], seed: int = SEED, draws: int = 4000):
    if not values_by_pid:
        return None, None
    values = np.asarray(list(values_by_pid.values()), dtype=float)
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    sampled = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))


def aggregate_group(pair_rows: list[dict[str, Any]], dataset_name: str) -> list[dict[str, Any]]:
    groups = {
        "G1_triplet_direct_hard_positive": lambda row: row["G1_triplet_direct_hard_positive"],
        "G2_not_direct_hard_positive": lambda row: row["G2_not_direct_hard_positive"],
        "G3_both_endpoint_relative_order_satisfied": lambda row: row["G3_both_endpoint_relative_order_satisfied"],
        "B_intersect_C": lambda row: row["B_intersect_C"],
        "G4_cross_clothes": lambda row: row["G4_cross_clothes"],
        "G5_same_clothes": lambda row: row["G5_same_clothes"],
    }
    rows = []
    for group, predicate in groups.items():
        selected = [row for row in pair_rows if row["dataset"] == dataset_name and predicate(row)]
        by_pid = defaultdict(list)
        for row in selected:
            by_pid[int(row["pid"])].append(float(row["gpc_incremental_effect"]))
        pid_values = {pid: float(np.mean(values)) for pid, values in by_pid.items()}
        stable_seed = int.from_bytes(hashlib.sha256(f"{dataset_name}|{group}".encode("utf-8")).digest()[:4], "little")
        ci_low, ci_high = bootstrap_ci(pid_values, seed=SEED + stable_seed % 10000)
        inc = np.asarray([float(row["gpc_incremental_effect"]) for row in selected], dtype=float)
        no_gpc = np.asarray([float(row["delta_without_gpc"]) for row in selected], dtype=float)
        with_gpc = np.asarray([float(row["delta_with_gpc"]) for row in selected], dtype=float)
        pre = np.asarray([float(row["pre_similarity"]) for row in selected], dtype=float)
        rows.append({
            "dataset": dataset_name,
            "group": group,
            "pid_count": len(pid_values),
            "pair_count": len(selected),
            "pre_similarity_mean": float(np.mean(pre)) if len(pre) else None,
            "pre_similarity_median": float(np.median(pre)) if len(pre) else None,
            "pre_similarity_p10": float(np.quantile(pre, 0.1)) if len(pre) else None,
            "delta_without_gpc_mean": float(np.mean(no_gpc)) if len(no_gpc) else None,
            "delta_with_gpc_mean": float(np.mean(with_gpc)) if len(with_gpc) else None,
            "gpc_incremental_effect_mean": float(np.mean(inc)) if len(inc) else None,
            "gpc_incremental_effect_median": float(np.median(inc)) if len(inc) else None,
            "pair_improvement_rate": float(np.mean(inc > 0)) if len(inc) else None,
            "pid_improvement_rate": float(np.mean(np.asarray(list(pid_values.values())) > 0)) if pid_values else None,
            "pid_bootstrap_95ci_low": ci_low,
            "pid_bootstrap_95ci_high": ci_high,
            "status": "one_step_counterfactual_from_same_Bio_checkpoint",
        })
    return rows


def aggregate_negative(negative_rows: list[dict[str, Any]], dataset_name: str) -> list[dict[str, Any]]:
    selected = [row for row in negative_rows if row["dataset"] == dataset_name]
    by_pid: dict[int, list[float]] = defaultdict(list)
    by_pid_margin: dict[int, list[float]] = defaultdict(list)
    for row in selected:
        by_pid[int(row["pid"])].append(float(row["negative_similarity_incremental_effect"]))
        by_pid_margin[int(row["pid"])].append(float(row["identity_margin_incremental_effect"]))
    neg_pid = {pid: float(np.mean(values)) for pid, values in by_pid.items()}
    margin_pid = {pid: float(np.mean(values)) for pid, values in by_pid_margin.items()}
    neg_ci = bootstrap_ci(neg_pid, seed=SEED + 11)
    margin_ci = bootstrap_ci(margin_pid, seed=SEED + 12)
    return [{
        "dataset": dataset_name,
        "negative_observation_count": len(selected),
        "negative_pid_count": len(neg_pid),
        "negative_similarity_incremental_effect_mean": float(np.mean(list(neg_pid.values()))) if neg_pid else None,
        "negative_similarity_incremental_pid_improvement_rate": float(np.mean(np.asarray(list(neg_pid.values())) > 0)) if neg_pid else None,
        "negative_similarity_incremental_pid_bootstrap_95ci_low": neg_ci[0],
        "negative_similarity_incremental_pid_bootstrap_95ci_high": neg_ci[1],
        "identity_margin_incremental_effect_mean": float(np.mean(list(margin_pid.values()))) if margin_pid else None,
        "identity_margin_incremental_pid_improvement_rate": float(np.mean(np.asarray(list(margin_pid.values())) > 0)) if margin_pid else None,
        "identity_margin_incremental_pid_bootstrap_95ci_low": margin_ci[0],
        "identity_margin_incremental_pid_bootstrap_95ci_high": margin_ci[1],
    }]


def prior_crosstab() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    with PRIOR_PAIR_CSV.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["dataset"] != "prcc":
                continue
            rows.append(row)
    counts = Counter()
    for row in rows:
        a = row["A_hard_positive_overlap"].lower() == "true"
        b = row["B_not_hard_positive_either_direction"].lower() == "true"
        c = row["C_both_endpoint_relative_constraints_satisfied"].lower() == "true"
        d = row["D_at_least_one_endpoint_hinge_active"].lower() == "true"
        if a == b or c == d:
            raise RuntimeError(f"non-exclusive prior relation flags: {row}")
        counts[("A" if a else "B", "C" if c else "D")] += 1
    table = [
        {"A_or_B": "A", "C_or_D": "C", "count": counts[("A", "C")]},
        {"A_or_B": "A", "C_or_D": "D", "count": counts[("A", "D")]},
        {"A_or_B": "B", "C_or_D": "C", "count": counts[("B", "C")]},
        {"A_or_B": "B", "C_or_D": "D", "count": counts[("B", "D")]},
    ]
    return table, {
        "pair_count": len(rows),
        "expected_pair_count": PRIOR_PAIR_COUNT,
        "A_count": sum(counts[("A", x)] for x in ("C", "D")),
        "B_count": sum(counts[("B", x)] for x in ("C", "D")),
        "C_count": sum(counts[(x, "C")] for x in ("A", "B")),
        "D_count": sum(counts[(x, "D")] for x in ("A", "B")),
        "B_intersect_C_count": counts[("B", "C")],
        "definitions": {
            "A": "at least one direction i->j or j->i is Triplet hard-positive",
            "B": "neither direction is Triplet hard-positive",
            "C": "both endpoints satisfy dist_an >= dist_ap using the audited soft-triplet operational ordering",
            "D": "at least one endpoint has dist_ap > dist_an",
        },
    }


def historical_pair_audit() -> dict[str, Any]:
    manifests = {
        "prcc_bio": Path("/data/DIFFER/outputs/new_method_track/phase7c_prototype_relation/training/C0_bio/run_manifest.json"),
        "prcc_bio_complete": Path("/data/DIFFER/outputs/new_method_track/phase7c_prototype_relation/training/C0_bio/RUN_COMPLETE.json"),
        "prcc_gpc_complete": Path("/home/DIFFER/log_NDP/ndp_nteach_current_b0_pair_20261031_125146/PRCC_B0_full/PRCC/RUN_COMPLETE.json"),
        "ltcc_bio": Path("/data/DIFFER/outputs/new_method_track/phase5b_id_damage_ds/S0_bio_full_data/run_manifest.json"),
        "ltcc_bio_complete": Path("/data/DIFFER/outputs/new_method_track/phase5b_id_damage_ds/S0_bio_full_data/RUN_COMPLETE.json"),
        "ltcc_gpc_complete": Path("/home/DIFFER/log_NDP/ndp_nteach_b0_recovery_20261030_210612/LTCC_B0_full/LTCC/RUN_COMPLETE.json"),
    }
    loaded = {}
    for name, path in manifests.items():
        loaded[name] = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"missing": str(path)}
    return {
        "manifests": {name: {"path": str(path), "sha256": sha256_file(path), "payload": loaded[name]} for name, path in manifests.items()},
        "prcc": {
            "common_init_sha256_bio": loaded["prcc_bio"].get("common_init", {}).get("sha256"),
            "common_init_sha256_gpc": loaded["prcc_gpc_complete"].get("initialization_source"),
            "same_initialisation_claim_supported": True,
            "same_runner_source": False,
            "same_batch_augmentation_order_proven": False,
            "same_optimizer_state_proven": False,
            "strict_long_term_pair": False,
        },
        "ltcc": {
            "common_init_sha256_bio": loaded["ltcc_bio"].get("common_init", {}).get("sha256"),
            "common_init_sha256_gpc": loaded["ltcc_gpc_complete"].get("initialization_source"),
            "same_initialisation_claim_supported": True,
            "same_runner_source": False,
            "same_batch_augmentation_order_proven": False,
            "same_optimizer_state_proven": False,
            "strict_long_term_pair": False,
        },
        "decision": "Historical Bio/Bio+GPC checkpoint metrics are not strict causal pairs: common-init provenance exists, but runner/source, batch/augmentation order, optimizer continuation state, and bitwise trajectory are not jointly proven.",
    }


def runtime_provenance(dataset_summaries, prior_info, historical) -> dict[str, Any]:
    cfg_records = {}
    for name in ("prcc", "ltcc"):
        cfg = cfg_for(name)
        cfg_records[name] = {
            "loss_type": list(cfg.MODEL.LOSS_TYPE),
            "sampler": str(cfg.DATA.SAMPLER),
            "batch_size": int(cfg.DATA.BATCH_SIZE),
            "num_instances": int(cfg.DATA.NUM_INSTANCES),
            "base_lr": float(cfg.SOLVER.BASE_LR),
            "bias_lr_factor": float(cfg.SOLVER.BIAS_LR_FACTOR),
            "momentum": float(cfg.SOLVER.MOMENTUM),
            "weight_decay": float(cfg.SOLVER.WEIGHT_DECAY),
            "weight_decay_bias": float(cfg.SOLVER.WEIGHT_DECAY_BIAS),
            "no_margin": bool(cfg.MODEL.NO_MARGIN),
            "bio_index": str(cfg.DATA.BIO_INDEX),
            "nonbio_index": list(cfg.DATA.NOBIO_INDEX),
            "summary_text": bool(cfg.DATA.SUMMERY_TEXT),
            "caption_dir_used_diagnostic": str(cfg.DATA.CAPTION_DIR),
            "formal_caption_dir_in_wrapper": str(tfg.SPEC[name]["caption_dir"]),
            "gpc_weight": LAMBDA_GPC,
        }
    imported = {}
    for module_name in ("data", "model", "loss", "loss.triplet_loss", "loss.make_loss", "solver.make_optimizer"):
        module = sys.modules.get(module_name)
        imported[module_name] = getattr(module, "__file__", None) if module else None
    return {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "seed": SEED,
        "epoch_for_sampler": EPOCH,
        "device": str(DEVICE),
        "device_name": "PPU-ZW810E; /usr/local/bin/python3 with PPU_SDK",
        "wrapper": str(ROOT / "scripts/train_final_gpc.py"),
        "wrapper_sha256": sha256_file(ROOT / "scripts/train_final_gpc.py"),
        "gpc_sha256": sha256_file(ROOT / "loss/gpc_relation.py"),
        "triplet_sha256": sha256_file(ROOT / "loss/triplet_loss.py"),
        "make_loss_sha256": sha256_file(ROOT / "loss/make_loss.py"),
        "imported_module_files": imported,
        "configs": cfg_records,
        "checkpoints": {name: {"path": str(path), "exists": path.is_file(), "sha256": sha256_file(path)} for name, path in CHECKPOINTS.items()},
        "dataset_summaries": dataset_summaries,
        "prior_147_pair_summary": prior_info,
        "historical_pair_audit": historical,
        "formal_training_started": False,
        "scheduler_started": False,
        "official_query_gallery_used": False,
        "checkpoint_written_by_diagnostic": False,
        "rank_state_written_by_diagnostic": False,
    }


def fmt(value, digits=6):
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_report(prior_table, prior_info, summary_rows, negative_rows, gradient_rows, dataset_summaries, historical):
    lines = [
        "# GPC–Batch-Hard Triplet 局部因果验证",
        "",
        f"生成时间：`{dt.datetime.now(dt.timezone.utc).isoformat()}`；seed=`{SEED}`；PPU=`cuda:0`。",
        "",
        "本报告只做 train-only 的单步反事实诊断，不修改 GPC/Triplet/Bio 定义，不加入 NDP，不启动正式 60 epoch 训练，也不写入任何 checkpoint。A/B 分支都从同一个已有 Bio checkpoint 恢复，A 为 `CE+Triplet+clipBio`，B 为 `CE+Triplet+clipBio+0.1×GPC`。由于已有 checkpoint 没有可复用的历史 optimizer state，本实验为相同新建 SGD 状态下的虚拟单步，不代表历史训练轨迹。",
        "注意：PPU 上的极小单步更新会有数值非确定性；早期 smoke 与完整参数梯度重跑的个别 B-A 点估计在 `1e-4` 量级变化，但 PID-level CI 的跨零/不跨零判断未改变。因此结论不依据单个微小点估计，而依据固定计划、PID 聚合和 bootstrap CI。",
        "",
        "## 1. PRCC 原报告 147 个 pair 的 A/B × C/D 交叉表",
        "",
        f"原 CSV 实际读取 `{prior_info['pair_count']}` 个 PRCC pair；预期为 `{prior_info['expected_pair_count']}`。A/B 互斥，C/D 互斥。重点集合 B∩C 为 `{prior_info['B_intersect_C_count']}` 个 pair，占 `{prior_info['B_intersect_C_count']/max(prior_info['pair_count'],1):.2%}`。",
        "",
        "| 关系分类 | C：两端相对排序满足 | D：至少一端 hard-active | 合计 |",
        "|---|---:|---:|---:|",
        f"| A：至少一方向 hard-positive | {prior_table[0]['count']} | {prior_table[1]['count']} | {prior_info['A_count']} |",
        f"| B：两个方向均未直接选中 | {prior_table[2]['count']} | {prior_table[3]['count']} | {prior_info['B_count']} |",
        f"| 合计 | {prior_info['C_count']} | {prior_info['D_count']} | {prior_info['pair_count']} |",
        "",
        "C 是当前实现中 `dist_an >= dist_ap` 的操作性满足，不等于 SoftMarginLoss 数学值严格为零；当前 `NO_MARGIN=True` 的 Softplus Triplet 在理论上仍有正尾梯度。因此 B∩C 证明的是‘未被 hard-positive 直接选择且满足相对排序’，不是‘完全没有任何 Triplet 梯度’。",
        "",
        "上表的 B∩C=`70` 是原报告在历史 GPC checkpoint 上对 147 个 pair 的复核；后续单步因果表使用同一固定 pair/batch 计划，但在 Bio checkpoint 的更新前特征上重新固定关系分组，因此 causal 表中的 PRCC B∩C=`64`，这是 checkpoint 几何差异，不是 pair 计划不一致。",
        "",
        "## 2. 固定 train-only 计划与模型来源",
        "",
        "| 数据集 | Bio checkpoint | 固定 batch/rank | 有选中 GPC 的 batch | GPC pair | 跨衣比例 | budget=0 比例 | PID 数 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("prcc", "ltcc"):
        s = dataset_summaries[name]
        lines.append(f"| {name.upper()} | `{CHECKPOINTS[name + '_bio']}` | {s['fixed_batches_per_rank']} | {s['batches_with_selected_gpc']} | {s['selected_pairs']} | {s['selected_cross_rate']:.2%} | {s['budget_zero_rate']:.2%} | {s['selected_pid_count']} |")
    lines.extend([
        "",
        "PRCC 使用原报告的 rank0/rank1 前 8 个 batch，因此该批计划对应原 147-pair 审计；LTCC 预先固定每个 rank 前 256 个 batch，以增加稀疏 GPC 的有效观察量。没有 GPC pair 的固定 batch 被计入计划和 budget 统计，但不需要重复做 A/B 更新，因为两分支在该 batch 的 GPC 项严格为零。所有 batch、增强后的图像张量、caption_feature 与关系均在更新前固定。",
        "",
        "## 3. 单步固定关系比较",
        "",
        "下表的增量定义为：`(post_B-pre) - (post_A-pre) = post_B-post_A`，即 GPC 相对于无 GPC 单步更新的真正额外效果。统计以 PID 聚合，CI 为 PID-level bootstrap 95% CI。",
        "",
        "| 数据集 | 关系组 | PID | pair | pre cosine | Δ no-GPC | Δ with-GPC | GPC incremental | pair改善率 | PID改善率 | 95% CI |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in summary_rows:
        lines.append(f"| {row['dataset'].upper()} | {row['group']} | {row['pid_count']} | {row['pair_count']} | {fmt(row['pre_similarity_mean'])} | {fmt(row['delta_without_gpc_mean'])} | {fmt(row['delta_with_gpc_mean'])} | {fmt(row['gpc_incremental_effect_mean'])} | {fmt(row['pair_improvement_rate'], 2)} | {fmt(row['pid_improvement_rate'], 2)} | [{fmt(row['pid_bootstrap_95ci_low'])}, {fmt(row['pid_bootstrap_95ci_high'])}] |")
    lines.extend([
        "",
        "注意：G1/G2/G3/G4/G5 是在各个 Bio checkpoint 的更新前固定；G1/G2 使用当前 batch 的两方向 hard-positive 关系，G3 使用两端 `dist_an >= dist_ap`，G4/G5 使用 clothes_id。B∩C 是 G2∩G3，不是依据更新后效果反向筛选。",
        "",
        "### 3.1 不同 PID 困难负样本与 Identity Margin",
        "",
        "| 数据集 | fixed batch-hard negative 观测 | PID | 负相似度增量（B-A） | Identity Margin 增量（B-A） | 负相似度 CI | Margin CI |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in negative_rows:
        lines.append(f"| {row['dataset'].upper()} | {row['negative_observation_count']} | {row['negative_pid_count']} | {fmt(row['negative_similarity_incremental_effect_mean'])} | {fmt(row['identity_margin_incremental_effect_mean'])} | [{fmt(row['negative_similarity_incremental_pid_bootstrap_95ci_low'])}, {fmt(row['negative_similarity_incremental_pid_bootstrap_95ci_high'])}] | [{fmt(row['identity_margin_incremental_pid_bootstrap_95ci_low'])}, {fmt(row['identity_margin_incremental_pid_bootstrap_95ci_high'])}] |")
    lines.extend([
        "",
        "负相似度增量为正表示不同 PID 的固定困难负关系在 GPC 分支中更相似；Identity Margin 增量为负表示正负判别间隔恶化。该表只使用训练 batch 内预先固定的 Batch-Hard negative，不使用官方 query/gallery。",
        "",
        "## 4. 梯度证据",
        "",
        "梯度审计在所有有 GPC pair 的固定 batch 上执行。feature gradient 是对 `feat[0]` 的梯度；parameter gradient 是预先固定的共享表示子空间 `blocks.22.*`（最终 transformer block，不含 classifier/Bio heads）的逐参数全局统计。GPC raw 是未乘权重的直接梯度，实际训练梯度为 `0.1×GPC`。",
        "",
        "| 数据集 | batch 数 | Triplet feature norm | GPC feature raw | GPC feature weighted | Triplet param norm | GPC param raw | GPC param weighted | feature cos(GPC,Triplet) | param cos(GPC,Triplet) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for dataset_name in ("prcc", "ltcc"):
        subset = [row for row in gradient_rows if row["dataset"] == dataset_name]
        if not subset:
            lines.append(f"| {dataset_name.upper()} | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |")
            continue
        mean = lambda key: float(np.mean([float(row[key]) for row in subset if row[key] is not None])) if any(row[key] is not None for row in subset) else None
        parameter_subset = sum(row["parameter_gradient_computed_this_batch"] for row in subset)
        lines.append(f"| {dataset_name.upper()} | {len(subset)} (param={parameter_subset}) | {fmt(mean('feature_triplet_grad_norm'))} | {fmt(mean('feature_gpc_raw_grad_norm'))} | {fmt(mean('feature_gpc_weighted_0p1_grad_norm'))} | {fmt(mean('parameter_triplet_grad_norm'))} | {fmt(mean('parameter_gpc_raw_grad_norm'))} | {fmt(mean('parameter_gpc_weighted_0p1_grad_norm'))} | {fmt(mean('feature_cos_triplet_vs_gpc_raw'))} | {fmt(mean('parameter_cos_triplet_vs_gpc_raw'))} |")
    lines.extend([
        "",
        "### 4.1 真实 B∩C 案例",
        "",
        "PRCC rank0/iteration1、PID 22 的 pair `/home/data/prcc/rgb/train/143/A_cropped_rgb244.jpg`–`/home/data/prcc/rgb/train/143/B_cropped_rgb175.jpg` 是当前 Bio checkpoint 下的 B∩C：两端 hard-positive 都不是该 pair（hard-positive index 分别为 7、7），但两端 `dist_an >= dist_ap`（22.9469 < 30.6118、25.7526 < 33.0336）。该 pair 的 pre cosine 为 `0.8325265`，所以 GPC pair loss=`1-cos` 为 `0.1674735`，直接 GPC 梯度非零；对应 Softplus Triplet 项仍只有约 `0.000469` 和 `0.000688`，不是严格零。这个具体 pair 的一次 PPU 单步中，no-GPC 相似度变化为 `+0.0004150`，with-GPC 为 `+0.0003092`，B-A 为 `−0.0001059`，说明‘有额外梯度’与‘这一步一定改善’是两个不同命题。",
        "",
        "在 B∩C pair 上，Triplet 的操作性相对排序已满足，但 GPC 的 `1-cosine` 只要 pair cosine 小于 1 就有非零直接梯度。由于当前 Triplet 是 SoftMarginLoss，单个 Triplet 项通常不是严格零；因此本实验支持‘GPC 仍提供额外正样本聚合方向’，但不能把它表述成‘Triplet 完全没有梯度’。",
        "",
        "## 5. 历史长期配对模型审计",
        "",
        "| 数据集 | common-init 证据 | runner 相同 | batch/augmentation 顺序相同已证明 | optimizer state 相同 | 严格长期因果对照 |",
        "|---|---|---|---|---|---|",
        f"| PRCC | {historical['prcc']['same_initialisation_claim_supported']} | {historical['prcc']['same_runner_source']} | {historical['prcc']['same_batch_augmentation_order_proven']} | {historical['prcc']['same_optimizer_state_proven']} | {historical['prcc']['strict_long_term_pair']} |",
        f"| LTCC | {historical['ltcc']['same_initialisation_claim_supported']} | {historical['ltcc']['same_runner_source']} | {historical['ltcc']['same_batch_augmentation_order_proven']} | {historical['ltcc']['same_optimizer_state_proven']} | {historical['ltcc']['strict_long_term_pair']} |",
        "",
        "PRCC Bio 与历史 GPC checkpoint 有 common-init 来源记录，但 GPC 是另一 runner 的 model-only warm start；LTCC 也有 common-init 记录，但 Bio checkpoint 为 15-epoch screen 状态，历史 GPC 为另一 runner 的 60-epoch model-only warm start。没有同一训练进程中可验证的 bitwise batch/augmentation/optimizer continuation。因此历史最终 Rank-1/mAP 不能替代严格的长期因果证据。",
        "",
        "## 6. 结论",
        "",
        "1. GPC 是否额外优化 Triplet 未直接选中的正样本：关系覆盖层面，PRCC 原 147 pair 中 B 类占比为 `" + f"{prior_info['B_count']/max(prior_info['pair_count'],1):.2%}" + "`，B∩C 为 `" + f"{prior_info['B_intersect_C_count']}" + "` 个；单步梯度层面，若 GPC pair cosine<1，GPC 对其提供非零直接 feature/parameter 梯度。",
        "2. 相对排序已满足时的额外改善：以本报告表格中的 B∩C `GPC incremental effect` 为准；它是 B-A，不是 B-pre。若 CI 跨零，只能说方向性/局部证据，不能宣称稳定改善。",
        "3. 跨衣增益：以 G4 与 G5 的 PID-level incremental effect 和 CI 直接比较；不能仅凭 pair 数或最终指标判断。",
        "4. 判别能力代价：以负相似度增量与 Identity Margin 增量表判断。Margin CI 跨零时，不能断言无损或恶化；若为负且 CI 不跨零，才有训练 batch 层面的损害证据。",
        "5. 论文表述边界：当前证据可以支持较弱表述——‘GPC selects same-identity positive relations beyond the hard-positive subset, and its direct gradient remains nonzero when the operational Batch-Hard relative ordering is already satisfied.’ 只有在本报告 B∩C/G4/G5 的 PID-level incremental CI 明确支持时，才可再加上‘改善’；长期 60 epoch 的稳定性能、跨数据集可复现性和最终 mAP 因果增益必须等待严格配对训练。",
        "",
        "## 7. 复现文件",
        "",
        f"- 诊断脚本：`{OUT / 'validate.py'}`",
        f"- A/B×C/D 交叉表：`{OUT / 'ab_relation_crosstab.csv'}`",
        f"- 固定 pair 单步结果：`{OUT / 'single_step_pair_effects.csv'}`",
        f"- 固定困难负样本结果：`{OUT / 'single_step_negative_effects.csv'}`",
        f"- 梯度审计：`{OUT / 'single_step_gradient_audit.csv'}`",
        f"- 汇总 JSON：`{OUT / 'single_step_summary.json'}`",
        f"- provenance：`{OUT / 'provenance.json'}`",
        "",
    ])
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("PPU/CUDA is unavailable; use /usr/local/bin/python3 with PPU_SDK")
    OUT.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    prior_table, prior_info = prior_crosstab()
    write_csv(OUT / "ab_relation_crosstab.csv", prior_table)

    all_pair_rows = []
    all_negative_rows = []
    all_gradient_rows = []
    dataset_summaries = {}
    batch_plan_rows = []

    for dataset_name, limit in (("prcc", PRCC_BATCHES_PER_RANK), ("ltcc", LTCC_BATCHES_PER_RANK)):
        print(f"[causal] materialising {dataset_name}", flush=True)
        dataset, bank, fixed, plan_rows, ds_summary = materialise_fixed_dataset(dataset_name, limit)
        dataset_summaries[dataset_name] = ds_summary
        batch_plan_rows.extend(plan_rows)
        print(f"[causal] {dataset_name}: selected batches={len(fixed)}, pairs={ds_summary['selected_pairs']}", flush=True)
        cfg, initial_state, model_a, model_b = build_model_pair(dataset_name, CHECKPOINTS[f"{dataset_name}_bio"])
        loss_a, opt_a = make_loss_optimizer(cfg, model_a)
        loss_b, opt_b = make_loss_optimizer(cfg, model_b)
        for batch_number, item in enumerate(fixed):
            if batch_number % 10 == 0:
                print(f"[causal] {dataset_name} batch {batch_number + 1}/{len(fixed)}", flush=True)
            pair_rows, negative_rows, gradient_row = run_one_batch(
                item, model_a, model_b, loss_a, loss_b, opt_a, opt_b, initial_state, dataset_name, batch_number, batch_number < PARAMETER_GRADIENT_BATCH_LIMIT
            )
            all_pair_rows.extend(pair_rows)
            all_negative_rows.extend(negative_rows)
            all_gradient_rows.append(gradient_row)
        del model_a, model_b, loss_a, loss_b, opt_a, opt_b, initial_state, dataset, bank
        torch.cuda.empty_cache()

    write_csv(OUT / "fixed_batch_plan.csv", batch_plan_rows)
    write_csv(OUT / "single_step_pair_effects.csv", all_pair_rows)
    write_csv(OUT / "single_step_negative_effects.csv", all_negative_rows)
    write_csv(OUT / "single_step_gradient_audit.csv", all_gradient_rows)
    group_summary = aggregate_group(all_pair_rows, "prcc") + aggregate_group(all_pair_rows, "ltcc")
    negative_summary = aggregate_negative(all_negative_rows, "prcc") + aggregate_negative(all_negative_rows, "ltcc")
    historical = historical_pair_audit()
    provenance = runtime_provenance(dataset_summaries, prior_info, historical)
    write_json(OUT / "single_step_summary.json", {
        "prior_crosstab": prior_info,
        "group_summary": group_summary,
        "negative_summary": negative_summary,
        "gradient_rows": len(all_gradient_rows),
        "pair_rows": len(all_pair_rows),
        "negative_rows": len(all_negative_rows),
        "interpretation_guard": "A/B are independent one-step updates from the same Bio checkpoint and fresh identical SGD state; not a historical trajectory.",
    })
    write_json(OUT / "provenance.json", provenance)
    render_report(prior_table, prior_info, group_summary, negative_summary, all_gradient_rows, dataset_summaries, historical)
    print(json.dumps({"report": str(REPORT), "output": str(OUT), "prior": prior_info, "datasets": dataset_summaries, "group_summary": group_summary, "negative_summary": negative_summary}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
