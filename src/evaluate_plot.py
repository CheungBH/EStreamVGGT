import os
import re
import json
import argparse
import numpy as np
import random
import torch
from types import SimpleNamespace
from accelerate import Accelerator
from accelerate.utils import set_seed
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import finetune as ft
from finetune import VGGT, build_dataset, test_one_epoch
from vggt.lora import apply_lora_to_aggregator


def list_checkpoints(folder):
    names = os.listdir(folder)
    pat = re.compile(r"^checkpoint-(\d+)\.pth$")
    items = []
    for n in names:
        m = pat.match(n)
        if m:
            items.append((int(m.group(1)), os.path.join(folder, n)))
    items.sort(key=lambda x: x[0])
    return items


def plot_per_view(output_dir, modality, num_views, prefix):
    mpath = os.path.join(output_dir, "metric_view.txt")
    if not os.path.exists(mpath):
        return
    data = []
    with open(mpath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            data.append(obj)
    if not data:
        return
    base = os.path.join(output_dir, "metrics_views")
    os.makedirs(base, exist_ok=True)
    metrics = [
        "auc30",
        "acc_mean",
        "acc_med",
        "comp_mean",
        "comp_med",
        "nc_mean",
        "nc_med",
        "depth_absrel",
        "depth_delta_125",
    ]
    def view_type(v, modality):
        if modality == "rgb":
            return "RGB"
        if modality == "event":
            return "event"
        if modality == "rgb_first_event":
            return "RGB" if v == 0 else "event"
        if modality == "rgb_event_loop":
            return "RGB" if (v % 2 == 0) else "event"
        return "RGB"
    for m in metrics:
        series = {}
        for obj in data:
            ep = obj.get("epoch")
            vals = obj.get(prefix, {})
            for v in range(1, num_views + 1):
                key = f"{m}_v{v}"
                if key in vals:
                    series.setdefault(v, []).append((ep, float(vals[key])))
        if not series:
            continue
        plt.figure(figsize=(8, 4))
        for v, pts in sorted(series.items()):
            pts = sorted(pts, key=lambda x: x[0])
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            lbl = f"v{v} ({view_type(v - 1, modality)})"
            plt.plot(xs, ys, marker="o", linewidth=2, label=lbl)
        plt.xlabel("epoch")
        plt.ylabel(m)
        plt.title(f"{prefix} - {m}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        safe = f"{prefix.replace(' ', '_').replace('/', '_')}__{m}.png"
        plt.tight_layout()
        plt.savefig(os.path.join(base, safe))
        plt.close()


from omegaconf import OmegaConf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()
    cfg = OmegaConf.load(args.config)
    folder = os.path.abspath(str(getattr(cfg, "output_dir")))
    # set seeds and deterministic behavior if requested
    seed = int(getattr(cfg, "seed", 0))
    set_seed(seed, device_specific=False)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if bool(getattr(cfg, "deterministic", False)):
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    ckpts = list_checkpoints(folder)
    run_items = []
    pretrained_path = getattr(cfg, "pretrained", None)
    if pretrained_path:
        pp = os.path.abspath(str(pretrained_path))
        if os.path.exists(pp):
            run_items.append((0, pp))
    run_items.extend(ckpts)
    if not run_items:
        raise RuntimeError("no checkpoints or pretrained weights found")
    accelerator = Accelerator()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_eval = os.path.join(folder, "evaluate")
    os.makedirs(out_eval, exist_ok=True)
    data_loader_test = build_dataset(
        str(getattr(cfg, "test_dataset")),
        int(getattr(cfg, "batch_size", 1)),
        int(getattr(cfg, "num_workers", 2)),
        accelerator=accelerator,
        test=True,
        fixed_length=True,
    )
    data_loader_test.dataset.set_epoch(0)
    model = VGGT()
    model.to(device)
    simple_args = SimpleNamespace(
        output_dir=out_eval,
        print_freq=int(getattr(cfg, "print_freq", 10)),
        amp=bool(getattr(cfg, "amp", False)),
        num_test_views=int(getattr(cfg, "num_test_views")),
        modality=str(getattr(cfg, "modality")),
    )
    crit_expr = str(getattr(cfg, "test_criterion"))
    if "(" not in crit_expr:
        crit_expr = f"{crit_expr}()"
    criterion = eval(crit_expr, ft.__dict__).to(device)
    used = set()
    total = len(run_items)
    print(f"[evaluate] total items: {total}")
    for idx, (eidx, path) in enumerate(tqdm(run_items, total=total, desc="Checkpoints", dynamic_ncols=True), start=1):
        t0 = time.time()
        print(f"[{idx}/{total}] evaluating: {os.path.basename(path)} (epoch_hint={eidx})")
        if eidx in used:
            continue
        used.add(eidx)
        ckpt = torch.load(path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict):
            if "model" in ckpt and isinstance(ckpt["model"], dict):
                sd = ckpt["model"]
            elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
                sd = ckpt["state_dict"]
            else:
                sd = ckpt
        else:
            sd = ckpt
        if isinstance(sd, dict) and len(sd) > 0 and next(iter(sd)).startswith("module."):
            sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
        if isinstance(sd, dict) and len(sd) > 0 and next(iter(sd)).startswith("model."):
            sd = {k.replace("model.", "", 1): v for k, v in sd.items()}
        # inject LoRA if checkpoint contains LoRA weights
        has_lora = any(".lora_A.weight" in k for k in sd.keys())
        if has_lora:
            # infer r from the first lora_A weight shape; set alpha=r for scaling=1
            for k, v in sd.items():
                if k.endswith(".lora_A.weight") and hasattr(v, "shape"):
                    r = int(v.shape[0])
                    alpha = r
                    apply_lora_to_aggregator(model.aggregator, r=r, alpha=alpha, target="all")
                    model.to(device)
                    break
        try:
            model.load_state_dict(sd, strict=True)
        except RuntimeError:
            if has_lora:
                model.load_state_dict(sd, strict=False)
            else:
                raise
        print(f"[{idx}/{total}] loaded weights, start test_one_epoch")
        t1 = time.time()
        stats = test_one_epoch(
            model,
            None,
            criterion,
            data_loader_test,
            accelerator,
            device,
            int(eidx),
            simple_args,
            log_writer=None,
            prefix=str(getattr(cfg, "prefix", "eval")),
        )
        # consolidate averaged metrics and write both metric.json (one metric per line) and a tabular metric.txt
        prefix = str(getattr(cfg, "prefix", "eval"))
        consolidated = {}
        def take(key):
            if key in (stats or {}):
                consolidated[key] = float(stats[key])
        # loss
        take("loss"); take("pose_loss")
        # depth
        for k in ("depth_absrel","depth_rmse","depth_log_rmse","depth_si_rmse","depth_delta_125","depth_delta_1252","depth_delta_1253"):
            take(k)
        # pose
        for k in ("pose_rot_deg","pose_trans_err","pose_auc30"):
            take(k)
        # track
        for k in ("conf_mean","track_conf_mean","track_vis_ratio"):
            take(k)
        # pts3d avg across views
        def avg_prefix(pfx):
            vals = [v for k, v in (stats or {}).items() if k.startswith(pfx + "/")]
            if vals:
                consolidated[pfx] = float(np.mean(vals))
        avg_prefix("Regr3DPose_pts3d")
        avg_prefix("Regr3DPose_ScaleInv_pts3d")
        # write metric.json (newline json, one metric per line)
        jpath = os.path.join(out_eval, "metric.json")
        with open(jpath, "a", encoding="utf-8") as jf:
            for name, val in consolidated.items():
                jf.write(json.dumps({"epoch": int(eidx), "name": name, "value": val}) + "\n")
        # write metric.txt as space-separated table (header + rows)
        tpath = os.path.join(out_eval, "metric.txt")
        order = []
        order += [k for k in ("loss","pose_loss") if k in consolidated]
        order += [k for k in ("depth_absrel","depth_rmse","depth_log_rmse","depth_si_rmse","depth_delta_125","depth_delta_1252","depth_delta_1253") if k in consolidated]
        order += [k for k in ("pose_rot_deg","pose_trans_err","pose_auc30") if k in consolidated]
        order += [k for k in ("Regr3DPose_pts3d","Regr3DPose_ScaleInv_pts3d") if k in consolidated]
        order += [k for k in ("conf_mean","track_conf_mean","track_vis_ratio") if k in consolidated]
        if not os.path.exists(tpath):
            with open(tpath, "w", encoding="utf-8") as tf:
                tf.write("epoch " + " ".join(order) + "\n")
        with open(tpath, "a", encoding="utf-8") as tf:
            vals = [str(int(eidx))] + [f"{consolidated[k]:.6f}" for k in order]
            tf.write(" ".join(vals) + "\n")
        dt = time.time() - t0
        print(f"[{idx}/{total}] done {os.path.basename(path)} in {dt:.1f}s (load {t1 - t0:.1f}s, eval {dt - (t1 - t0):.1f}s)")
    plot_per_view(out_eval, str(getattr(cfg, "modality")), int(getattr(cfg, "num_test_views")), str(getattr(cfg, "prefix", "eval")))
    ft.plot_category_dashboards(out_eval)


if __name__ == "__main__":
    main()
