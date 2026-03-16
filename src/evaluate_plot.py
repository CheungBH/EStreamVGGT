import os
import re
import json
import argparse
import numpy as np
import torch
from types import SimpleNamespace
from accelerate import Accelerator
import matplotlib.pyplot as plt
import finetune as ft
from finetune import VGGT, build_dataset, test_one_epoch


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
    base = os.path.join(output_dir, "evaluate", "metrics_views")
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
    for obj in data:
        ep = obj.get("epoch")
        vals = obj.get(prefix, {})
        for m in metrics:
            series = {}
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
    ckpts = list_checkpoints(folder)
    if not ckpts:
        raise RuntimeError("no checkpoints found")
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
    ep = 1
    for eidx, path in ckpts:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        sd = ckpt["model"] if isinstance(ckpt, dict) else ckpt
        if isinstance(sd, dict) and len(sd) > 0 and next(iter(sd)).startswith("module."):
            sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=True)
        test_one_epoch(
            model,
            None,
            criterion,
            data_loader_test,
            accelerator,
            device,
            ep,
            simple_args,
            log_writer=None,
            prefix=str(getattr(cfg, "prefix", "eval")),
        )
        ep += 1
    plot_per_view(out_eval, str(getattr(cfg, "modality")), int(getattr(cfg, "num_test_views")), str(getattr(cfg, "prefix", "eval")))


if __name__ == "__main__":
    main()
