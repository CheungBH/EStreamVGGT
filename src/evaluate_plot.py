import os
import re
import argparse
import numpy as np
import random
import time
from types import SimpleNamespace

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from tqdm import tqdm

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()
    cfg = OmegaConf.load(args.config)
    folder = os.path.abspath(str(getattr(cfg, "output_dir")))
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
    prefix = str(getattr(cfg, "prefix", "eval"))
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
        has_lora = any(".lora_A.weight" in k for k in sd.keys())
        if has_lora:
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
        test_one_epoch(
            model,
            None,
            criterion,
            data_loader_test,
            accelerator,
            device,
            int(eidx),
            simple_args,
            log_writer=None,
            prefix=prefix,
        )
        dt = time.time() - t0
        print(f"[{idx}/{total}] done {os.path.basename(path)} in {dt:.1f}s (load {t1 - t0:.1f}s, eval {dt - (t1 - t0):.1f}s)")
    ft.plot_view_metrics(out_eval, str(getattr(cfg, "modality")), int(getattr(cfg, "num_test_views")))
    ft.plot_category_dashboards(out_eval)


if __name__ == "__main__":
    main()
