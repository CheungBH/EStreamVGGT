# --------------------------------------------------------
# training code for CUT3R
# --------------------------------------------------------
# References:
# DUSt3R: https://github.com/naver/dust3r
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import sys
import time
import math
from collections import defaultdict
from pathlib import Path
from typing import Sized
from itertools import islice

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12

from dust3r.model import (
    PreTrainedModel,
    ARCroco3DStereo,
    ARCroco3DStereoConfig,
    inf,
    strip_module,
)  # noqa: F401, needed when loading the model
from dust3r.datasets import get_data_loader
from dust3r.losses import *  # noqa: F401, needed when loading the model
from dust3r.inference import loss_of_one_batch  # noqa
from dust3r.viz import colorize
from dust3r.utils.render import get_render_results
import dust3r.utils.path_to_croco  # noqa: F401
import croco.utils.misc as misc  # noqa
from croco.utils.misc import NativeScalerWithGradNormCount as NativeScaler  # noqa

import hydra
from omegaconf import OmegaConf
import logging
import pathlib
from tqdm import tqdm
import random
import builtins
import shutil
import imageio.v2 as iio
import matplotlib.pyplot as plt
import plot

from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs, InitProcessGroupKwargs
from accelerate.logging import get_logger
from datetime import timedelta
import torch.multiprocessing

from vggt.models.vggt import VGGT
from vggt.lora import apply_lora_to_aggregator, mark_only_lora_trainable

torch.multiprocessing.set_sharing_strategy("file_system")

printer = get_logger(__name__, log_level="DEBUG")


def setup_for_distributed(accelerator: Accelerator):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        force = force or (accelerator.num_processes > 8)
        if accelerator.is_main_process or force:
            now = datetime.datetime.now().time()
            builtin_print("[{}] ".format(now), end="")  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def save_current_code(outdir):
    now = datetime.datetime.now()  # current date and time
    date_time = now.strftime("%m_%d_%H_%M_%S")
    src_dir = "."
    dst_dir = os.path.join(outdir, "code", "{}".format(date_time))
    shutil.copytree(
        src_dir,
        dst_dir,
        ignore=shutil.ignore_patterns(
            ".vscode*",
            "assets*",
            "example*",
            "checkpoints*",
            "OLD*",
            "logs*",
            "out*",
            "runs*",
            "*.png",
            "*.mp4",
            "*__pycache__*",
            "*.git*",
            "*.idea*",
            "*.zip",
            "*.jpg",
        ),
        dirs_exist_ok=True,
    )
    return dst_dir


def train(args):

    # dynamic mixed precision for GPU arch (Ampere+ uses bf16, otherwise fp16)
    try:
        cap = torch.cuda.get_device_capability(0)[0] if torch.cuda.is_available() else 0
    except Exception:
        cap = 0
    mp = "bf16" if cap >= 8 else "fp16"
    # allow backend override via config or env (default nccl)
    import os
    dist_backend = getattr(args, "dist_backend", None) or os.environ.get("DIST_BACKEND", "nccl")
    use_dp = getattr(args, "use_dp", False) or os.environ.get("USE_DP", "").lower() in ("1", "true", "yes")
    use_fsdp = getattr(args, "use_fsdp", False) or os.environ.get("USE_FSDP", "").lower() in ("1", "true", "yes")
    if use_dp:
        accelerator = DPAccelerator()
    else:
        accelerator = Accelerator(
            gradient_accumulation_steps=args.accum_iter,
            mixed_precision=mp,
            kwargs_handlers=[
                DistributedDataParallelKwargs(find_unused_parameters=True),
                InitProcessGroupKwargs(timeout=timedelta(seconds=6000), backend=dist_backend),
            ],
        )
    device = accelerator.device

    setup_for_distributed(accelerator)

    printer.info("output_dir: " + args.output_dir)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # plot-only fast path: regenerate figures from existing metric files without training/eval
    if getattr(args, "plot_only", False):
        try:
            plot.plot_view_metrics(args.output_dir, getattr(args, "modality", "rgb"), getattr(args, "num_test_views", 0))
            plot.plot_category_dashboards(args.output_dir)
            printer.info("Regenerated plots from existing metrics under: %s", args.output_dir)
        except Exception as e:
            printer.error("Plot-only path failed: %s", str(e))
        return

    if accelerator.is_main_process:
        dst_dir = save_current_code(outdir=args.output_dir)
        printer.info(f"Saving current code to {dst_dir}")

    # metrics preflight: validate dataset fields without running full eval
    if getattr(args, "eval_check", False):
        def _shape(t):
            try:
                return list(t.shape)
            except Exception:
                return None
        for test_name, testset in data_loader_test.items():
            try:
                batch = next(iter(testset))
            except Exception:
                printer.error(f"[{test_name}] cannot fetch a batch")
                continue
            ok = True
            for vi, view in enumerate(batch):
                cp = view.get("camera_pose", None)
                K = view.get("camera_intrinsics", None)
                gd = view.get("depthmap", None)
                if not (isinstance(cp, torch.Tensor) and (_shape(cp) and (_shape(cp)[-2:] == [3,4] or _shape(cp)[-1] == 7))):
                    printer.error(f"[{test_name}] view{vi+1} missing/invalid camera_pose shape={_shape(cp)}")
                    ok = False
                if not (isinstance(K, torch.Tensor) and (_shape(K) and _shape(K)[-2:] == [3,3])):
                    printer.error(f"[{test_name}] view{vi+1} missing/invalid camera_intrinsics shape={_shape(K)}")
                    ok = False
                if not (isinstance(gd, torch.Tensor) and (_shape(gd) and len(_shape(gd)) in (3,4))):
                    printer.error(f"[{test_name}] view{vi+1} missing/invalid depthmap shape={_shape(gd)}")
                    ok = False
            if ok:
                printer.info(f"[{test_name}] metrics prerequisites present for all views")
        return
    # auto resume
    if not args.resume:
        last_ckpt_fname = os.path.join(args.output_dir, f"checkpoint-last.pth")
        args.resume = last_ckpt_fname if os.path.isfile(last_ckpt_fname) else None

    printer.info("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))

    # fix the seed
    seed = args.seed + accelerator.state.process_index
    printer.info(
        f"Setting seed to {seed} for process {accelerator.state.process_index}"
    )
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = args.benchmark

    # training dataset and loader
    printer.info("Building train dataset %s", args.train_dataset)
    #  dataset and loader
    data_loader_train = build_dataset(
        args.train_dataset,
        args.batch_size,
        args.num_workers,
        accelerator=accelerator,
        test=False,
        fixed_length=args.fixed_length
    )
    printer.info("Building test dataset %s", args.test_dataset)
    data_loader_test = {
        dataset.split("(")[0]: build_dataset(
            dataset,
            args.batch_size,
            args.num_workers,
            accelerator=accelerator,
            test=True,
            fixed_length=True
        )
        for dataset in args.test_dataset.split("+")
    }

    # model
    printer.info("Loading model")
    model = VGGT()

    # LoRA injection on aggregator trunk if requested
    lora_cfg = getattr(args, "lora", None)
    lora_enable = False
    lora_r = 8
    lora_alpha = 8
    lora_update_base = False
    try:
        import os as _os
        lora_enable = bool(getattr(lora_cfg, "enable", False)) or _os.environ.get("LORA_ENABLE", "").lower() in ("1", "true", "yes")
        if getattr(lora_cfg, "r", None) is not None:
            lora_r = int(lora_cfg.r)
        elif _os.environ.get("LORA_R"):
            lora_r = int(_os.environ.get("LORA_R"))
        if getattr(lora_cfg, "alpha", None) is not None:
            lora_alpha = int(lora_cfg.alpha)
        elif _os.environ.get("LORA_ALPHA"):
            lora_alpha = int(_os.environ.get("LORA_ALPHA"))
        lora_update_base = bool(getattr(lora_cfg, "update_base", False)) or _os.environ.get("LORA_UPDATE_BASE", "").lower() in ("1","true","yes")
        lora_target = getattr(lora_cfg, "targets", None) or _os.environ.get("LORA_TARGETS", "") or "all"
    except Exception:
        lora_target = "all"
    # defer LoRA injection until after checkpoint loading

    # model: PreTrainedModel = eval(args.model)
    printer.info(f"All model parameters: {sum(p.numel() for p in model.parameters())}")


    printer.info(f">> Creating train criterion = {args.train_criterion}")
    train_criterion = eval(args.train_criterion).to(device)
    printer.info(
        f">> Creating test criterion = {args.test_criterion or args.train_criterion}"
    )
    test_criterion = eval(args.test_criterion or args.criterion).to(device)

    model.to(device)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    if args.long_context:
        model.fixed_input_length = False

    if args.pretrained and not args.resume:
        printer.info(f"Loading pretrained: {args.pretrained}")
        ckpt = torch.load(args.pretrained, map_location=device)
        printer.info(
            model.load_state_dict(ckpt, strict=True)
        )
        del ckpt  # in case it occupies memory

    # freeze
    printer.info("Freezing patch embedding and positional encoding parameters...")
    frozen_params = 0
    total_params = 0

    frozen_param_names = []

    for name, param in model.named_parameters():
        total_params += param.numel()
        param.requires_grad = True

    if hasattr(model, 'aggregator') and hasattr(model.aggregator, 'patch_embed'):
        for param in model.aggregator.patch_embed.parameters():
            if param.requires_grad:
                param.requires_grad = False

    if hasattr(model, 'aggregator') and hasattr(model.aggregator, 'camera_token'):
        model.aggregator.camera_token.requires_grad = False

    if hasattr(model, 'aggregator') and hasattr(model.aggregator, 'register_token'):
        model.aggregator.register_token.requires_grad = False

    model.camera_head.requires_grad = False
    model.depth_head.requires_grad = False
    model.track_head.requires_grad = False

    


    for name, p in model.named_parameters():
        if not p.requires_grad:
            frozen_params += p.numel()
            frozen_param_names.append(name)

    printer.info(
        f"Frozen {frozen_params:,} parameters out of {total_params:,} total parameters. ({frozen_params / total_params:.2%})")
    printer.info(
        f"Trainable parameters: {total_params - frozen_params:,} ({(total_params - frozen_params) / total_params:.2%})")
    if frozen_param_names:
        printer.info(
            f"Example frozen parameters: {', '.join(frozen_param_names[:5])}{'...' if len(frozen_param_names) > 5 else ''}")

    # now inject LoRA after loading checkpoints and applying freezes
    if lora_enable:
        printer.info(f"Applying LoRA on aggregator trunk (r={lora_r}, alpha={lora_alpha}, target={lora_target})")
        apply_lora_to_aggregator(model.aggregator, r=lora_r, alpha=lora_alpha, target=lora_target)
        if not lora_update_base:
            mark_only_lora_trainable(model.aggregator)
            printer.info("LoRA-only trainable on aggregator; base weights frozen")



    # following timm: set wd as 0 for bias and norm layers
    param_groups = misc.get_parameter_groups(model, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    # print(optimizer)
    loss_scaler = NativeScaler(accelerator=accelerator)

    best_so_far = misc.load_model(
        args=args, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler
    )
    if best_so_far is None:
        best_so_far = float("inf")

    accelerator.even_batches = False
    optimizer, model, data_loader_train = accelerator.prepare(
        optimizer, model, data_loader_train
    )

    # eval-only fast path: load checkpoint and run tests to regenerate metrics/files
    if getattr(args, "eval_only", False) or getattr(args, "eval_sweep", False):
        def load_ckpt_into_model(ckpt_path):
            unwrapped = accelerator.unwrap_model(model)
            obj = torch.load(ckpt_path, map_location=device)
            sd = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
            unwrapped.load_state_dict(sd, strict=False)
            ep = obj["epoch"] if isinstance(obj, dict) and "epoch" in obj else args.start_epoch
            return ep
        ckpt_list = []
        if getattr(args, "eval_sweep", False):
            base = getattr(args, "ckpt_dir", None) or args.output_dir
            pats = []
            for name in os.listdir(base):
                if name.startswith("checkpoint-") and name.endswith(".pth"):
                    try:
                        ep = int(name.split("-")[1].split(".")[0])
                    except Exception:
                        ep = None
                    ckpt_list.append((ep, os.path.join(base, name)))
            def key_fn(t):
                e, path = t
                if e is not None:
                    return e
                try:
                    obj = torch.load(path, map_location="cpu")
                    return obj.get("epoch", -1) if isinstance(obj, dict) else -1
                except Exception:
                    return -1
            ckpt_list = sorted(list({p for p in ckpt_list}), key=key_fn)
        else:
            if getattr(args, "resume", None):
                ckpt_list = [(None, args.resume)]
            else:
                ckpt_list = []
        for ep_hint, ckpt_path in ckpt_list:
            try:
                ep = load_ckpt_into_model(ckpt_path)
            except Exception:
                continue
            test_stats = {}
            for test_name, testset in data_loader_test.items():
                stats = test_one_epoch(
                    model,
                    None,
                    test_criterion,
                    testset,
                    accelerator,
                    device,
                    ep_hint if ep_hint is not None else ep,
                    args=args,
                    log_writer=None,
                    prefix=test_name,
                )
                test_stats[test_name] = stats
            if accelerator.is_main_process:
                log_stats = dict(epoch=ep_hint if ep_hint is not None else ep, **{f"train_{k}": v for k, v in {}.items()})
                for test_name in data_loader_test:
                    if test_name in test_stats:
                        log_stats.update({test_name + "_" + k: v for k, v in test_stats[test_name].items()})
                with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")
                metrics_line = {"epoch": ep_hint if ep_hint is not None else ep}
                for test_name in data_loader_test:
                    if test_name in test_stats:
                        metrics_line[test_name] = test_stats[test_name]
                with open(os.path.join(args.output_dir, "metric.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(metrics_line) + "\n")
        if accelerator.is_main_process:
            plot.plot_view_metrics(args.output_dir, getattr(args, "modality", "rgb"), getattr(args, "num_test_views", 0))
            plot.plot_category_dashboards(args.output_dir)
        return

    def write_log_stats(epoch, train_stats, test_stats):
        if accelerator.is_main_process:
            if log_writer is not None:
                log_writer.flush()

            log_stats = dict(
                epoch=epoch, **{f"train_{k}": v for k, v in train_stats.items()}
            )
            for test_name in data_loader_test:
                if test_name not in test_stats:
                    continue
                log_stats.update(
                    {test_name + "_" + k: v for k, v in test_stats[test_name].items()}
                )

            with open(
                os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(log_stats) + "\n")
            metrics_line = {"epoch": epoch}
            for test_name in data_loader_test:
                if test_name in test_stats:
                    metrics_line[test_name] = test_stats[test_name]
            with open(
                os.path.join(args.output_dir, "metric.txt"), mode="a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(metrics_line) + "\n")

    def save_model(epoch, fname, best_so_far, data_iter_step):
        misc.save_model(
            accelerator=accelerator,
            args=args,
            model_without_ddp=model,
            optimizer=optimizer,
            loss_scaler=loss_scaler,
            epoch=epoch,
            step=data_iter_step,
            fname=fname,
            best_so_far=best_so_far,
        )

    log_writer = (
        SummaryWriter(log_dir=args.output_dir) if accelerator.is_main_process else None
    )

    printer.info(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    train_stats = test_stats = {}

    for epoch in range(args.start_epoch, args.epochs + 1):

        # Save immediately the last checkpoint
        if epoch > args.start_epoch:
            if (
                args.save_freq
                and np.allclose(epoch / args.save_freq, int(epoch / args.save_freq))
                or epoch == args.epochs
            ):
                save_model(epoch - 1, "last", best_so_far, args.start_step)

        new_best = False
        eval_every = 1
        if eval_every is None:
            eval_every = getattr(args, "eval_freq", 0)
        if eval_every > 0 and epoch % eval_every == 0:
            test_stats = {}
            for test_name, testset in data_loader_test.items():
                stats = test_one_epoch(
                    model,
                    None,
                    test_criterion,
                    testset,
                    accelerator,
                    device,
                    epoch,
                    args=args,
                    log_writer=log_writer,
                    prefix=test_name,
                )
                test_stats[test_name] = stats
                if "loss_med" in stats and stats["loss_med"] < best_so_far:
                    best_so_far = stats["loss_med"]
                    new_best = True
            write_log_stats(epoch, train_stats, test_stats)
        else:
            write_log_stats(epoch, train_stats, {})

        if epoch > args.start_epoch:
            if args.keep_freq and epoch % args.keep_freq == 0:
                save_model(epoch - 1, str(epoch), best_so_far, args.start_step)
            if new_best:
                save_model(epoch - 1, "best", best_so_far, args.start_step)
        if epoch >= args.epochs:
            break  # exit after writing last test to disk


        # Train
        train_stats = train_one_epoch(
            model,
            train_criterion,
            data_loader_train,
            optimizer,
            accelerator,
            epoch,
            loss_scaler,
            log_writer=log_writer,
            args=args
        )


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    printer.info("Training time {}".format(total_time_str))

    save_final_model(accelerator, args, args.epochs, model, best_so_far=best_so_far)
    if accelerator.is_main_process:
        plot.plot_view_metrics(args.output_dir, getattr(args, "modality", "rgb"), getattr(args, "num_test_views", 0))
        plot.plot_category_dashboards(args.output_dir)


def save_final_model(accelerator, args, epoch, model_without_ddp, best_so_far=None):
    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / "checkpoint-final.pth"
    to_save = {
        "args": args,
        "model": (
            model_without_ddp
            if isinstance(model_without_ddp, dict)
            else model_without_ddp.cpu().state_dict()
        ),
        "epoch": epoch,
    }
    if best_so_far is not None:
        to_save["best_so_far"] = best_so_far
    printer.info(f">> Saving model to {checkpoint_path} ...")
    misc.save_on_master(accelerator, to_save, checkpoint_path)

def plot_all_metrics(output_dir):
    import json
    import os
    import numpy as np
    mpath = os.path.join(output_dir, "metric.txt")
    if not os.path.exists(mpath):
        return
    series = {}
    with open(mpath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            epoch = obj.get("epoch")
            for dataset, metrics in obj.items():
                if dataset == "epoch":
                    continue
                if not isinstance(metrics, dict):
                    continue
                for k, v in metrics.items():
                    key = f"{dataset}/{k}"
                    series.setdefault(key, []).append((epoch, v))
    outdir = os.path.join(output_dir, "visualize", "metrics")
    os.makedirs(outdir, exist_ok=True)
    for key, pts in series.items():
        pts = sorted([(e, v) for e, v in pts if isinstance(e, (int, float)) and isinstance(v, (int, float, np.number))], key=lambda x: x[0])
        if len(pts) < 1:
            continue
        xs = [p[0] for p in pts]
        ys = [float(p[1]) for p in pts]
        plt.figure(figsize=(8, 4))
        plt.plot(xs, ys, marker="o", linewidth=2)
        plt.xlabel("epoch")
        plt.ylabel(key.split("/", 1)[1])
        plt.title(key)
        plt.grid(True, alpha=0.3)
        safe_key = key.replace("/", "__")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{safe_key}.png"))
        plt.close()

def plot_view_metrics(output_dir, modality, num_views):
    import json
    import os
    import numpy as np
    mpath = os.path.join(output_dir, "metric_view.txt")
    if not os.path.exists(mpath):
        return
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
    wanted = ["auc30", "acc_mean", "acc_med", "comp_mean", "comp_med", "nc_mean", "nc_med", "depth_absrel", "depth_delta_125"]
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
    prefixes = []
    for obj in data:
        for k in obj.keys():
            if k != "epoch":
                prefixes.append(k)
    prefixes = sorted(list(set(prefixes)))
    base = os.path.join(output_dir, "visualize", "metrics_views")
    os.makedirs(base, exist_ok=True)
    for prefix in prefixes:
        series = {}
        for obj in data:
            ep = obj.get("epoch")
            vals = obj.get(prefix, {})
            for m in wanted:
                for v in range(1, (num_views or 0) + 1):
                    key = f"{m}_v{v}"
                    if key in vals and isinstance(ep, (int, float)):
                        series.setdefault(m, {}).setdefault(v, []).append((ep, float(vals[key])))
        for m, by_view in series.items():
            if not by_view:
                continue
            plt.figure(figsize=(8, 4))
            for v, pts in sorted(by_view.items()):
                pts = sorted(pts, key=lambda x: x[0])
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                typ = view_type(v - 1, modality)
                lbl = f"v{v} ({typ})"
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
        # paired plots: mean & med in同一图（左右子图）
        paired = [("acc_mean", "acc_med", "acc"), ("comp_mean", "comp_med", "comp"), ("nc_mean", "nc_med", "nc")]
        for m_mean, m_med, name in paired:
            if m_mean in series and m_med in series:
                fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
                for ax, mkey, title_suffix in zip(axes, [m_mean, m_med], ["Mean", "Med"]):
                    by_view = series[mkey]
                    for v, pts in sorted(by_view.items()):
                        pts = sorted(pts, key=lambda x: x[0])
                        xs = [p[0] for p in pts]
                        ys = [p[1] for p in pts]
                        typ = view_type(v - 1, modality)
                        lbl = f"v{v} ({typ})"
                        ax.plot(xs, ys, marker="o", linewidth=2, label=lbl)
                    ax.set_xlabel("epoch")
                    ax.set_ylabel(mkey)
                    ax.set_title(f"{title_suffix}")
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                fig.suptitle(f"{prefix} - {name} (Mean/Med)")
                fig.tight_layout()
                safe = f"{prefix.replace(' ', '_').replace('/', '_')}__{name}_pair.png"
                fig.savefig(os.path.join(base, safe))
                plt.close(fig)

def plot_category_dashboards(output_dir):
    import json
    import os
    import numpy as np
    mpath = os.path.join(output_dir, "metric.txt")
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
    prefixes = []
    for obj in data:
        for k in obj.keys():
            if k != "epoch":
                prefixes.append(k)
    prefixes = sorted(list(set(prefixes)))
    cat_map = {
        "depth_error": [
            "depth_absrel_avg", "depth_absrel_med",
            "depth_delta_125_avg", "depth_delta_125_med",
            "depth_rmse_avg", "depth_rmse_med",
            "depth_log_rmse_avg", "depth_log_rmse_med",
            "depth_si_rmse_avg", "depth_si_rmse_med",
        ],
        "pose": [
            "pose_rot_deg", "pose_trans_err", "pose_auc30",
        ],
        "geometry": [
            "pts3d_acc_mean", "pts3d_acc_med",
            "pts3d_comp_mean", "pts3d_comp_med",
            "pts3d_nc_mean", "pts3d_nc_med",
            "pts3d_chamfer_l1", "pts3d_chamfer_l2",
        ],
        "confidence_visibility": [
            "conf_mean", "track_conf_mean", "track_vis_ratio",
        ],
        "loss": [
            "loss_avg", "loss_med", "pose_loss_avg", "pose_loss_med",
        ],
    }
    outdir = os.path.join(output_dir, "visualize", "metrics_dashboards")
    os.makedirs(outdir, exist_ok=True)
    def collect_series(prefix, keys):
        series = {}
        for obj in data:
            ep = obj.get("epoch")
            vals = obj.get(prefix, {})
            for k in keys:
                if k in vals and isinstance(ep, (int, float)):
                    series.setdefault(k, []).append((ep, float(vals[k])))
        for k in list(series.keys()):
            series[k] = sorted(series[k], key=lambda x: x[0])
        return series
    def plot_cat(prefix, cname, keys):
        series = collect_series(prefix, keys)
        if not series:
            return
        n = len(series)
        cols = 2 if n > 1 else 1
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 3.2 * rows), squeeze=False)
        axes = axes.flatten()
        for ax, (k, pts) in zip(axes, series.items()):
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, marker="o", linewidth=2)
            ax.set_title(k)
            ax.set_xlabel("epoch")
            ax.set_ylabel(k)
            ax.grid(True, alpha=0.3)
        # hide extra axes
        for j in range(len(series), len(axes)):
            axes[j].axis("off")
        fig.suptitle(f"{prefix} - {cname}")
        fig.tight_layout()
        safe = f"{prefix.replace(' ', '_').replace('/', '_')}__{cname}.png"
        fig.savefig(os.path.join(outdir, safe))
        plt.close(fig)
    for prefix in prefixes:
        for cname, keys in cat_map.items():
            plot_cat(prefix, cname, keys)

def build_dataset(dataset, batch_size, num_workers, accelerator, test=False, fixed_length=False):
    split = ["Train", "Test"][test]
    printer.info(f"Building {split} Data loader for dataset: {dataset}")
    loader = get_data_loader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_mem=True,
        shuffle=not (test),
        drop_last=not (test),
        accelerator=accelerator,
        fixed_length=fixed_length
    )
    return loader


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Sized,
    optimizer: torch.optim.Optimizer,
    accelerator: Accelerator,
    epoch: int,
    loss_scaler,
    args,
    log_writer=None,
):
    assert torch.backends.cuda.matmul.allow_tf32 == True

    model.train(True)
    device = accelerator.device
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    accum_iter = args.accum_iter

    def save_model(epoch, fname, best_so_far, data_iter_step):
        unwrapped_model = accelerator.unwrap_model(model)
        misc.save_model(
            accelerator=accelerator,
            args=args,
            model_without_ddp=unwrapped_model,
            optimizer=optimizer,
            loss_scaler=loss_scaler,
            epoch=epoch,
            step=data_iter_step,
            fname=fname,
            best_so_far=best_so_far,
        )

    if log_writer is not None:
        printer.info("log_dir: {}".format(log_writer.log_dir))

    if hasattr(data_loader, "dataset") and hasattr(data_loader.dataset, "set_epoch"):
        data_loader.dataset.set_epoch(epoch)
    if (
        hasattr(data_loader, "batch_sampler")
        and hasattr(data_loader.batch_sampler, "batch_sampler")
        and hasattr(data_loader.batch_sampler.batch_sampler, "set_epoch")
    ):
        data_loader.batch_sampler.batch_sampler.set_epoch(epoch)


    optimizer.zero_grad()

    start_step = args.start_step

    data_iter = metric_logger.log_every(data_loader, args.print_freq, accelerator, header)

    for data_iter_step, batch in enumerate(data_iter):
            
        with accelerator.accumulate(model):
            # change the range of the image to [0, 1]
            if isinstance(batch, dict) and "img" in batch:
                batch["img"] = (batch["img"] + 1.0) / 2.0
                model_dtype = next(model.parameters()).dtype
                batch["img"] = batch["img"].to(device=device, dtype=model_dtype)
            elif isinstance(batch, list) and all(isinstance(v, dict) and "img" in v for v in batch):
                for view in batch:
                    view["img"] = (view["img"] + 1.0) / 2.0
                    model_dtype = next(model.parameters()).dtype
                    view["img"] = view["img"].to(device=device, dtype=model_dtype)
                    if "camera_pose" in view:
                        x = view["camera_pose"]
                        if isinstance(x, np.ndarray):
                            x = torch.from_numpy(x)
                        view["camera_pose"] = x.to(device=device, dtype=model_dtype)
                    if "camera_intrinsics" in view:
                        x = view["camera_intrinsics"]
                        if isinstance(x, np.ndarray):
                            x = torch.from_numpy(x)
                        view["camera_intrinsics"] = x.to(device=device, dtype=model_dtype)
                    if "depthmap" in view:
                        x = view["depthmap"]
                        if isinstance(x, np.ndarray):
                            x = torch.from_numpy(x)
                        view["depthmap"] = x.to(device=device, dtype=model_dtype)
                    if "pts3d" in view:
                        x = view["pts3d"]
                        if isinstance(x, np.ndarray):
                            x = torch.from_numpy(x)
                        view["pts3d"] = x.to(device=device, dtype=model_dtype)
                    for k in ("valid_mask", "sky_mask", "img_mask", "ray_mask"):
                        if k in view:
                            x = view[k]
                            if isinstance(x, np.ndarray):
                                x = torch.from_numpy(x)
                            if isinstance(x, torch.Tensor):
                                view[k] = x.to(device=device, dtype=torch.bool)

            epoch_f = epoch + data_iter_step / len(data_loader)
            # we use a per iteration (instead of per epoch) lr scheduler
            if data_iter_step % accum_iter == 0:
                misc.adjust_learning_rate(optimizer, epoch_f, args)

            epoch_f = epoch + data_iter_step / len(data_loader)
            step = int(epoch_f * len(data_loader))

            result = loss_of_one_batch(
                batch,
                model,
                criterion,
                accelerator,
                inference=False,
                symmetrize_batch=False,
                use_amp=bool(args.amp),
            )
      
            loss, loss_details = result["loss"]  # criterion returns two values

            loss_value = float(loss)

            if not math.isfinite(loss_value):
                print(
                    f"Loss is {loss_value}, stopping training, loss details: {loss_details}"
                )
                sys.exit(1)
            if not result.get("already_backprop", False):
                loss_scaler(
                    loss,
                    optimizer,
                    parameters=model.parameters(),
                    update_grad=True,
                    clip_grad=1.0,
                )
                optimizer.zero_grad()

            is_metric = batch[0]["is_metric"]
            curr_num_view = len(batch)

            del loss
            tb_vis_img = (data_iter_step + 1) % accum_iter == 0 and (
                (step + 1) % (args.print_img_freq)
            ) == 0
            if not tb_vis_img:
                del batch
            else:
                torch.cuda.empty_cache()

            lr = optimizer.param_groups[0]["lr"]
            metric_logger.update(epoch=epoch_f)
            metric_logger.update(lr=lr)
            metric_logger.update(step=step)
            #
            metric_logger.update(loss=loss_value, **loss_details)
            #
            if (data_iter_step + 1) % accum_iter == 0 and (
                (data_iter_step + 1) % (accum_iter * args.print_freq)
            ) == 0:
                loss_value_reduce = accelerator.gather(
                    torch.tensor(loss_value).to(accelerator.device)
                ).mean()  # MUST BE EXECUTED BY ALL NODES

                if log_writer is None:
                    continue
                """ We use epoch_1000x as the x-axis in tensorboard.
                This calibrates different curves when batch size changes.
                """
                epoch_1000x = int(epoch_f * 1000)
                log_writer.add_scalar("train_loss", loss_value_reduce, step)
                log_writer.add_scalar("train_lr", lr, step)
                log_writer.add_scalar("train_iter", epoch_1000x, step)
                for name, val in loss_details.items():
                    if isinstance(val, torch.Tensor):
                        if val.ndim > 0:
                            continue
                    if isinstance(val, dict):
                        continue
                    log_writer.add_scalar("train_" + name, val, step)

            if tb_vis_img:
                if log_writer is None:
                    continue
                with torch.no_grad():
                    depths_cross, gt_depths_cross = get_render_results(
                        batch, result["pred"], self_view=False
                    )
                    for k in range(len(batch)):

                        loss_details[f"pred_depth_{k+1}"] = (
                            depths_cross[k].detach().cpu()
                        )
                        loss_details[f"gt_depth_{k+1}"] = (
                            gt_depths_cross[k].detach().cpu()
                        )

                # imgs_stacked_dict = get_vis_imgs_new(
                #     loss_details, args.num_imgs_vis, curr_num_view, is_metric=is_metric
                # )
                # save_vis_imgs(args.output_dir, "train", epoch, imgs_stacked_dict, step=data_iter_step)
                del batch
            del loss_details
            del result
            if (data_iter_step + 1) % (accum_iter * args.print_freq) == 0:
                try:
                    mem_alloc = torch.cuda.memory_allocated(device) / (1024**2)
                    mem_rsrv = torch.cuda.memory_reserved(device) / (1024**2)
                    printer.info(f"[mem] alloc={mem_alloc:.1f}MB reserved={mem_rsrv:.1f}MB")
                except Exception:
                    pass
                torch.cuda.empty_cache()

        if (
            data_iter_step % int(args.save_freq * len(data_loader)) == 0
            and data_iter_step != 0
            and data_iter_step != len(data_loader) - 1
        ):
            print("saving at step", data_iter_step)
            save_model(epoch - 1, "last", float("inf"), data_iter_step)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes(accelerator)
    printer.info("Averaged stats: %s", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def test_one_epoch(
    model: torch.nn.Module,
    teacher: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Sized,
    accelerator: Accelerator,
    device: torch.device,
    epoch: int,
    args,
    log_writer=None,
    prefix="test",
):

    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.meters = defaultdict(lambda: misc.SmoothedValue(window_size=9**9))
    header = "Test Epoch: [{}]".format(epoch)

    if log_writer is not None:
        printer.info("log_dir: {}".format(log_writer.log_dir))

    if hasattr(data_loader, "dataset") and hasattr(data_loader.dataset, "set_epoch"):
        data_loader.dataset.set_epoch(0)
        try:
            from dust3r.datasets.base.easy_dataset import ResizedDataset
            if isinstance(data_loader.dataset, ResizedDataset):
                import numpy as np
                rng = np.random.default_rng(42)
                base_len = len(data_loader.dataset.dataset)
                new_size = len(data_loader.dataset)
                perm = rng.permutation(base_len)
                shuffled_idxs = np.concatenate([perm] * (1 + (new_size - 1) // base_len))
                data_loader.dataset._idxs_mapping = shuffled_idxs[: new_size]
        except Exception:
            pass
    if (
        hasattr(data_loader, "batch_sampler")
        and hasattr(data_loader.batch_sampler, "batch_sampler")
        and hasattr(data_loader.batch_sampler.batch_sampler, "set_epoch")
    ):
        data_loader.batch_sampler.batch_sampler.set_epoch(0)

    # per-view aggregation across the whole epoch
    per_view_metrics = {
        "depth_absrel": {},
        "depth_delta_125": {},
        "auc30": {},
        "acc_mean": {},
        "acc_med": {},
        "comp_mean": {},
        "comp_med": {},
        "nc_mean": {},
        "nc_med": {},
    }
    def _agg(metric_name, vi, val):
        if metric_name not in per_view_metrics:
            return
        if val is None:
            return
        per_view_metrics[metric_name].setdefault(vi, []).append(float(val))

    for _, batch in enumerate(
        metric_logger.log_every(data_loader, args.print_freq, accelerator, header)
    ):
        if isinstance(batch, dict) and "img" in batch:
            model_dtype = next(model.parameters()).dtype
            batch["img"] = batch["img"].to(device=device, dtype=model_dtype)
        elif isinstance(batch, list) and all(isinstance(v, dict) and "img" in v for v in batch):
            model_dtype = next(model.parameters()).dtype
            for view in batch:
                view["img"] = view["img"].to(device=device, dtype=model_dtype)
                if "camera_pose" in view:
                    x = view["camera_pose"]
                    if isinstance(x, np.ndarray):
                        x = torch.from_numpy(x)
                    view["camera_pose"] = x.to(device=device, dtype=model_dtype)
                if "camera_intrinsics" in view:
                    x = view["camera_intrinsics"]
                    if isinstance(x, np.ndarray):
                        x = torch.from_numpy(x)
                    view["camera_intrinsics"] = x.to(device=device, dtype=model_dtype)
                if "depthmap" in view:
                    x = view["depthmap"]
                    if isinstance(x, np.ndarray):
                        x = torch.from_numpy(x)
                    view["depthmap"] = x.to(device=device, dtype=model_dtype)
                if "pts3d" in view:
                    x = view["pts3d"]
                    if isinstance(x, np.ndarray):
                        x = torch.from_numpy(x)
                    view["pts3d"] = x.to(device=device, dtype=model_dtype)
                for k in ("valid_mask", "sky_mask", "img_mask", "ray_mask"):
                    if k in view:
                        x = view[k]
                        if isinstance(x, np.ndarray):
                            x = torch.from_numpy(x)
                        if isinstance(x, torch.Tensor):
                            view[k] = x.to(device=device, dtype=torch.bool)
        result = loss_of_one_batch(
            batch,
            model,
            criterion,
            accelerator,
            teacher=teacher,
            symmetrize_batch=False,
            use_amp=bool(args.amp),
        )

        loss_value, loss_details = result["loss"]  # criterion returns two values
        metric_logger.update(loss=float(loss_value), **loss_details)

        if isinstance(batch, list):
            preds = result.get("pred", [])
            depth_absrels = []
            depth_rmses = []
            depth_log_rmses = []
            depth_si_rmses = []
            depth_delta_125s = []
            depth_delta_1252s = []
            depth_delta_1253s = []
            pose_rot_degs = []
            pose_trans_errs = []
            pts3d_chamfer_l1s = []
            pts3d_chamfer_l2s = []
            conf_means = []
            track_conf_means = []
            track_vis_ratios = []
            pose_auc30s = []
            acc_means = []
            acc_meds = []
            comp_means = []
            comp_meds = []
            nc_means = []
            nc_meds = []
            for vi, view in enumerate(batch):
                pred_vi = None
                if isinstance(preds, list) and len(preds) > vi:
                    pred_vi = preds[vi]
                elif isinstance(preds, dict):
                    pred_vi = preds
                gt_depth = view.get("depthmap", None)
                mask = view.get("ray_mask", None)
                if mask is None:
                    mask = view.get("valid_mask", None)
                if pred_vi is not None:
                    pr_depth = pred_vi.get("depth", None)
                else:
                    pr_depth = None
                if gt_depth is not None and isinstance(gt_depth, torch.Tensor) and pr_depth is not None and isinstance(pr_depth, torch.Tensor):
                    pd = pr_depth
                    g = gt_depth
                    # squeeze trailing channel if present
                    if pd.ndim == 4 and pd.shape[-1] == 1:
                        pd = pd.squeeze(-1)
                    if pd.ndim == 4 and pd.shape[1] == 1:
                        pd = pd.squeeze(1)
                    if g.ndim == 4 and g.shape[-1] == 1:
                        g = g.squeeze(-1)
                    if g.ndim == 4 and g.shape[1] == 1:
                        g = g.squeeze(1)
                    # resize prediction to GT resolution if needed
                    if pd.shape[-2:] != g.shape[-2:]:
                        _pd = pd
                        if _pd.ndim == 3:
                            _pd = _pd.unsqueeze(1)
                        _pd = torch.nn.functional.interpolate(_pd.float(), size=g.shape[-2:], mode="bilinear", align_corners=True)
                        pd = _pd.squeeze(1)
                    # build mask with correct shape
                    if isinstance(mask, torch.Tensor) and mask.shape[-2:] == g.shape[-2:]:
                        m = mask.bool()
                        if m.ndim == 2:
                            m = m.unsqueeze(0).expand_as(g)
                        elif m.ndim == 3 and m.shape != g.shape and m.shape[0] == 1 and g.shape[0] > 1:
                            m = m.expand_as(g)
                    else:
                        m = torch.ones_like(g, dtype=torch.bool)
                    depth_min = torch.tensor(1e-3, device=g.device, dtype=g.dtype)
                    valid = m & (g > depth_min) & (pd > depth_min)
                    if valid.any():
                        rel = ((pd - g).abs() / g.clamp_min(depth_min)).masked_select(valid).mean().item()
                        rmse = torch.sqrt(((pd - g).square()).masked_select(valid).mean()).item()
                        # log RMSE
                        pd_safe = pd.clamp_min(depth_min)
                        g_safe = g.clamp_min(depth_min)
                        log_diff = (pd_safe.log() - g_safe.log())
                        log_rmse = torch.sqrt((log_diff.square()).masked_select(valid).mean()).item()
                        # scale-invariant RMSE (remove mean of log diff)
                        mu = log_diff.masked_select(valid).mean()
                        si_rmse = torch.sqrt((((log_diff - mu).square()).masked_select(valid)).mean()).item()
                        # delta accuracies
                        ratio = torch.maximum(pd_safe / g_safe, g_safe / pd_safe)
                        d125 = ratio.masked_select(valid).lt(1.25).float().mean().item()
                        d1252 = ratio.masked_select(valid).lt(1.25**2).float().mean().item()
                        d1253 = ratio.masked_select(valid).lt(1.25**3).float().mean().item()
                        depth_absrels.append(rel)
                        depth_rmses.append(rmse)
                        depth_log_rmses.append(log_rmse)
                        depth_si_rmses.append(si_rmse)
                        depth_delta_125s.append(d125)
                        depth_delta_1252s.append(d1252)
                        depth_delta_1253s.append(d1253)
                        _agg("depth_absrel", vi, rel)
                        _agg("depth_delta_125", vi, d125)
                gt_pose = view.get("camera_pose", None)
                pr_pose = pred_vi.get("camera_pose", None) if pred_vi is not None else None
                if gt_pose is not None and isinstance(gt_pose, torch.Tensor) and pr_pose is not None and isinstance(pr_pose, torch.Tensor):
                    gp = gt_pose
                    pp = pr_pose
                    if pp.ndim == 3 and pp.shape[-2:] == (3, 4):
                        Rp = pp[:, :3, :3]
                        Rg = gp[:, :3, :3]
                        Rrel = Rp @ Rg.transpose(1, 2)
                        tr = Rrel[:, 0, 0] + Rrel[:, 1, 1] + Rrel[:, 2, 2]
                        val = torch.clamp((tr - 1) / 2, -1.0, 1.0)
                        ang = torch.rad2deg(torch.acos(val)).mean().item()
                        tp = pp[:, :3, 3]
                        tg = gp[:, :3, 3]
                        terr = torch.linalg.norm(tp - tg, dim=1).mean().item()
                        pose_rot_degs.append(ang)
                        pose_trans_errs.append(terr)
                        _agg("pose_rot_deg", vi, ang)
                        _agg("pose_trans_err", vi, terr)
                        ths = torch.linspace(0, 30, steps=31, device=pp.device)
                        auc = (ang <= ths).float().mean().item()
                        pose_auc30s.append(auc)
                        _agg("auc30", vi, auc)
                    elif pp.ndim == 2 and pp.shape[-1] == 7:
                        t = pp[:, :3]
                        q = pp[:, 3:]
                        qw, qx, qy, qz = q[:, 3], q[:, 0], q[:, 1], q[:, 2]
                        R11 = 1 - 2 * (qy * qy + qz * qz)
                        R12 = 2 * (qx * qy - qz * qw)
                        R13 = 2 * (qx * qz + qy * qw)
                        R21 = 2 * (qx * qy + qz * qw)
                        R22 = 1 - 2 * (qx * qx + qz * qz)
                        R23 = 2 * (qy * qz - qx * qw)
                        R31 = 2 * (qx * qz - qy * qw)
                        R32 = 2 * (qy * qz + qx * qw)
                        R33 = 1 - 2 * (qx * qx + qy * qy)
                        Rp = torch.stack(
                            [
                                torch.stack([R11, R12, R13], dim=-1),
                                torch.stack([R21, R22, R23], dim=-1),
                                torch.stack([R31, R32, R33], dim=-1),
                            ],
                            dim=1,
                        )
                        Rg = gp[:, :3, :3]
                        Rrel = Rp @ Rg.transpose(1, 2)
                        tr = Rrel[:, 0, 0] + Rrel[:, 1, 1] + Rrel[:, 2, 2]
                        val = torch.clamp((tr - 1) / 2, -1.0, 1.0)
                        ang = torch.rad2deg(torch.acos(val)).mean().item()
                        tg = gp[:, :3, 3]
                        terr = torch.linalg.norm(t - tg, dim=1).mean().item()
                        pose_rot_degs.append(ang)
                        pose_trans_errs.append(terr)
                        ths = torch.linspace(0, 30, steps=31, device=pp.device)
                        auc = (ang <= ths).float().mean().item()
                        pose_auc30s.append(auc)
                pr_pts3d = pred_vi.get("pts3d_in_other_view", None) if pred_vi is not None else None
                if pr_pts3d is not None and isinstance(pr_pts3d, torch.Tensor):
                    from dust3r.utils.geometry import depthmap_to_pts3d, geotrf
                    K = view.get("camera_intrinsics", None)
                    gp = view.get("camera_pose", None)
                    if not (gt_depth is not None and isinstance(gt_depth, torch.Tensor)):
                        raise RuntimeError("Missing gt depth for geometry metrics")
                    if not (K is not None and isinstance(K, torch.Tensor)):
                        raise RuntimeError("Missing camera intrinsics for geometry metrics")
                    if not (gp is not None and isinstance(gp, torch.Tensor)):
                        raise RuntimeError("Missing camera pose for geometry metrics")
                    pr = pr_pts3d
                    # ensure gt_depth [B,H,W]
                    gtd = gt_depth
                    if gtd.ndim == 4 and gtd.shape[1] == 1:
                        gtd = gtd.squeeze(1)
                    if gtd.ndim == 4 and gtd.shape[-1] == 1:
                        gtd = gtd.squeeze(-1)
                    B, H, W = gtd.shape[:3]
                    fu = K[:, 0, 0]
                    fv = K[:, 1, 1]
                    cu = K[:, 0, 2]
                    cv = K[:, 1, 2]
                    fu_map = fu.view(B, 1, 1).expand(B, H, W)
                    fv_map = fv.view(B, 1, 1).expand(B, H, W)
                    pseudo_focal = torch.stack([fu_map, fv_map], dim=1)  # [B,2,H,W]
                    pp = torch.stack([cu, cv], dim=1)  # [B,2]
                    gt_pts = depthmap_to_pts3d(depth=gtd, pseudo_focal=pseudo_focal, pp=pp)
                    gt_world = geotrf(gp, gt_pts)
                    pr_flat = pr.reshape(pr.shape[0], -1, 3)
                    gt_flat = gt_world.reshape(gt_world.shape[0], -1, 3)
                    if mask is not None and isinstance(mask, torch.Tensor):
                        m_flat = mask.reshape(mask.shape[0], -1)
                    else:
                        m_flat = torch.ones(pr_flat.shape[:2], dtype=torch.bool, device=pr_flat.device)
                    pr_sel = pr_flat[m_flat]
                    gt_sel = gt_flat[m_flat]
                    if pr_sel.numel() > 0 and gt_sel.numel() > 0:
                        pr_sel = pr_sel.view(-1, 3)
                        gt_sel = gt_sel.view(-1, 3)
                        dmat = torch.cdist(pr_sel.unsqueeze(0), gt_sel.unsqueeze(0), p=2).squeeze(0)
                        d_pred_to_gt = dmat.min(dim=1).values
                        d_gt_to_pred = dmat.min(dim=0).values
                        l1 = d_pred_to_gt.mean().item()
                        l2 = torch.sqrt(d_pred_to_gt.square().mean()).item()
                        pts3d_chamfer_l1s.append(l1)
                        pts3d_chamfer_l2s.append(l2)
                        acc_means.append(d_pred_to_gt.mean().item())
                        acc_meds.append(d_pred_to_gt.median().item())
                        comp_means.append(d_gt_to_pred.mean().item())
                        comp_meds.append(d_gt_to_pred.median().item())
                        _agg("acc_mean", vi, d_pred_to_gt.mean().item())
                        _agg("acc_med", vi, d_pred_to_gt.median().item())
                        _agg("comp_mean", vi, d_gt_to_pred.mean().item())
                        _agg("comp_med", vi, d_gt_to_pred.median().item())
                        B = pr.shape[0]
                        H, W = gt_pts.shape[-3:-1]
                        pr_grid = pr.reshape(B, H, W, 3)
                        gt_grid = gt_world.reshape(B, H, W, 3)
                        dx_pr = pr_grid[:, :, 1:, :] - pr_grid[:, :, :-1, :]
                        dy_pr = pr_grid[:, 1:, :, :] - pr_grid[:, :-1, :, :]
                        dx_gt = gt_grid[:, :, 1:, :] - gt_grid[:, :, :-1, :]
                        dy_gt = gt_grid[:, 1:, :, :] - gt_grid[:, :-1, :, :]
                        nx_pr = torch.linalg.cross(dx_pr[:, 1:, :, :], dy_pr[:, :, 1:, :], dim=-1)
                        nx_gt = torch.linalg.cross(dx_gt[:, 1:, :, :], dy_gt[:, :, 1:, :], dim=-1)
                        n_pr = torch.nn.functional.normalize(nx_pr, dim=-1)
                        n_gt = torch.nn.functional.normalize(nx_gt, dim=-1)
                        cos = (n_pr * n_gt).sum(dim=-1).clamp(-1, 1)
                        cos = cos.reshape(-1)
                        nc_means.append(cos.mean().item())
                        nc_meds.append(cos.median().item())
                        _agg("nc_mean", vi, cos.mean().item())
                        _agg("nc_med", vi, cos.median().item())
                pr_conf = pred_vi.get("conf", None) if pred_vi is not None else None
                if pr_conf is not None and isinstance(pr_conf, torch.Tensor):
                    conf_means.append(pr_conf.mean().item())
                if pred_vi is not None:
                    tconf = pred_vi.get("track_conf", None)
                    tvis = pred_vi.get("vis", None)
                    if isinstance(tconf, torch.Tensor):
                        track_conf_means.append(tconf.mean().item())
                        _agg("track_conf_mean", vi, tconf.mean().item())
                    if isinstance(tvis, torch.Tensor):
                        track_vis_ratios.append(tvis.float().mean().item())
                        _agg("track_vis_ratio", vi, tvis.float().mean().item())
            if depth_absrels:
                metric_logger.update(depth_absrel=float(np.mean(depth_absrels)))
            if depth_rmses:
                metric_logger.update(depth_rmse=float(np.mean(depth_rmses)))
            if depth_log_rmses:
                metric_logger.update(depth_log_rmse=float(np.mean(depth_log_rmses)))
            if depth_si_rmses:
                metric_logger.update(depth_si_rmse=float(np.mean(depth_si_rmses)))
            if depth_delta_125s:
                metric_logger.update(depth_delta_125=float(np.mean(depth_delta_125s)))
            if depth_delta_1252s:
                metric_logger.update(depth_delta_1252=float(np.mean(depth_delta_1252s)))
            if depth_delta_1253s:
                metric_logger.update(depth_delta_1253=float(np.mean(depth_delta_1253s)))
            if pose_rot_degs:
                metric_logger.update(pose_rot_deg=float(np.mean(pose_rot_degs)))
            if pose_trans_errs:
                metric_logger.update(pose_trans_err=float(np.mean(pose_trans_errs)))
            if pose_auc30s:
                metric_logger.update(pose_auc30=float(np.mean(pose_auc30s)))
            if pts3d_chamfer_l1s:
                metric_logger.update(pts3d_chamfer_l1=float(np.mean(pts3d_chamfer_l1s)))
            if pts3d_chamfer_l2s:
                metric_logger.update(pts3d_chamfer_l2=float(np.mean(pts3d_chamfer_l2s)))
            if acc_means:
                metric_logger.update(pts3d_acc_mean=float(np.mean(acc_means)))
            if acc_meds:
                metric_logger.update(pts3d_acc_med=float(np.mean(acc_meds)))
            if comp_means:
                metric_logger.update(pts3d_comp_mean=float(np.mean(comp_means)))
            if comp_meds:
                metric_logger.update(pts3d_comp_med=float(np.mean(comp_meds)))
            if nc_means:
                metric_logger.update(pts3d_nc_mean=float(np.mean(nc_means)))
            if nc_meds:
                metric_logger.update(pts3d_nc_med=float(np.mean(nc_meds)))
            if conf_means:
                metric_logger.update(conf_mean=float(np.mean(conf_means)))
            if track_conf_means:
                metric_logger.update(track_conf_mean=float(np.mean(track_conf_means)))
            if track_vis_ratios:
                metric_logger.update(track_vis_ratio=float(np.mean(track_vis_ratios)))

    printer.info("Averaged stats: %s", metric_logger)

    aggs = [("avg", "global_avg"), ("med", "median")]
    results = {
        f"{k}_{tag}": getattr(meter, attr)
        for k, meter in metric_logger.meters.items()
        for tag, attr in aggs
    }
    # write per-view metrics (paper subset) to metric_view.txt on main process
    try:
        if accelerator.is_main_process and hasattr(args, "output_dir"):
            outp = os.path.join(args.output_dir, "metric_view.txt")
            line = {"epoch": epoch}
            # compute per-view means; keep only: AUC@30, (Acc/Comp/NC)x(Mean/Med), AbsRel, delta<1.25
            view_dict = {}
            paper_keep = {
                "auc30",
                "acc_mean", "acc_med",
                "comp_mean", "comp_med",
                "nc_mean", "nc_med",
                "depth_absrel", "depth_delta_125",
            }
            for mname, mvals in per_view_metrics.items():
                if mname not in paper_keep:
                    continue
                for vi, arr in mvals.items():
                    if not arr:
                        continue
                    key = f"{mname}_v{vi+1}"
                    view_dict[key] = float(np.mean(arr))
            line[prefix] = view_dict
            with open(outp, "a", encoding="utf-8") as f:
                f.write(json.dumps(line) + "\n")
    except Exception:
        pass

    if log_writer is not None:
        for name, val in results.items():
            if isinstance(val, torch.Tensor):
                if val.ndim > 0:
                    continue
            if isinstance(val, dict):
                continue
            log_writer.add_scalar(prefix + "_" + name, val, 1000 * epoch)

    if getattr(args, "num_imgs_vis", 0) and args.num_imgs_vis > 0:
        depths_cross, gt_depths_cross = get_render_results(
            batch, result["pred"], self_view=False
        )
        for k in range(len(batch)):
            loss_details[f"pred_depth_{k+1}"] = depths_cross[k].detach().cpu()
            loss_details[f"gt_depth_{k+1}"] = gt_depths_cross[k].detach().cpu()

    # add original visualization inputs: gt_img_k, pred_rgb_k, masks and conf
    for k in range(len(batch)):
        view = batch[k]

        imgs = view["img"].detach().cpu()
            # [B,3,H,W] -> [B,H,W,3], keep value range as is (typically [-1,1])
        imgs_hw = imgs.permute(0, 2, 3, 1)
        loss_details[f"gt_img{k+1}"] = imgs_hw
        loss_details[f"pred_rgb_{k+1}"] = imgs_hw

        loss_details[f"img_mask_{k+1}"] = view["img_mask"].detach().cpu()

        loss_details[f"ray_mask_{k+1}"] = view["ray_mask"].detach().cpu()

        # conf: prefer depth_conf, fallback to conf
        pred_vi = result["pred"][k] if isinstance(result["pred"], list) else result["pred"]
        if isinstance(pred_vi, dict):
            loss_details[f"conf_{k+1}"] = pred_vi["depth_conf"].detach().cpu()

    if getattr(args, "num_imgs_vis", 0) and args.num_imgs_vis > 0:
        imgs_stacked_dict = get_vis_imgs_new(
            loss_details,
            args.num_imgs_vis,
            args.num_test_views,
            is_metric=batch[0]["is_metric"],
        )
        save_vis_imgs(args.output_dir, prefix, epoch, imgs_stacked_dict, num_views=args.num_test_views, modality=args.modality)

    del loss_details, loss_value, batch
    torch.cuda.empty_cache()

    return results


def batch_append(original_list, new_list):
    for sublist, new_item in zip(original_list, new_list):
        sublist.append(new_item)
    return original_list


def gen_mask_indicator(img_mask_list, ray_mask_list, num_views, h, w):
    output = []
    for img_mask, ray_mask in zip(img_mask_list, ray_mask_list):
        out = torch.zeros((h, w * num_views, 3))
        for i in range(num_views):
            if img_mask[i] and not ray_mask[i]:
                offset = 0
            elif not img_mask[i] and ray_mask[i]:
                offset = 1
            else:
                offset = 0.5
            out[:, i * w : (i + 1) * w] += offset
        output.append(out)
    return output


def get_vis_imgs_new(loss_details, num_imgs_vis, num_views, is_metric=False):
    preds = []
    gts = []
    for k in range(1, num_views + 1):
        pk = loss_details.get(f"pred_depth_{k}")
        gk = loss_details.get(f"gt_depth_{k}")
        if pk is None or gk is None:
            continue
        if isinstance(pk, torch.Tensor):
            pk = pk.squeeze().detach().cpu().numpy()
        if isinstance(gk, torch.Tensor):
            gk = gk.squeeze().detach().cpu().numpy()
        pm = colorize(pk)
        gm = colorize(gk)
        preds.append(pm)
        gts.append(gm)
    if not preds or not gts:
        return {}
    h, w, c = preds[0].shape
    num = min(num_imgs_vis, len(preds))
    pred_stack = np.concatenate(preds[:num], axis=1)
    gt_stack = np.concatenate(gts[:num], axis=1)
    return {"pred_depth": pred_stack, "gt_depth": gt_stack}


from PIL import Image, ImageDraw, ImageFont


def _view_type_label(idx: int, modality: str) -> str:
    if modality == "rgb":
        return "RGB"
    if modality == "event":
        return "event"
    if modality == "rgb_first_event":
        return "RGB" if idx == 0 else "event"
    if modality == "rgb_event_loop":
        return "RGB" if (idx % 2 == 0) else "event"
    return "RGB"


def save_vis_imgs(outdir, prefix, epoch, imgs_stacked_dict, step=None, num_views: int = None, modality: str = "rgb"):
    base = os.path.join(outdir, "visualize", f"epoch_{epoch}", str(prefix))
    os.makedirs(base, exist_ok=True)
    for name, imgs_stacked in imgs_stacked_dict.items():
        row_names = None
        if isinstance(imgs_stacked, dict) and "image" in imgs_stacked:
            arr_t = imgs_stacked["image"]
            row_names = imgs_stacked.get("row_names", None)
        else:
            arr_t = imgs_stacked
        arr = arr_t.detach().cpu().numpy()
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 1)
            arr = (arr * 255).astype(np.uint8)
        # annotate rows/columns
        try:
            H, W, C = arr.shape
            header_h = 56
            left_w = 180
            canvas = Image.new("RGB", (W + left_w, H + header_h), (0, 0, 0))
            img = Image.fromarray(arr)
            canvas.paste(img, (left_w, header_h))
            draw = ImageDraw.Draw(canvas)
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", size=18)
            except Exception:
                font = ImageFont.load_default()
            # column labels
            if num_views and num_views > 0:
                col_w = W // num_views
                for v in range(num_views):
                    x0 = left_w + v * col_w + 10
                    lbl = f"v{v+1} ({_view_type_label(v, modality)})"
                    draw.text((x0, 12), lbl, fill=(255, 255, 255), font=font)
            # row labels (dynamic rows)
            if row_names is None:
                # default to 6行；匹配 vis_and_cat 的拼接顺序
                row_names = ["Ray mask", "GT RGB", "Pred RGB", "GT Depth", "Pred Depth", "Conf"]
            n_rows = max(1, len(row_names))
            seg_h = H // n_rows
            for i, rn in enumerate(row_names):
                y = header_h + i * seg_h + seg_h // 2 - 10
                draw.text((12, y), rn, fill=(255, 255, 255), font=font)
            out_img = np.array(canvas)
        except Exception:
            out_img = arr
        fname = f"{name}.png" if step is None else f"{name}_step{step}.png"
        iio.imwrite(os.path.join(base, fname), out_img)


def vis_and_cat(
    gt_imgs,
    pred_imgs,
    cross_gt_depths,
    cross_pred_depths,
    cross_conf,
    ray_indicator,
    is_metric,
):
    cross_depth_gt_min = torch.quantile(cross_gt_depths, 0.01).item()
    cross_depth_gt_max = torch.quantile(cross_gt_depths, 0.99).item()
    cross_depth_pred_min = torch.quantile(cross_pred_depths, 0.01).item()
    cross_depth_pred_max = torch.quantile(cross_pred_depths, 0.99).item()
    cross_depth_min = min(cross_depth_gt_min, cross_depth_pred_min)
    cross_depth_max = max(cross_depth_gt_max, cross_depth_pred_max)

    cross_gt_depths_vis = colorize(
        cross_gt_depths,
        range=(
            (cross_depth_min, cross_depth_max)
            if is_metric
            else (cross_depth_gt_min, cross_depth_gt_max)
        ),
        append_cbar=True,
    )
    cross_pred_depths_vis = colorize(
        cross_pred_depths,
        range=(
            (cross_depth_min, cross_depth_max)
            if is_metric
            else (cross_depth_pred_min, cross_depth_pred_max)
        ),
        append_cbar=True,
    )


    if len(cross_conf) > 0:
        cross_conf_vis = colorize(cross_conf, append_cbar=True)

    gt_imgs_vis = torch.zeros_like(cross_gt_depths_vis)
    gt_imgs_vis[: gt_imgs.shape[0], : gt_imgs.shape[1]] = gt_imgs
    pred_imgs_vis = torch.zeros_like(cross_gt_depths_vis)
    pred_imgs_vis[: pred_imgs.shape[0], : pred_imgs.shape[1]] = pred_imgs
    ray_indicator_vis = torch.cat(
        [
            ray_indicator,
            torch.zeros(
                ray_indicator.shape[0],
                cross_pred_depths_vis.shape[1] - ray_indicator.shape[1],
                3,
            ),
        ],
        dim=1,
    )
    out = torch.cat(
        [
            ray_indicator_vis,
            gt_imgs_vis,
            pred_imgs_vis,
            cross_gt_depths_vis,
            cross_pred_depths_vis,
            cross_conf_vis,
        ],
        dim=0,
    )
    return out


def get_vis_imgs_new(loss_details, num_imgs_vis, num_views, is_metric):
    pred_keys = sorted([k for k in loss_details.keys() if k.startswith("pred_depth_")], key=lambda x: int(x.split("_")[-1]))
    gt_keys = sorted([k for k in loss_details.keys() if k.startswith("gt_depth_")], key=lambda x: int(x.split("_")[-1]))
    if len(pred_keys) == 0 or len(gt_keys) == 0:
        return {}
    eff_views = min(num_views, len(pred_keys), len(gt_keys))
    B = loss_details[pred_keys[0]].shape[0]
    n_vis = min(B, num_imgs_vis if num_imgs_vis and num_imgs_vis > 0 else B)
    ret_dict = {}
    gt_rows = []
    pred_rows = []
    conf_rows = []
    img_masks = []
    ray_masks = []
    for b in range(n_vis):
        gt_view_imgs = []
        pred_view_imgs = []
        conf_view_imgs = []
        img_mask_views = []
        ray_mask_views = []
        for vi in range(eff_views):
            pd = loss_details[pred_keys[vi]][b]
            gd = loss_details[gt_keys[vi]][b]
            gt_key = f"gt_img{vi+1}"
            pred_key = f"pred_rgb_{vi+1}"
            assert gt_key in loss_details
            assert pred_key in loss_details
            gt_img = 0.5 * (loss_details[gt_key][b] + 1).detach().cpu()
            pred_img = 0.5 * (loss_details[pred_key][b] + 1).detach().cpu()
            gt_view_imgs.append(gt_img)
            pred_view_imgs.append(pred_img)
            conf_key = f"conf_{vi+1}"
            assert conf_key in loss_details
            conf_view = loss_details[conf_key][b].detach().cpu()
            conf_view_imgs.append(conf_view)
            img_key = f"img_mask_{vi+1}"
            ray_key = f"ray_mask_{vi+1}"
            assert img_key in loss_details
            assert ray_key in loss_details
            img_mask_views.append(loss_details[img_key][b].detach().cpu())
            ray_mask_views.append(loss_details[ray_key][b].detach().cpu())
        gt_rows.append(torch.cat(gt_view_imgs, dim=1))
        pred_rows.append(torch.cat(pred_view_imgs, dim=1))
        conf_rows.append(torch.cat(conf_view_imgs, dim=1))
        img_masks.append(torch.stack(img_mask_views, dim=0))
        ray_masks.append(torch.stack(ray_mask_views, dim=0))
    # build cross-view depth grayscale by horizontal concat per sample
    cross_gt_depths = [torch.cat([loss_details[gt_keys[vi]][i] for vi in range(eff_views)], dim=1) for i in range(n_vis)]
    cross_pred_depths = [torch.cat([loss_details[pred_keys[vi]][i] for vi in range(eff_views)], dim=1) for i in range(n_vis)]
    # ray indicator
    indicators = gen_mask_indicator(img_masks, ray_masks, eff_views, 30, gt_rows[0].shape[1] // eff_views)
    for i in range(n_vis):
        out = vis_and_cat(
            gt_rows[i],
            pred_rows[i],
            cross_gt_depths[i],
            cross_pred_depths[i],
            conf_rows[i],
            indicators[i],
            is_metric if not isinstance(is_metric, (list, tuple)) else is_metric[i],
        )
        # 动态行名（与 vis_and_cat 的拼接顺序一致）
        row_names = ["Ray mask", "GT RGB", "Pred RGB", "GT Depth", "Pred Depth", "Conf"]
        ret_dict[f"imgs_{i}"] = {"image": out, "row_names": row_names}
    return ret_dict


@hydra.main(
    version_base=None,
    config_path=str(os.path.dirname(os.path.abspath(__file__))) + "/../config",
    config_name="train.yaml",
)
def run(cfg: OmegaConf):
    OmegaConf.resolve(cfg)
    logdir = pathlib.Path(cfg.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    train(cfg)


if __name__ == "__main__":
    def _rewrite_hydra_args(argv):
        out = [argv[0]]
        i = 1
        while i < len(argv):
            if argv[i] == "--modality" and i + 1 < len(argv):
                out.append(f"modality={argv[i+1]}")
                i += 2
                continue
            out.append(argv[i])
            i += 1
        return out
    sys.argv = _rewrite_hydra_args(sys.argv)
    run()
