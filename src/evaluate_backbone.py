import argparse
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

import finetune as ft


def parse_args():
    parser = argparse.ArgumentParser("Evaluate RGB/Event backbone variants")
    parser.add_argument("--config", type=str, required=True, help="Path to finetune yaml config")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save evaluation outputs")
    parser.add_argument("--lora-ckpt", type=str, default=None, help="LoRA checkpoint from finetune_backbone.py")
    parser.add_argument("--pretrained", type=str, default=None, help="Optional override for pretrained VGGT checkpoint")
    parser.add_argument("--batch-size", type=int, default=None, help="Optional override for eval batch size")
    parser.add_argument("--num-workers", type=int, default=None, help="Optional override for eval workers")
    parser.add_argument(
        "--modes",
        type=str,
        nargs="+",
        default=["rgb_on_rgb", "rgb_on_event", "event_lora_on_event"],
        choices=["rgb_on_rgb", "rgb_on_event", "event_lora_on_event"],
        help="Evaluation modes to run",
    )
    return parser.parse_args()


def cfg_to_args(cfg):
    data = OmegaConf.to_container(cfg, resolve=True)
    return SimpleNamespace(**data)


def replace_modality(expr: str, modality: str) -> str:
    new_expr, count = re.subn(r'modality\s*=\s*["\'][^"\']+["\']', f'modality="{modality}"', expr)
    if count == 0:
        raise RuntimeError(f"Cannot replace modality in dataset expression:\n{expr}")
    return new_expr


def resolve_pretrained(config_path: str, pretrained: str) -> str:
    p = Path(pretrained)
    if p.is_absolute():
        return str(p)
    return str((Path(config_path).resolve().parent / p).resolve())


def load_lora_meta(lora_ckpt: str):
    ckpt_path = Path(lora_ckpt).resolve()
    args_path = ckpt_path.parent / "args.json"
    meta = {"lora_r": 8, "lora_alpha": 8, "lora_target": "all"}
    if args_path.exists():
        with open(args_path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        meta["lora_r"] = int(saved.get("lora_r", meta["lora_r"]))
        meta["lora_alpha"] = int(saved.get("lora_alpha", meta["lora_alpha"]))
        meta["lora_target"] = saved.get("lora_target", meta["lora_target"])
    return meta


def load_lora_weights(model, lora_ckpt: str):
    payload = torch.load(lora_ckpt, map_location="cpu")
    if isinstance(payload, dict) and "student_state_dict" in payload:
        missing, unexpected = model.load_state_dict(payload["student_state_dict"], strict=False)
    elif isinstance(payload, dict) and "lora" in payload:
        missing, unexpected = model.load_state_dict(payload["lora"], strict=False)
    else:
        missing, unexpected = model.load_state_dict(payload, strict=False)
    return missing, unexpected


def build_model(args, mode: str, device: torch.device):
    model = ft.VGGT()
    ckpt = torch.load(args.pretrained, map_location=device)
    model.load_state_dict(ckpt, strict=True)

    if mode == "event_lora_on_event":
        if not args.lora_ckpt:
            raise RuntimeError("event_lora_on_event requires --lora-ckpt")
        meta = load_lora_meta(args.lora_ckpt)
        ft.apply_lora_to_aggregator(
            model.aggregator,
            r=meta["lora_r"],
            alpha=meta["lora_alpha"],
            target=meta["lora_target"],
        )
        missing, unexpected = load_lora_weights(model, args.lora_ckpt)
        ft.printer.info(
            "Loaded LoRA checkpoint %s (missing=%d, unexpected=%d)",
            args.lora_ckpt,
            len(missing),
            len(unexpected),
        )

    model.to(device)
    model.eval()
    return model


def build_eval_args(base_args, mode: str, out_dir: str):
    args = SimpleNamespace(**vars(base_args))
    modality = {
        "rgb_on_rgb": "rgb",
        "rgb_on_event": "event",
        "event_lora_on_event": "event",
    }[mode]
    args.modality = modality
    args.output_dir = out_dir
    args.dataset_test = replace_modality(args.dataset_test, modality)
    args.test_dataset = replace_modality(args.test_dataset, modality)
    return args


def summarize_stats(stats):
    summary = {}
    preferred = [
        "depth_absrel_avg",
        "depth_rmse_avg",
        "depth_log_rmse_avg",
        "depth_si_rmse_avg",
        "depth_delta_125_avg",
        "pose_rot_deg_avg",
        "pose_trans_err_avg",
        "pose_auc30_avg",
        "pts3d_acc_avg",
        "pts3d_comp_avg",
        "pts3d_nc_avg",
        "pts3d_chamfer_l1_avg",
        "pts3d_chamfer_l2_avg",
        "track_conf_mean_avg",
        "track_vis_ratio_avg",
        "Regr3DPose_pts3d_avg",
        "Regr3DPose_ScaleInv_pts3d_avg",
    ]
    for key in preferred:
        if key in stats:
            summary[key] = float(stats[key])
    return summary


def run_one_mode(base_args, mode: str, accelerator):
    mode_dir = os.path.join(base_args.output_dir, mode)
    os.makedirs(mode_dir, exist_ok=True)
    args = build_eval_args(base_args, mode, mode_dir)
    device = accelerator.device

    ft.printer.info("Running mode `%s` with modality `%s`", mode, args.modality)
    ft.printer.info("Dataset: %s", args.test_dataset)

    data_loader = ft.build_dataset(
        args.test_dataset,
        args.batch_size,
        args.num_workers,
        accelerator=accelerator,
        test=True,
        fixed_length=True,
    )
    model = build_model(args, mode, device)
    criterion_expr = args.test_criterion or args.criterion
    criterion = eval(criterion_expr, vars(ft)).to(device)

    stats = ft.test_one_epoch(
        model,
        None,
        criterion,
        data_loader,
        accelerator,
        device,
        epoch=0,
        args=args,
        log_writer=None,
        prefix=mode,
    )
    ft.plot_view_metrics(mode_dir, args.modality, getattr(args, "num_test_views", 0))
    ft.plot_category_dashboards(mode_dir)
    return summarize_stats(stats)


def main():
    cli_args = parse_args()
    cfg = OmegaConf.load(cli_args.config)
    OmegaConf.resolve(cfg)
    args = cfg_to_args(cfg)

    if cli_args.pretrained is not None:
        args.pretrained = cli_args.pretrained
    args.pretrained = resolve_pretrained(cli_args.config, args.pretrained)
    args.output_dir = cli_args.output_dir
    args.lora_ckpt = cli_args.lora_ckpt
    if cli_args.batch_size is not None:
        args.batch_size = cli_args.batch_size
    if cli_args.num_workers is not None:
        args.num_workers = cli_args.num_workers

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "cmd.txt"), "w", encoding="utf-8") as f:
        f.write(" ".join(sys.argv) + "\n")
    with open(os.path.join(args.output_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(cli_args), f, indent=2, ensure_ascii=False)

    accelerator = ft.Accelerator()
    ft.setup_for_distributed(accelerator)

    summary = {}
    for mode in cli_args.modes:
        summary[mode] = run_one_mode(args, mode, accelerator)

    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    ft.printer.info("Saved summary to %s", os.path.join(args.output_dir, "summary.json"))


if __name__ == "__main__":
    main()
