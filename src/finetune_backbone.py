import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed as accelerate_set_seed
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm import tqdm

from vggt.lora import LoRALinear, apply_lora_to_aggregator, mark_only_lora_trainable
from vggt.models.vggt import VGGT


def parse_args():
    parser = argparse.ArgumentParser("Backbone LoRA distillation for event encoder")
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Dataset root containing train/ and val/ subfolders with paired name.png/name_event.png",
    )
    parser.add_argument("--pretrained", type=str, default=None, help="VGGT checkpoint path")
    parser.add_argument("--output-dir", type=str, required=True, help="Where to save LoRA checkpoints")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--event-suffix", type=str, default="_event")
    parser.add_argument("--image-ext", type=str, default=".png")
    parser.add_argument("--feature-level", type=str, default="all", choices=["last", "all"])
    parser.add_argument("--normalize-features", action="store_true")
    parser.add_argument("--use-special-tokens", action="store_true")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=8)
    parser.add_argument(
        "--lora-target",
        type=str,
        default="all",
        choices=["all", "attn", "qkv", "qk", "mlp"],
    )
    parser.add_argument("--img-size", type=int, default=518)
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--embed-dim", type=int, default=1024)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_image_transform(img_size: int):
    return T.Compose(
        [
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


class PairedImageDataset(Dataset):
    def __init__(self, root: str, event_suffix: str, image_ext: str, img_size: int):
        self.root = Path(root)
        self.event_suffix = event_suffix
        self.image_ext = image_ext
        self.transform = build_image_transform(img_size)
        self.pairs = self._scan_pairs()

    def _scan_pairs(self):
        if not self.root.exists():
            raise RuntimeError(f"Dataset folder not found: {self.root}")
        pairs = []

        def sort_key(name: str):
            stem = Path(name).stem
            if stem.endswith(self.event_suffix):
                stem = stem[: -len(self.event_suffix)]
            return (0, int(stem)) if stem.isdigit() else (1, stem)

        with os.scandir(self.root) as seq_iter:
            seq_entries = sorted(
                [entry for entry in seq_iter if entry.is_dir()],
                key=lambda x: x.name,
            )

        for seq_entry in seq_entries:
            with os.scandir(seq_entry.path) as file_iter:
                filenames = [
                    entry.name
                    for entry in file_iter
                    if entry.is_file() and entry.name.endswith(self.image_ext)
                ]
            filename_set = set(filenames)
            for name in sorted(filenames, key=sort_key):
                stem = Path(name).stem
                if stem.endswith(self.event_suffix):
                    continue
                event_name = f"{stem}{self.event_suffix}{self.image_ext}"
                if event_name in filename_set:
                    pairs.append(
                        (
                            Path(seq_entry.path) / name,
                            Path(seq_entry.path) / event_name,
                        )
                    )

        if not pairs:
            raise RuntimeError(
                f"No RGB/event pairs found in {self.root}. "
                f"Expected files like `name{self.image_ext}` and `name{self.event_suffix}{self.image_ext}`."
            )
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        rgb_path, event_path = self.pairs[idx]
        rgb = Image.open(rgb_path).convert("RGB")
        event = Image.open(event_path).convert("RGB")
        return {
            "rgb": self.transform(rgb),
            "event": self.transform(event),
            "rgb_path": str(rgb_path),
            "event_path": str(event_path),
        }


def build_loader(args, root: str, is_train: bool):
    dataset = PairedImageDataset(
        root=root,
        event_suffix=args.event_suffix,
        image_ext=args.image_ext,
        img_size=args.img_size,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=is_train,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=is_train,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=(4 if args.num_workers > 0 else None),
    )


def resolve_split_roots(args):
    data_root = Path(args.data_root)
    train_root = data_root / "train"
    val_root = data_root / "val"
    if not train_root.exists():
        raise RuntimeError(
            f"Expected training folder at `{train_root}`. "
            "Expected `--data-root` to contain `train/` and `val/` subfolders."
        )
    if not val_root.exists():
        raise RuntimeError(
            f"Expected validation folder at `{val_root}`. "
            "Expected `--data-root` to contain `train/` and `val/` subfolders."
        )
    return str(train_root), str(val_root)


def unwrap_checkpoint(ckpt):
    if isinstance(ckpt, dict):
        for key in ("model", "state_dict"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    return ckpt


def build_model(args, ckpt_path: str, device: torch.device, use_lora: bool):
    model = VGGT(img_size=args.img_size, patch_size=args.patch_size, embed_dim=args.embed_dim)
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = unwrap_checkpoint(ckpt)
    model.load_state_dict(state_dict, strict=True)
    if use_lora:
        apply_lora_to_aggregator(model.aggregator, r=args.lora_r, alpha=args.lora_alpha, target=args.lora_target)
        mark_only_lora_trainable(model)
    else:
        for p in model.parameters():
            p.requires_grad = False
    model.to(device)
    return model


def lora_state_dict(model):
    state = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear) and module.lora_A is not None:
            state[f"{name}.lora_A.weight"] = module.lora_A.weight.detach().cpu()
            state[f"{name}.lora_B.weight"] = module.lora_B.weight.detach().cpu()
    return state


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def extract_backbone_features(model, images, device, accelerator, use_amp):
    images = images.to(device=device, non_blocking=True).unsqueeze(1)
    images = (images + 1.0) / 2.0
    with accelerator.autocast():
        feats, patch_start_idx = model.aggregator(images)
    return feats, patch_start_idx


def compute_feature_loss(student_feats, teacher_feats, patch_start_idx, args):
    if args.feature_level == "last":
        pairs = [(student_feats[-1], teacher_feats[-1])]
    else:
        pairs = list(zip(student_feats, teacher_feats))

    total = 0.0
    for student_feat, teacher_feat in pairs:
        if not args.use_special_tokens:
            student_feat = student_feat[:, :, patch_start_idx:, :]
            teacher_feat = teacher_feat[:, :, patch_start_idx:, :]
        teacher_feat = teacher_feat.detach()
        if args.normalize_features:
            student_feat = F.normalize(student_feat, dim=-1)
            teacher_feat = F.normalize(teacher_feat, dim=-1)
        total = total + F.mse_loss(student_feat.float(), teacher_feat.float())
    return total / len(pairs)


def save_checkpoint(student, optimizer, epoch, step, output_dir, accelerator, is_best=False):
    if not accelerator.is_main_process:
        return
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_student = accelerator.unwrap_model(student)
    payload = {
        "epoch": epoch,
        "step": step,
        "lora": lora_state_dict(raw_student),
        "student_state_dict": raw_student.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(payload, output_dir / f"checkpoint_epoch_{epoch:03d}.pth")
    if is_best:
        torch.save(payload, output_dir / "checkpoint_best.pth")
    torch.save(payload, output_dir / "checkpoint_last.pth")


def evaluate(student, teacher, data_loader, device, args, accelerator):
    teacher.eval()
    student.eval()
    total_loss = 0.0
    total_weight = 0.0
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Eval", leave=False, disable=not accelerator.is_local_main_process):
            teacher_feats, patch_start_idx = extract_backbone_features(teacher, batch["rgb"], device, accelerator, args.amp)
            student_feats, _ = extract_backbone_features(student, batch["event"], device, accelerator, args.amp)
            loss = compute_feature_loss(student_feats, teacher_feats, patch_start_idx, args)
            local_bs = torch.tensor([batch["rgb"].shape[0]], device=device, dtype=torch.float32)
            weighted_loss = loss.detach() * local_bs
            gathered_loss = accelerator.gather_for_metrics(weighted_loss.reshape(1))
            gathered_bs = accelerator.gather_for_metrics(local_bs)
            total_loss += float(gathered_loss.sum().item())
            total_weight += float(gathered_bs.sum().item())
    return total_loss / max(total_weight, 1.0)


def main():
    args = parse_args()
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=("fp16" if args.amp else "no"),
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False)],
    )
    set_seed(args.seed)
    accelerate_set_seed(args.seed, device_specific=True)
    if not args.pretrained:
        raise RuntimeError("Please provide --pretrained")

    device = accelerator.device
    train_root, val_root = resolve_split_roots(args)

    train_loader = build_loader(args, root=train_root, is_train=True)
    eval_loader = build_loader(args, root=val_root, is_train=False)

    teacher = build_model(args, args.pretrained, device, use_lora=False)
    student = build_model(args, args.pretrained, device, use_lora=True)

    trainable = [p for p in student.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable LoRA parameters found in student model")
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    student, optimizer, train_loader, eval_loader = accelerator.prepare(
        student, optimizer, train_loader, eval_loader
    )

    output_dir = Path(args.output_dir)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "cmd.txt", "w", encoding="utf-8") as f:
            f.write(" ".join(sys.argv) + "\n")
        with open(output_dir / "args.json", "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2, ensure_ascii=False)
    accelerator.wait_for_everyone()

    accelerator.print(f"Teacher params trainable: {count_trainable_parameters(teacher)}")
    accelerator.print(f"Student LoRA params trainable: {count_trainable_parameters(accelerator.unwrap_model(student))}")
    accelerator.print(f"Train loader batches: {len(train_loader)}")
    accelerator.print(f"Eval loader batches: {len(eval_loader)}")
    accelerator.print(f"World size: {accelerator.num_processes}")

    best_eval = float("inf")
    global_step = 0

    for epoch in range(args.epochs):
        teacher.eval()
        student.train()
        running_loss = 0.0
        running_weight = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", disable=not accelerator.is_local_main_process)
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(pbar):
            with accelerator.accumulate(student):
                with torch.no_grad():
                    teacher_feats, patch_start_idx = extract_backbone_features(teacher, batch["rgb"], device, accelerator, args.amp)
                student_feats, _ = extract_backbone_features(student, batch["event"], device, accelerator, args.amp)

                loss = compute_feature_loss(student_feats, teacher_feats, patch_start_idx, args)
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            loss_value = float(loss.item())
            local_bs = float(batch["rgb"].shape[0])
            running_loss += loss_value * local_bs
            running_weight += local_bs
            global_step += 1

            if step % args.log_every == 0:
                pbar.set_postfix(loss=f"{loss_value:.6f}", avg=f"{running_loss / max(running_weight, 1.0):.6f}")

        train_loss_tensor = torch.tensor([running_loss, running_weight], device=device, dtype=torch.float32)
        train_loss_tensor = accelerator.reduce(train_loss_tensor, reduction="sum")
        train_loss_avg = float(train_loss_tensor[0].item() / max(train_loss_tensor[1].item(), 1.0))

        eval_loss = evaluate(student, teacher, eval_loader, device, args, accelerator)
        is_best = eval_loss < best_eval
        best_eval = min(best_eval, eval_loss)
        accelerator.print(f"[Epoch {epoch}] train_loss={train_loss_avg:.6f} eval_loss={eval_loss:.6f}")

        if (epoch + 1) % args.save_every == 0 or is_best:
            save_checkpoint(student, optimizer, epoch, global_step, output_dir, accelerator, is_best=is_best)
        accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
