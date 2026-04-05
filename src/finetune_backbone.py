import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
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
        for rgb_path in sorted(self.root.rglob(f"*{self.image_ext}")):
            if rgb_path.stem.endswith(self.event_suffix):
                continue
            event_path = rgb_path.with_name(f"{rgb_path.stem}{self.event_suffix}{rgb_path.suffix}")
            if event_path.exists():
                pairs.append((rgb_path, event_path))
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


def extract_backbone_features(model, images, device, use_amp):
    images = images.to(device=device, non_blocking=True).unsqueeze(1)
    images = (images + 1.0) / 2.0
    if device.type == "cuda":
        amp_ctx = torch.amp.autocast("cuda", enabled=use_amp)
    else:
        amp_ctx = torch.amp.autocast("cpu", enabled=False)
    with amp_ctx:
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


def save_checkpoint(student, optimizer, epoch, step, output_dir, is_best=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "step": step,
        "lora": lora_state_dict(student),
        "student_state_dict": student.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(payload, output_dir / f"checkpoint_epoch_{epoch:03d}.pth")
    if is_best:
        torch.save(payload, output_dir / "checkpoint_best.pth")
    torch.save(payload, output_dir / "checkpoint_last.pth")


def evaluate(student, teacher, data_loader, device, args):
    teacher.eval()
    student.eval()
    total_loss = 0.0
    total_steps = 0
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Eval", leave=False):
            teacher_feats, patch_start_idx = extract_backbone_features(teacher, batch["rgb"], device, args.amp)
            student_feats, _ = extract_backbone_features(student, batch["event"], device, args.amp)
            loss = compute_feature_loss(student_feats, teacher_feats, patch_start_idx, args)
            total_loss += float(loss.item())
            total_steps += 1
    return total_loss / max(total_steps, 1)


def main():
    args = parse_args()
    set_seed(args.seed)
    if not args.pretrained:
        raise RuntimeError("Please provide --pretrained")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_root, val_root = resolve_split_roots(args)

    train_loader = build_loader(args, root=train_root, is_train=True)
    eval_loader = build_loader(args, root=val_root, is_train=False)

    teacher = build_model(args, args.pretrained, device, use_lora=False)
    student = build_model(args, args.pretrained, device, use_lora=True)

    trainable = [p for p in student.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable LoRA parameters found in student model")
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "cmd.txt", "w", encoding="utf-8") as f:
        f.write(" ".join(sys.argv) + "\n")
    with open(output_dir / "args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    print(f"Teacher params trainable: {count_trainable_parameters(teacher)}")
    print(f"Student LoRA params trainable: {count_trainable_parameters(student)}")
    print(f"Train loader batches: {len(train_loader)}")
    print(f"Eval loader batches: {len(eval_loader)}")

    best_eval = float("inf")
    global_step = 0

    for epoch in range(args.epochs):
        teacher.eval()
        student.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for step, batch in enumerate(pbar):
            optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                teacher_feats, patch_start_idx = extract_backbone_features(teacher, batch["rgb"], device, args.amp)
            student_feats, _ = extract_backbone_features(student, batch["event"], device, args.amp)

            loss = compute_feature_loss(student_feats, teacher_feats, patch_start_idx, args)
            loss.backward()
            optimizer.step()

            loss_value = float(loss.item())
            running_loss += loss_value
            global_step += 1

            if step % args.log_every == 0:
                pbar.set_postfix(loss=f"{loss_value:.6f}", avg=f"{running_loss / (step + 1):.6f}")

        eval_loss = evaluate(student, teacher, eval_loader, device, args)
        is_best = eval_loss < best_eval
        best_eval = min(best_eval, eval_loss)
        print(f"[Epoch {epoch}] train_loss={running_loss / max(len(train_loader), 1):.6f} eval_loss={eval_loss:.6f}")

        if (epoch + 1) % args.save_every == 0 or is_best:
            save_checkpoint(student, optimizer, epoch, global_step, output_dir, is_best=is_best)


if __name__ == "__main__":
    main()
