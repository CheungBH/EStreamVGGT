# Commands

## Assumptions

- Project root: `/Users/cheungbh/Documents/PhDCode/EStreamVGGT`
- Run commands from `src/`
- `finetune_backbone.py` expects:
  - `DATA_ROOT/train/<sequence>/000000.png`
  - `DATA_ROOT/train/<sequence>/000000_event.png`
  - `DATA_ROOT/val/<sequence>/000000.png`
  - `DATA_ROOT/val/<sequence>/000000_event.png`

Example:

```text
DATA_ROOT/
  train/
    seq_001/
      000000.png
      000000_event.png
      000002.png
      000002_event.png
  val/
    seq_002/
      000000.png
      000000_event.png
```

## Common Setup

```bash
cd /Users/cheungbh/Documents/PhDCode/EStreamVGGT/src
```

## Backbone Distillation

### Single GPU

#### DSEC

```bash
CUDA_VISIBLE_DEVICES=0 python finetune_backbone.py \
  --data-root /path/to/DSEC \
  --pretrained /path/to/vggt_checkpoint.pth \
  --output-dir ../outputs/backbone_dsec \
  --epochs 10 \
  --batch-size 8 \
  --num-workers 4 \
  --feature-level all \
  --normalize-features \
  --amp
```

#### EventScape

```bash
CUDA_VISIBLE_DEVICES=0 python finetune_backbone.py \
  --data-root /path/to/EventScape \
  --pretrained /path/to/vggt_checkpoint.pth \
  --output-dir ../outputs/backbone_eventscape \
  --epochs 10 \
  --batch-size 8 \
  --num-workers 4 \
  --feature-level all \
  --normalize-features \
  --amp
```

### Multi GPU with Accelerate

First time only:

```bash
accelerate config
```

#### 2 GPUs

```bash
accelerate launch --multi_gpu --num_processes 2 finetune_backbone.py \
  --data-root /path/to/DSEC \
  --pretrained /path/to/vggt_checkpoint.pth \
  --output-dir ../outputs/backbone_dsec \
  --epochs 10 \
  --batch-size 8 \
  --num-workers 4 \
  --feature-level all \
  --normalize-features \
  --amp
```

#### 4 GPUs

```bash
accelerate launch --multi_gpu --num_processes 4 finetune_backbone.py \
  --data-root /path/to/DSEC \
  --pretrained /path/to/vggt_checkpoint.pth \
  --output-dir ../outputs/backbone_dsec \
  --epochs 10 \
  --batch-size 8 \
  --num-workers 4 \
  --feature-level all \
  --normalize-features \
  --amp
```

### Useful Variants

#### Last layer only

```bash
CUDA_VISIBLE_DEVICES=0 python finetune_backbone.py \
  --data-root /path/to/DSEC \
  --pretrained /path/to/vggt_checkpoint.pth \
  --output-dir ../outputs/backbone_dsec_last \
  --epochs 10 \
  --batch-size 8 \
  --num-workers 4 \
  --feature-level last \
  --normalize-features \
  --amp
```

#### Custom event suffix

```bash
CUDA_VISIBLE_DEVICES=0 python finetune_backbone.py \
  --data-root /path/to/DSEC \
  --pretrained /path/to/vggt_checkpoint.pth \
  --output-dir ../outputs/backbone_dsec_evt \
  --epochs 10 \
  --batch-size 8 \
  --num-workers 4 \
  --event-suffix _evt \
  --amp
```

#### Custom image extension

```bash
CUDA_VISIBLE_DEVICES=0 python finetune_backbone.py \
  --data-root /path/to/DSEC \
  --pretrained /path/to/vggt_checkpoint.pth \
  --output-dir ../outputs/backbone_dsec_jpg \
  --epochs 10 \
  --batch-size 8 \
  --num-workers 4 \
  --image-ext .jpg \
  --amp
```

#### Different LoRA target

```bash
CUDA_VISIBLE_DEVICES=0 python finetune_backbone.py \
  --data-root /path/to/DSEC \
  --pretrained /path/to/vggt_checkpoint.pth \
  --output-dir ../outputs/backbone_dsec_attn \
  --epochs 10 \
  --batch-size 8 \
  --num-workers 4 \
  --lora-r 8 \
  --lora-alpha 8 \
  --lora-target attn \
  --feature-level all \
  --normalize-features \
  --amp
```

#### Gradient accumulation

```bash
accelerate launch --multi_gpu --num_processes 4 finetune_backbone.py \
  --data-root /path/to/DSEC \
  --pretrained /path/to/vggt_checkpoint.pth \
  --output-dir ../outputs/backbone_dsec_accum \
  --epochs 10 \
  --batch-size 4 \
  --gradient-accumulation-steps 2 \
  --num-workers 4 \
  --feature-level all \
  --normalize-features \
  --amp
```

## Backbone Evaluation

`evaluate_backbone.py` uses an existing Hydra yaml config and runs:

- `rgb_on_rgb`
- `rgb_on_event`
- `event_lora_on_event`

Outputs are written under separate subdirectories in `--output-dir`.

### DSEC

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_backbone.py \
  --config ../config/finetune_dsec_event.yaml \
  --lora-ckpt ../outputs/backbone_dsec/checkpoint_best.pth \
  --output-dir ../outputs/eval_backbone_dsec
```

### EventScape

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_backbone.py \
  --config ../config/finetune_eventscape_event.yaml \
  --lora-ckpt ../outputs/backbone_eventscape/checkpoint_best.pth \
  --output-dir ../outputs/eval_backbone_eventscape
```

### Run selected modes only

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_backbone.py \
  --config ../config/finetune_dsec_event.yaml \
  --lora-ckpt ../outputs/backbone_dsec/checkpoint_best.pth \
  --output-dir ../outputs/eval_backbone_dsec_selected \
  --modes rgb_on_event event_lora_on_event
```

### Override pretrained checkpoint in evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_backbone.py \
  --config ../config/finetune_dsec_event.yaml \
  --pretrained /path/to/vggt_checkpoint.pth \
  --lora-ckpt ../outputs/backbone_dsec/checkpoint_best.pth \
  --output-dir ../outputs/eval_backbone_dsec_override
```

### Override batch size and workers in evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_backbone.py \
  --config ../config/finetune_dsec_event.yaml \
  --lora-ckpt ../outputs/backbone_dsec/checkpoint_best.pth \
  --output-dir ../outputs/eval_backbone_dsec_fast \
  --batch-size 1 \
  --num-workers 8
```

## Useful Output Files

### finetune_backbone.py

- `OUTPUT_DIR/cmd.txt`
- `OUTPUT_DIR/args.json`
- `OUTPUT_DIR/checkpoint_last.pth`
- `OUTPUT_DIR/checkpoint_best.pth`
- `OUTPUT_DIR/checkpoint_epoch_XXX.pth`

### evaluate_backbone.py

- `OUTPUT_DIR/summary.json`
- `OUTPUT_DIR/rgb_on_rgb/metric.json`
- `OUTPUT_DIR/rgb_on_event/metric.json`
- `OUTPUT_DIR/event_lora_on_event/metric.json`
- plots under each mode's `visualize/`
