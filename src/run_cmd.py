import os

cmds = [
    # "python finetune.py --config-name finetune_5090_RGB_event_loop +lora.enable=true",
    # "python finetune.py --config-name finetune_5090_RGB_event +lora.enable=true",
    "python finetune.py --config-name finetune_5090_RGB_empty +lora.enable=true",
    # "python finetune.py --config-name finetune_5090_RGB +lora.enable=true",
    # "python finetune.py --config-name finetune_5090_event +lora.enable=true",
    # "python evaluate_plot.py --config ../config/finetune_5090_event.yaml",
    # "python evaluate_plot.py --config ../config/finetune_5090_RGB.yaml",
    # "python evaluate_plot.py --config ../config/finetune_5090_RGB_event.yaml",
    # "python evaluate_plot.py --config ../config/finetune_5090_RGB_event_loop.yaml",

]

for cmd in cmds:
    print(f"Running: {cmd}")
    os.system(cmd)
