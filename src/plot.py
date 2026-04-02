import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def plot_all_metrics(output_dir):
    mpath = os.path.join(output_dir, "metric.json")
    if not os.path.exists(mpath):
        return
    try:
        with open(mpath, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        obj = {}
    if not isinstance(obj, dict):
        return
    series = {}
    prefix = "eval"
    for epoch_key, metrics in obj.items():
        if not (isinstance(epoch_key, str) and epoch_key.startswith("Epoch")):
            continue
        if not isinstance(metrics, dict):
            continue
        try:
            epoch = int(epoch_key.replace("Epoch", ""))
        except Exception:
            continue
        for k, v in metrics.items():
            if isinstance(v, (int, float, np.number)):
                key = f"{prefix}/{k}"
                series.setdefault(key, []).append((epoch, float(v)))
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
    mpath = os.path.join(output_dir, "metric_views.json")

    def view_type(v, modality):
        if modality == "rgb":
            return "RGB"
        if modality == "event":
            return "event"
        if modality == "rgb_first_event":
            return "RGB" if v == 0 else "event"
        if modality == "rgb_event_loop":
            return "RGB" if (v % 2 == 0) else "event"
        if modality == "rgb_empty":
            return "RGB" if v == 0 else "white"
        return "RGB"
    wanted = ["auc30", "acc", "comp", "nc", "depth_absrel", "depth_delta_125"]
    with open(mpath, "r", encoding="utf-8") as f:
        obj = json.load(f)

    prefixes = ["eval"]
    base = os.path.join(output_dir, "visualize", "metrics_views")
    os.makedirs(base, exist_ok=True)
    for prefix in prefixes:
        series = {}
        view_ids = set()
        for epoch_key, views in obj.items():
            for view_key in views.keys():
                if isinstance(view_key, str) and view_key.startswith("view"):
                    view_ids.add(int(view_key.replace("view", "")))

        view_range = range(1, num_views + 1)

        for epoch_key, views in obj.items():
            ep = int(epoch_key.replace("epoch", ""))
            for m in wanted:
                for v in view_range:
                    vals = views.get(f"view{v}", {})
                    val = vals.get(m, None) if isinstance(vals, dict) else None
                    if isinstance(val, (int, float, np.number)):
                        series.setdefault(m, {}).setdefault(v, []).append((ep, float(val)))
        for m in wanted:
            by_view = series.get(m, {})
            views_sorted = sorted(by_view.keys())
            if not views_sorted:
                plt.figure(figsize=(8, 4))
                plt.xlabel("epoch")
                plt.ylabel(m)
                plt.title(f"{prefix} - {m}")
                plt.grid(True, alpha=0.3)
                plt.text(0.5, 0.5, "no data recorded for this metric", ha="center", va="center", transform=plt.gca().transAxes)
                safe = f"{prefix.replace(' ', '_').replace('/', '_')}__{m}.png"
                plt.tight_layout()
                plt.savefig(os.path.join(base, safe))
                plt.close()
                continue
            plt.figure(figsize=(8, 4))
            for v in views_sorted:
                pts = sorted(by_view[v], key=lambda x: x[0])
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

def plot_category_dashboards(output_dir):
    mpath = os.path.join(output_dir, "metric.json")

    with open(mpath, "r", encoding="utf-8") as f:
        obj = json.load(f)

    prefixes = ["eval"]
    def build_cat_map(prefix):
        keys = set()
        for vals in obj.values():
            if isinstance(vals, dict):
                keys |= set(vals.keys())
        static_map = {
            "depth_error": ["depth_absrel", "depth_delta_125", "depth_rmse", "depth_log_rmse", "depth_si_rmse"],
            "pose": ["pose_rot_deg", "pose_trans_err", "pose_auc30"],
            "geometry": ["pts3d_acc", "pts3d_comp", "pts3d_nc", "chamfer_l1", "chamfer_l2", "Regr3DPose_pts3d", "Regr3DPose_ScaleInv_pts3d"],
            "track": ["track_conf_mean", "track_vis_ratio"],
            "loss": ["loss", "pose_loss"]
        }
        final_map = {}
        for cat, possible_keys in static_map.items():
            matched = []
            for pk in possible_keys:
                if pk in keys:
                    matched.append(pk)
                elif pk + "_avg" in keys:
                    matched.append(pk + "_avg")
                elif cat != "loss" and pk + "_med" in keys:
                    matched.append(pk + "_med")
            if cat == "depth_error":
                matched.extend([k for k in keys if k.startswith("depth_") and k not in matched])
            elif cat == "pose":
                matched = [k for k in matched if ("/" not in k and "loss" not in k)]
            elif cat == "geometry":
                matched.extend([k for k in keys if k.startswith("pts3d_") and k not in matched])
            elif cat == "track":
                matched.extend([k for k in keys if ("track" in k or "vis" in k) and k not in matched])
            elif cat == "loss":
                matched.extend([k for k in keys if (("loss" in k and not k.endswith("_med")) or k in ("total", "total_avg")) and k not in matched])
            final_map[cat] = sorted(list(set(matched)))
        return final_map
    outdir = os.path.join(output_dir, "visualize", "metrics_dashboards")
    os.makedirs(outdir, exist_ok=True)

    def collect_series(prefix, keys):
        series = {}
        for epoch_key, vals in obj.items():
            ep = int(epoch_key.replace("Epoch", ""))
            for k in keys:
                if k in vals and isinstance(vals[k], (int, float, np.number)):
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
        for j in range(len(series), len(axes)):
            axes[j].axis("off")
        fig.suptitle(f"{prefix} - {cname}")
        fig.tight_layout()
        safe = f"{prefix.replace(' ', '_').replace('/', '_')}__{cname}.png"
        fig.savefig(os.path.join(outdir, safe))
        plt.close(fig)
    for prefix in prefixes:
        cat_map = build_cat_map(prefix)
        for cname, keys in cat_map.items():
            plot_cat(prefix, cname, keys)
