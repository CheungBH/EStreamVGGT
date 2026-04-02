import os
import json
import numpy as np
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
        if modality == "rgb_empty":
            return "RGB" if v == 0 else "white"
        return "RGB"
    wanted = ["auc30", "acc_mean", "acc_med", "comp_mean", "comp_med", "nc_mean", "nc_med", "depth_absrel", "depth_delta_125"]
    try:
        with open(mpath, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        obj = {}
    if not isinstance(obj, dict) or not obj:
        return
    prefixes = ["eval"]
    base = os.path.join(output_dir, "visualize", "metrics_views")
    os.makedirs(base, exist_ok=True)
    for prefix in prefixes:
        series = {}
        view_ids = set()
        for epoch_key, views in obj.items():
            if not (isinstance(epoch_key, str) and epoch_key.startswith("epoch")):
                continue
            if not isinstance(views, dict):
                continue
            for view_key in views.keys():
                if isinstance(view_key, str) and view_key.startswith("view"):
                    try:
                        view_ids.add(int(view_key.replace("view", "")))
                    except Exception:
                        pass
        if num_views and num_views > 0:
            view_range = range(1, num_views + 1)
        else:
            vmax = max(view_ids) if view_ids else 0
            view_range = range(1, vmax + 1)
        for epoch_key, views in obj.items():
            if not (isinstance(epoch_key, str) and epoch_key.startswith("epoch")):
                continue
            if not isinstance(views, dict):
                continue
            try:
                ep = int(epoch_key.replace("epoch", ""))
            except Exception:
                continue
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
                # still produce an empty figure with caption
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
    mpath = os.path.join(output_dir, "metric.json")
    if not os.path.exists(mpath):
        return
    try:
        with open(mpath, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        obj = {}
    if not isinstance(obj, dict) or not obj:
        return
    prefixes = ["eval"]
    def build_cat_map(prefix):
        keys = set()
        for vals in obj.values():
            if isinstance(vals, dict):
                keys |= set(vals.keys())
        static_map = {
            "depth_error": ["depth_absrel", "depth_delta_125", "depth_rmse", "depth_log_rmse", "depth_si_rmse"],
            "pose": ["pose_rot_deg", "pose_trans_err", "pose_auc30"],
            "geometry": ["pts3d_acc_mean", "pts3d_acc_med", "pts3d_comp_mean", "pts3d_comp_med", "pts3d_nc_mean", "pts3d_nc_med", "pts3d_chamfer_l1", "pts3d_chamfer_l2", "acc_mean", "acc_med", "comp_mean", "comp_med", "nc_mean", "nc_med", "chamfer_l1", "chamfer_l2"],
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
                elif pk + "_med" in keys:
                    matched.append(pk + "_med")
            if cat == "depth_error":
                matched.extend([k for k in keys if k.startswith("depth_") and k not in matched])
            elif cat == "pose":
                matched.extend([k for k in keys if (k.startswith("pose_") or k.startswith("Regr3DPose")) and k not in matched])
            elif cat == "geometry":
                matched.extend([k for k in keys if (k.startswith("pts3d_") or k.startswith("acc_") or k.startswith("comp_") or k.startswith("nc_") or k.startswith("chamfer_")) and k not in matched])
            elif cat == "track":
                matched.extend([k for k in keys if ("track" in k or "vis" in k) and k not in matched])
            elif cat == "loss":
                matched.extend([k for k in keys if ("loss" in k or k in ("total", "total_avg", "total_med")) and k not in matched])
            final_map[cat] = sorted(list(set(matched)))
        return final_map
    outdir = os.path.join(output_dir, "visualize", "metrics_dashboards")
    os.makedirs(outdir, exist_ok=True)
    def collect_series(prefix, keys):
        series = {}
        for epoch_key, vals in obj.items():
            if not (isinstance(epoch_key, str) and epoch_key.startswith("Epoch")):
                continue
            if not isinstance(vals, dict):
                continue
            try:
                ep = int(epoch_key.replace("Epoch", ""))
            except Exception:
                continue
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
