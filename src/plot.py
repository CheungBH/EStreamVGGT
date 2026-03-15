import os
import json
import numpy as np
import matplotlib.pyplot as plt

def plot_all_metrics(output_dir):
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
    def build_cat_map(prefix):
        keys = set()
        for obj in data:
            vals = obj.get(prefix, {})
            if isinstance(vals, dict):
                keys |= set(vals.keys())
        depth = sorted([k for k in keys if k.startswith("depth_")])
        pose = sorted([k for k in keys if k.startswith("pose_") or k.startswith("Regr3DPose")])
        geometry = sorted([k for k in keys if k.startswith("pts3d_")])
        confvis = sorted([k for k in keys if "conf" in k or "vis" in k])
        loss = sorted([k for k in keys if k.endswith("loss_avg") or k.endswith("loss_med") or k.startswith("loss_") or k in ("loss_avg","loss_med","pose_loss_avg","pose_loss_med")])
        # fallbacks to paper subset if empty
        if not depth:
            depth = [
                "depth_absrel_avg","depth_absrel_med",
                "depth_delta_125_avg","depth_delta_125_med",
                "depth_rmse_avg","depth_rmse_med",
                "depth_log_rmse_avg","depth_log_rmse_med",
                "depth_si_rmse_avg","depth_si_rmse_med",
            ]
        if not pose:
            pose = ["pose_rot_deg","pose_trans_err","pose_auc30"]
        if not geometry:
            geometry = [
                "pts3d_acc_mean","pts3d_acc_med",
                "pts3d_comp_mean","pts3d_comp_med",
                "pts3d_nc_mean","pts3d_nc_med",
                "pts3d_chamfer_l1","pts3d_chamfer_l2",
            ]
        if not confvis:
            confvis = ["conf_mean","track_conf_mean","track_vis_ratio"]
        if not loss:
            loss = ["loss_avg","loss_med","pose_loss_avg","pose_loss_med"]
        return {
            "depth_error": depth,
            "pose": pose,
            "geometry": geometry,
            "confidence_visibility": confvis,
            "loss": loss,
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
