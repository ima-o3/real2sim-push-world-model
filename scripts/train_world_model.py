"""Train a small action-conditioned box-motion predictor on tracked pilot videos.

Train on whole trials, hold out one entire trial. This is dynamics prediction,
NOT a learned robot controller, and the dataset is much too small to imply
robust generalisation.

Requires: numpy, pandas. No scikit-learn needed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


WINDOWS = {
    "P01_center": [(144, 200)],
    "P02_offset": [(132, 170), (229, 246), (293, 313)],
    "P03_long": [(180, 247)],
}

COLUMNS = ["box_x_mm", "box_y_mm", "pusher_x_mm", "pusher_y_mm"]
FEATURE_NAMES = [
    "pusher_step_mm", "relative_box_along_mm", "relative_box_across_mm",
    "previous_box_along_mm", "previous_box_across_mm", "first_step",
]


def read_window(directory: Path, trial: str, start: int, end: int) -> pd.DataFrame:
    source = directory / f"{trial}.csv"
    if not source.exists():
        raise FileNotFoundError(f"Missing {source}")
    all_rows = pd.read_csv(source)
    for column in ["frame", "time_s", *COLUMNS]:
        if column not in all_rows.columns:
            raise ValueError(f"{source} is missing column {column}")
    rows = all_rows.loc[all_rows["frame"].between(start, end)].copy().reset_index(drop=True)
    expected = np.arange(start, end + 1)
    if not np.array_equal(rows["frame"].to_numpy(), expected):
        raise ValueError(f"{source}: incomplete frame window {start}:{end}")
    if len(rows) < 4:
        raise ValueError(f"Window too short: {trial} {start}-{end}")
    return rows


def local_frame(delta_p: np.ndarray, previous_delta_p: np.ndarray | None = None):
    norm = float(np.linalg.norm(delta_p))
    direction = delta_p if norm > 0.05 else previous_delta_p
    if direction is None or np.linalg.norm(direction) < 0.05:
        direction = np.array([1.0, 0.0])
    u = direction / np.linalg.norm(direction)
    v = np.array([-u[1], u[0]])
    return u, v


def encode_step(box_pos, pusher_pos, dp, prior_box_delta, is_first, prior_dp=None):
    u, v = local_frame(dp, prior_dp)
    rel = box_pos - pusher_pos
    features = np.array([
        np.linalg.norm(dp),
        np.dot(rel, u), np.dot(rel, v),
        np.dot(prior_box_delta, u), np.dot(prior_box_delta, v),
        float(is_first),
    ], dtype=np.float64)
    return features, u, v


def samples_from_window(df: pd.DataFrame):
    """Only use steps with observed pusher and box at both ends.

    Avoid training on interpolated tool coordinates, which would create
    spurious dynamics labels in a tiny dataset.
    """
    b = df[["box_x_mm", "box_y_mm"]].to_numpy(dtype=float)
    p = df[["pusher_x_mm", "pusher_y_mm"]].to_numpy(dtype=float)
    features, targets, baseline_dp, baseline_db, baseline_prev = [], [], [], [], []
    for i in range(len(df) - 1):
        if not (np.isfinite(b[i]).all() and np.isfinite(b[i + 1]).all()
                and np.isfinite(p[i]).all() and np.isfinite(p[i + 1]).all()):
            continue
        prev_b = b[i] - b[i - 1] if i >= 1 and np.isfinite(b[i - 1]).all() else np.zeros(2)
        prev_p = p[i] - p[i - 1] if i >= 1 and np.isfinite(p[i - 1]).all() else None
        dp = p[i + 1] - p[i]
        x, u, v = encode_step(b[i], p[i], dp, prev_b, i == 0, prev_p)
        db = b[i + 1] - b[i]
        features.append(x)
        targets.append([np.dot(db, u), np.dot(db, v)])
        baseline_dp.append(dp)
        baseline_db.append(db)
        baseline_prev.append(prev_b)
    if not features:
        raise ValueError("No consecutive valid box/pusher steps found")
    return {
        "X": np.asarray(features), "y": np.asarray(targets),
        "dp": np.asarray(baseline_dp), "db": np.asarray(baseline_db),
        "prev": np.asarray(baseline_prev),
    }


class RidgeModel:
    def __init__(self, regularization: float = 10.0):
        self.regularization = regularization

    def fit(self, X, y):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std < 1e-8] = 1.0
        z = (X - self.mean) / self.std
        z = np.column_stack([np.ones(len(z)), z])
        penalty = np.eye(z.shape[1]) * self.regularization
        penalty[0, 0] = 0.0  # Do not regularise intercept.
        self.coef = np.linalg.solve(z.T @ z + penalty, z.T @ y)
        return self

    def predict(self, X):
        arr = np.atleast_2d(np.asarray(X, dtype=float))
        z = (arr - self.mean) / self.std
        z = np.column_stack([np.ones(len(z)), z])
        return z @ self.coef


def vector_error(pred, truth):
    err = np.linalg.norm(pred - truth, axis=1)
    return {"mean_euclidean_mm": float(err.mean()),
            "rmse_euclidean_mm": float(np.sqrt(np.mean(err ** 2)))}


def rollout(df: pd.DataFrame, model: RidgeModel, alpha: float):
    b = df[["box_x_mm", "box_y_mm"]].to_numpy(dtype=float)
    raw_p = df[["pusher_x_mm", "pusher_y_mm"]].to_numpy(dtype=float)
    # Allow only isolated single missing pusher frames; interpolate inside
    # a window or hold the closest observation at its edge. This remains
    # an approximate action measurement, and every imputation is reported.
    valid = np.isfinite(raw_p).all(axis=1)
    if np.any(~valid[1:] & ~valid[:-1]):
        raise ValueError("Hold-out has a pusher gap longer than one frame")
    p = df[["pusher_x_mm", "pusher_y_mm"]].interpolate(
        limit=1, limit_direction="both"
    ).to_numpy(dtype=float)
    if not np.isfinite(b).all() or not np.isfinite(p).all():
        raise ValueError("Missing box/pusher positions in hold-out event")

    pred = np.zeros_like(b)
    scaled_baseline = np.zeros_like(b)
    pred[0] = b[0]
    scaled_baseline[0] = b[0]
    last_db = np.zeros(2)
    for i in range(len(df) - 1):
        dp = p[i + 1] - p[i]
        prev_dp = p[i] - p[i - 1] if i else None
        x, u, v = encode_step(pred[i], p[i], dp, last_db, i == 0, prev_dp)
        y_pred = model.predict(x)[0]
        db = y_pred[0] * u + y_pred[1] * v
        pred[i + 1] = pred[i] + db
        last_db = db
        scaled_baseline[i + 1] = scaled_baseline[i] + alpha * dp

    errors = np.linalg.norm(pred - b, axis=1)
    end_error = float(np.linalg.norm(pred[-1] - b[-1]))
    base_end_error = float(np.linalg.norm(scaled_baseline[-1] - b[-1]))
    return {
        "true": b, "pusher": p, "pred": pred, "baseline": scaled_baseline,
        "summary": {
            "frames": int(len(df)),
            "imputed_single_pusher_frames": int((~valid).sum()),
            "rollout_path_mean_error_mm": float(errors.mean()),
            "rollout_final_vector_error_mm": end_error,
            "rollout_final_magnitude_error_mm": float(abs(
                np.linalg.norm(pred[-1]-pred[0]) - np.linalg.norm(b[-1]-b[0]))),
            "baseline_scaled_action_final_vector_error_mm": base_end_error,
            "real_box_displacement_mm": float(np.linalg.norm(b[-1]-b[0])),
            "predicted_box_displacement_mm": float(np.linalg.norm(pred[-1]-pred[0])),
            "scaled_action_baseline_displacement_mm": float(np.linalg.norm(
                scaled_baseline[-1]-scaled_baseline[0])),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/pilots"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/world_model"))
    parser.add_argument("--holdout", choices=list(WINDOWS), default="P03_long")
    parser.add_argument("--ridge", type=float, default=10.0)
    args = parser.parse_args()
    if args.ridge < 0:
        parser.error("--ridge must be non-negative")

    train_trials = [name for name in WINDOWS if name != args.holdout]
    train = []
    for trial in train_trials:
        for start, end in WINDOWS[trial]:
            piece = samples_from_window(read_window(args.data_dir, trial, start, end))
            train.append(piece)
            print(f"Training: {trial} frames {start}-{end}, {len(piece['X'])} valid steps")

    X = np.vstack([d["X"] for d in train])
    y = np.vstack([d["y"] for d in train])
    dp = np.vstack([d["dp"] for d in train])
    db = np.vstack([d["db"] for d in train])
    if len(X) < 20:
        raise ValueError("Insufficient training steps for this proof-of-concept")

    ridge = RidgeModel(args.ridge).fit(X, y)
    alpha = float(np.sum(dp * db) / max(np.sum(dp * dp), 1e-12))

    print(f"\nTRAINED ridge world model: {len(X)} steps from {train_trials}")
    print(f"Fitted displacement-per-pusher-motion baseline alpha: {alpha:.4f}")
    results = []
    args.results_dir.mkdir(parents=True, exist_ok=True)
    for start, end in WINDOWS[args.holdout]:
        df = read_window(args.data_dir, args.holdout, start, end)
        test = samples_from_window(df)
        y_pred = ridge.predict(test["X"])
        # Recover model prediction world coords using direction of each dp.
        # For zero displacement the correct basis is unknown; use the same
        # previous-pusher direction as the per-step feature encoder.
        p = df[["pusher_x_mm", "pusher_y_mm"]].to_numpy(dtype=float)
        b = df[["box_x_mm", "box_y_mm"]].to_numpy(dtype=float)
        true_world, pred_world, base_world, prev_world = [], [], [], []
        k=0
        for i in range(len(df)-1):
            if not (np.isfinite(b[i]).all() and np.isfinite(b[i+1]).all()
                    and np.isfinite(p[i]).all() and np.isfinite(p[i+1]).all()):
                continue
            local_dp = p[i+1]-p[i]
            prior_dp = p[i]-p[i-1] if i>0 and np.isfinite(p[i-1]).all() else None
            u, v = local_frame(local_dp, prior_dp)
            pred_world.append(y_pred[k,0]*u + y_pred[k,1]*v)
            true_world.append(b[i+1]-b[i])
            base_world.append(alpha*local_dp)
            prev_world.append(test["prev"][k])
            k+=1
        per_step_metrics={
            "learned":vector_error(np.array(pred_world),np.array(true_world)),
            "stationary":vector_error(np.zeros_like(true_world),np.array(true_world)),
            "scaled_pusher_motion":vector_error(np.array(base_world),np.array(true_world)),
            "constant_box_velocity":vector_error(np.array(prev_world),np.array(true_world)),
        }
        rolling = rollout(df,ridge,alpha)
        out_df = pd.DataFrame({
            "frame":df["frame"],"time_s":df["time_s"],
            "true_box_x_mm":rolling["true"][:,0],
            "true_box_y_mm":rolling["true"][:,1],
            "pred_box_x_mm":rolling["pred"][:,0],
            "pred_box_y_mm":rolling["pred"][:,1],
            "baseline_box_x_mm":rolling["baseline"][:,0],
            "baseline_box_y_mm":rolling["baseline"][:,1],
        })
        csv_path = args.results_dir/f"{args.holdout}_{start}_{end}_heldout_rollout.csv"
        out_df.to_csv(csv_path,index=False)
        # Optional plot: helpful for inspecting whether an apparently good
        # displacement magnitude hides a substantial trajectory error.
        figure_path = args.results_dir / f"{args.holdout}_{start}_{end}_rollout.png"
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.plot(rolling["true"][:,0], rolling["true"][:,1], label="Real box trajectory")
            ax.plot(rolling["pred"][:,0], rolling["pred"][:,1], label="Learned rollout")
            ax.plot(rolling["baseline"][:,0], rolling["baseline"][:,1], label="Scaled-action baseline")
            ax.scatter(*rolling["true"][0], marker="o", s=35, label="Start")
            ax.set_xlabel("Box X (mm, tracked camera axes)")
            ax.set_ylabel("Box Y (mm, tracked camera axes)")
            ax.set_title(f"Held-out {args.holdout}: frames {start}-{end}")
            ax.grid(alpha=0.25)
            ax.axis("equal")
            ax.legend()
            fig.tight_layout()
            fig.savefig(figure_path, dpi=160)
            plt.close(fig)
            print(f"Saved trajectory comparison: {figure_path}")
        except ImportError:
            print("matplotlib not installed; skipping optional plot")
        entry={
            "holdout":args.holdout,
            "start_frame":start,
            "end_frame":end,
            "valid_one_step_test_samples":len(test["X"]),
            "one_step":per_step_metrics,
            "rollout":rolling["summary"],
            "csv":str(csv_path),
            "plot":str(figure_path) if figure_path.exists() else None,
        }
        results.append(entry)
        print(f"\nHELD-OUT {args.holdout} frames {start}-{end}")
        for name, metrics in per_step_metrics.items():
            print(f"  one-step {name:22} RMSE {metrics['rmse_euclidean_mm']:.2f} mm")
        for name,v in rolling["summary"].items():
            print(f"  rollout {name}: {v:.2f}" if isinstance(v,float) else f"  rollout {name}: {v}")
        print(f"Saved rollout: {csv_path}")

    summary={
        "description":"Tiny real-video-trained state-action box dynamics predictor; not a robot policy",
        "feature_names":FEATURE_NAMES,
        "trained_trials":train_trials,
        "heldout_trial":args.holdout,
        "training_samples":len(X),
        "ridge_penalty":args.ridge,
        "calibration_note":"Approximate tape-based pixel-to-mm scale; P03 can reuse P02 with fixed camera",
        "tracking_note":"Red-pusher centroid is a proxy for actual contact tip",
        "validation_note":"Only three pilot videos; sequence autocorrelation; held-out trial, not proof of generalisation",
        "baseline_alpha":alpha,
        "trained_feature_mean":ridge.mean.tolist(),
        "trained_feature_std":ridge.std.tolist(),
        "coefficient_matrix_intercept_first":ridge.coef.tolist(),
        "results":results,
    }
    summary_path=args.results_dir/f"holdout_{args.holdout}_summary.json"
    summary_path.write_text(json.dumps(summary,indent=2))
    print(f"\nSaved summary: {summary_path}")

if __name__ == "__main__":
    main()
