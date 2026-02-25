from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import fnmatch

import sys
sys.path.insert(0, "../") 
from experiment_factors import Gestures, Velocity

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

@dataclass(frozen=True)
class ForceTrialResult:
    trial_index: int
    gesture: str
    velocity: str
    sample_id: int
    start_ts_us: int
    end_ts_us: int
    n_samples: int
    max_force: float
    min_force: float
    rep: int        

    # Raw Data               
    t_us: np.ndarray               
    force: np.ndarray              

class TrialBlockForceAnalyser:
   
    def __init__(
        self,
        data_dir: str | Path,
        folder_prefix: str = "Twintig Tap Pads",
        notification_filename: str = "Notification.csv",
        data_filename: str = "SerialAccessory.csv",
    ):
        self.data_dir = Path(data_dir)
        self.folder_prefix = folder_prefix
        self.notification_filename = notification_filename
        self.data_filename = data_filename

        self._folder: Optional[Path] = None
        self._trials: List[ForceTrialResult] = []
        self._summary: Optional[pd.DataFrame] = None

    def run(self) -> None:
        folder = self.__find_data_folder()
        notif = self.__load_notification(folder)
        serial = self.__load_serial(folder)
        print(f"[ForceAnalyser] Using data folder: {folder}")

        self._folder = folder
        self._trials = self.__segment_trials(notif, serial)
        self._summary = pd.DataFrame([t.__dict__ for t in self._trials])
        print(f"[ForceAnalyser] Trials found: {len(self._trials)}")
        print("[ForceAnalyser] Summary dataframe:")
        print(self._summary)

    def get_folder(self) -> Optional[Path]:
        return self._folder
    
    def get_trials(self) -> List[ForceTrialResult]:
        return self._trials

    def get_dataframe(self) -> pd.DataFrame:
        return self._summary if self._summary is not None else pd.DataFrame()

    def __find_data_folder(self) -> Path:
        # Same style as your VelocityCalibrationAnalyser
        return next(
            p for p in self.data_dir.iterdir()
            if p.is_dir() and p.name.startswith(self.folder_prefix)
        )
    
    def __load_notification(self, folder: Path) -> pd.DataFrame:
        df = pd.read_csv(folder / self.notification_filename)
        df["Timestamp (us)"] = df["Timestamp (us)"].astype(np.int64)
        df["String"] = df["String"].astype(str)
        return df

    def __load_serial(self, folder: Path) -> pd.DataFrame:
        path = folder / self.data_filename

        with open(path, "r") as f:
            _header = f.readline()
            first_data = f.readline().strip()

        n_fields = len(first_data.split(","))
        if n_fields < 2:
            raise ValueError(f"Unexpected SerialAccessory format: {path}")

        n_channels = n_fields - 1
        colnames = ["Timestamp (us)"] + [f"CH{i}" for i in range(n_channels)]

        df = pd.read_csv(
            path,
            skiprows=1,
            header=None,
            names=colnames,
            engine="python",
        )

        df["Timestamp (us)"] = df["Timestamp (us)"].astype(np.int64)
        for c in colnames[1:]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df
    
    def __is_trial_begin_marker(self, s: str) -> bool:
        return fnmatch.fnmatch(s.strip(), "TRIAL * BEGIN; gesture: *; sample_id: *; velocity: *")

    def __is_trial_end_marker(self, s: str) -> bool:
        return fnmatch.fnmatch(s.strip(), "TRIAL * END; gesture: *; sample_id: *; velocity: *")

    def __parse_marker(self, s: str) -> Tuple[int, str, int, str]:
        parts = [p.strip() for p in s.split(";") if p.strip()]
        # parts[0] = "TRIAL 73 BEGIN" or "TRIAL 73 END"
        trial_index = int(parts[0].split()[1])

        def get_kv(prefix: str) -> str:
            for p in parts[1:]:
                if p.lower().startswith(prefix.lower()):
                    return p.split(":", 1)[1].strip()
            raise ValueError(f"Missing field {prefix} in marker: {s}")

        gesture = get_kv("gesture")
        sample_id = int(get_kv("sample_id"))
        velocity = get_kv("velocity").lower()
        return trial_index, gesture, sample_id, velocity
    
    def __segment_trials(self, notification_df: pd.DataFrame, serial_df: pd.DataFrame) -> List[ForceTrialResult]:
        notification_df = notification_df.sort_values("Timestamp (us)").reset_index(drop=True)

        begins = notification_df[notification_df["String"].apply(self.__is_trial_begin_marker)].copy()
        ends = notification_df[notification_df["String"].apply(self.__is_trial_end_marker)].copy()
        ends = ends.sort_values("Timestamp (us)").reset_index(drop=True)

        begin_ts_by_trial: Dict[int, int] = {}
        for _, row in begins.iterrows():
            ts = int(row["Timestamp (us)"])
            try:
                trial_index, _, _, _ = self.__parse_marker(row["String"])
            except Exception:
                continue
            begin_ts_by_trial[trial_index] = ts

        timestamp = serial_df["Timestamp (us)"].to_numpy()
        available_channels = [c for c in serial_df.columns if c.startswith("CH")]

        results: List[ForceTrialResult] = []
        previous_end_timestamp = int(timestamp.min()) - 1

        rep_counter: Dict[Tuple[str, int, str], int] = {}

        for _, row in ends.iterrows():
            end_timestamp = int(row["Timestamp (us)"])
            trial_index, gesture, sample_id, velocity = self.__parse_marker(row["String"])

            start_timestamp = begin_ts_by_trial.get(trial_index, int(previous_end_timestamp + 1))
            if start_timestamp > end_timestamp:
                start_timestamp = int(previous_end_timestamp + 1)

            ch_name = f"CH{sample_id}"
            if ch_name not in serial_df.columns:
                raise ValueError(
                    f"sample_id={sample_id} refers to missing column {ch_name}. "
                    f"Available: {available_channels}"
                )

            mask = (timestamp >= start_timestamp) & (timestamp <= end_timestamp)
            seg_t = timestamp[mask]
            seg_force = serial_df.loc[mask, ch_name].to_numpy()

            rep_key = (gesture, sample_id, velocity)
            if rep_key not in rep_counter:
                rep_counter[rep_key] = 0
            rep = rep_counter[rep_key]
            rep_counter[rep_key] += 1

            results.append(
                ForceTrialResult(
                    trial_index=trial_index,
                    gesture=gesture,
                    velocity=velocity,
                    sample_id=sample_id,
                    start_ts_us=int(start_timestamp),
                    end_ts_us=int(end_timestamp),
                    n_samples=int(seg_force.size),
                    max_force=float(np.nanmax(seg_force)) if seg_force.size else float("nan"),
                    min_force=float(np.nanmin(seg_force)) if seg_force.size else float("nan"),
                    rep=rep,
                    t_us=np.asarray(seg_t, dtype=np.int64),
                    force=np.asarray(seg_force, dtype=float),
                )
            )
            previous_end_timestamp = end_timestamp

        return results

    # ---------- Plot ----------
    def plot_scatter_by_sample_id(self, title: str | None = None) -> None:
        df = self.get_dataframe()
        if df.empty:
            raise RuntimeError("No trials found (no matching TRIAL * END markers?)")

        sample_ids = sorted(df["sample_id"].dropna().unique().tolist())
        x_map = {sid: i for i, sid in enumerate(sample_ids)}
        df = df.copy()
        df["x"] = df["sample_id"].map(x_map).astype(float)

        fast = df[df["velocity"] == "velocity.fast"]
        slow = df[df["velocity"] == "velocity.slow"]

        rng = np.random.default_rng(0)
        jitter = 0.04

        slow_x = slow["x"] + rng.uniform(-jitter, jitter, size=len(slow))
        fast_x = fast["x"] + rng.uniform(-jitter, jitter, size=len(fast))

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.scatter(slow_x, slow["min_force"], alpha=0.6, label="slow")
        ax.scatter(fast_x, fast["min_force"], alpha=0.6, label="fast")

        ax.set_xticks(list(x_map.values()))
        ax.set_xticklabels([str(s) for s in sample_ids])
        ax.set_xlabel("sample_id")
        ax.set_ylabel("Min force (per trial)")
        ax.set_title(title or "Min force per trial by sample_id (fast vs slow)")
        ax.grid(axis="y", alpha=0.3)
        ax.legend()

        plt.tight_layout()
        plt.show()

    
    def plot_raw_trials_side_by_side(self) -> None:
        trials = self.get_trials()
        if not trials:
            raise RuntimeError("No trials loaded. Call run() first.")

        # Index trials by (gesture, sample_id, rep, velocity)
        by_key = {}
        for t in trials:
            vel = t.velocity.strip().lower()
            by_key[(t.gesture, t.sample_id, t.rep, vel)] = t

        # Build paired list (slow, fast) for same gesture/sample_id/rep
        pairs = []
        keys = sorted({(g, sid, rep) for (g, sid, rep, _vel) in by_key.keys()})
        for g, sid, rep in keys:
            slow = by_key.get((g, sid, rep, "velocity.slow"))
            fast = by_key.get((g, sid, rep, "velocity.fast"))
            # keep even if one is missing (still useful)
            pairs.append((g, sid, rep, slow, fast))

        if not pairs:
            raise RuntimeError("No trial pairs found.")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        axL, axR = axes
        idx = 0

        def _plot_one(ax, trial, title):
            ax.clear()
            ax.set_title(title)
            ax.set_xlabel("Time (ms)")
            ax.grid(True, alpha=0.3)

            if trial is None or trial.n_samples == 0:
                ax.text(0.5, 0.5, "Missing / empty", ha="center", va="center", transform=ax.transAxes)
                return

            t_ms = (trial.t_us - trial.t_us[0]) / 1000.0
            ax.plot(t_ms, trial.force)
            ax.set_ylabel("Force (raw)")

        def redraw():
            nonlocal idx
            g, sid, rep, slow, fast = pairs[idx]

            fig.suptitle(f"{idx+1}/{len(pairs)}  |  gesture={g}  sample_id={sid}  rep={rep}")

            _plot_one(axL, slow, "slow")
            _plot_one(axR, fast, "fast")

            # Optional: match y-lims for readability
            ys = []
            for t in (slow, fast):
                if t is not None and t.n_samples:
                    ys.append(np.nanmin(t.force))
                    ys.append(np.nanmax(t.force))
            if ys:
                lo, hi = float(np.min(ys)), float(np.max(ys))
                pad = 0.05 * (hi - lo) if hi > lo else 1.0
                axL.set_ylim(lo - pad, hi + pad)

            fig.canvas.draw_idle()

        def on_key(event):
            nonlocal idx
            if event.key in ("right", "n"):
                idx = (idx + 1) % len(pairs)
                redraw()
            elif event.key in ("left", "p"):
                idx = (idx - 1) % len(pairs)
                redraw()
            elif event.key in ("escape", "q"):
                plt.close(fig)

        fig.canvas.mpl_connect("key_press_event", on_key)
        redraw()
        plt.tight_layout()
        plt.show()