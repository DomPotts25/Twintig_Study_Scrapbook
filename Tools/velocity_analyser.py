from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import fnmatch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

@dataclass(frozen=True)
class VelocityCalibrationTrialResult:
    trial_index: int
    gesture: str
    velocity: str
    start_ts_us: int
    end_ts_us: int
    n_samples: int
    max_force: float

class VelocityCalibrationAnalyser:

    def __init__(
            self,
            data_dir: str | Path,
            folder_prefix: str = "Twintig Tap Pads",
            notification_filename: str = "Notification.csv",
            data_filename: str = "SerialAccessory.csv",
            force_channel_index: int = 1,  # CH1 = first channel after timestamp
        ):
            self.data_dir = Path(data_dir)
            self.folder_prefix = folder_prefix
            self.notification_filename = notification_filename
            self.data_filename = data_filename
            self.force_channel_index = force_channel_index

            self._folder: Optional[Path] = None
            self._trials: List[VelocityCalibrationTrialResult] = []
            self._summary: Optional[pd.DataFrame] = None

    def run(self) -> None:
        folder = self.__find_data_folder()
        notif = self.__load_notification(folder)
        serial = self.__load_serial(folder)

        self._folder = folder
        self._trials = self.__segment_trials(notif, serial)
        self._summary = self.__compute_summary(self._trials)

    def get_trials(self) -> List[VelocityCalibrationTrialResult]:
        return self._trials

    def get_summary(self) -> pd.DataFrame:
        return self._summary if self._summary is not None else pd.DataFrame()

    def get_max_force_by_condition(self) -> Dict[Tuple[str, str], List[float]]:
        out: Dict[Tuple[str, str], List[float]] = {}
        for t in self._trials:
            out.setdefault((t.gesture, t.velocity), []).append(t.max_force)
        return out

    def get_folder(self) -> Optional[Path]:
        return self._folder


    def __find_data_folder(self) -> Path:
        return next(p for p in self.data_dir.iterdir() if p.is_dir() and p.name.startswith(self.folder_prefix))

    def __load_notification(self, folder: Path) -> pd.DataFrame:
        df = pd.read_csv(folder / self.notification_filename)
        df["Timestamp (us)"] = df["Timestamp (us)"].astype(np.int64)
        df["String"] = df["String"].astype(str)
        return df

    def __load_serial(self, folder: Path) -> pd.DataFrame:

        path = folder / self.data_filename
        colnames = ["Timestamp (us)"] + [f"CH{i}" for i in range(1, 9)]
        df = pd.read_csv(path, header=0, names=colnames, engine="python")

        df["Timestamp (us)"] = df["Timestamp (us)"].astype(np.int64)
        for c in colnames[1:]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    def __is_trial_end_marker(self, s: str) -> bool:
        return fnmatch.fnmatch(s.strip(), "TRIAL * END; gesture: *; velocity: *")

    def __parse_trial_end(self, s: str) -> Tuple[int, str, str]:
        # split into parts based on ;
        trial_part, gesture_part, velocity_part = [p.strip() for p in s.split(";")[:3]]

        # trial_index: "TRIAL X END"
        # gesture_part: "gesture: G"
        # velocity_part: "velocity: V"
        trial_index = int(trial_part.split()[1])
        gesture = gesture_part.split(":", 1)[1].strip()
        velocity = velocity_part.split(":", 1)[1].strip()
        return trial_index, gesture, velocity

    def __segment_trials(self, notification_df: pd.DataFrame, serial_df: pd.DataFrame) -> List[VelocityCalibrationTrialResult]:
        markers = notification_df[notification_df["String"].apply(self.__is_trial_end_marker)].copy()
        markers = markers.sort_values("Timestamp (us)").reset_index(drop=True)

        timestamp = serial_df["Timestamp (us)"].to_numpy()
        force = serial_df[f"CH{self.force_channel_index}"].to_numpy()

        results: List[VelocityCalibrationTrialResult] = []
        previous_end_timestamp = int(timestamp.min()) - 1

        for _, row in markers.iterrows():
            end_timestamp = int(row["Timestamp (us)"])
            trial_index, gesture, velocity = self.__parse_trial_end(row["String"])

            mask = (timestamp > previous_end_timestamp) & (timestamp <= end_timestamp)
            seg_force = force[mask]

            results.append(
                VelocityCalibrationTrialResult(
                    trial_index=trial_index,
                    gesture=gesture,
                    velocity=velocity,
                    start_ts_us=int(previous_end_timestamp + 1),
                    end_ts_us=end_timestamp,
                    n_samples=int(seg_force.size),
                    max_force=float(np.nanmax(seg_force)) if seg_force.size else float("nan"),
                )
            )
            previous_end_timestamp = end_timestamp

        return results

    def __compute_summary(self, trials: List[VelocityCalibrationTrialResult]) -> pd.DataFrame:
        df = pd.DataFrame([t.__dict__ for t in trials])
        if df.empty:
            return df
        return (
            df.groupby(["gesture", "velocity"])["max_force"]
              .agg(count="count", mean="mean", std="std", min="min", max="max")
              .reset_index()
        )
    
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
            seg_force = serial_df.loc[mask, ch_name].to_numpy()

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

        ax.scatter(slow_x, slow["max_force"], alpha=0.6, label="slow")
        ax.scatter(fast_x, fast["max_force"], alpha=0.6, label="fast")

        ax.set_xticks(list(x_map.values()))
        ax.set_xticklabels([str(s) for s in sample_ids])
        ax.set_xlabel("sample_id")
        ax.set_ylabel("Max force (per trial)")
        ax.set_title(title or "Max force per trial by sample_id (fast vs slow)")
        ax.grid(axis="y", alpha=0.3)
        ax.legend()

        plt.tight_layout()
        plt.show()

def main():
    analyser = TrialBlockForceAnalyser(
        r"C:\Users\dm-potts-admin\Documents\Postdoc\UWE\Outside_Interactions\Object_Characterisation_Study\Study_Program\Twintig_Study_Scrapbook\Logged_Data\Trial_Force_Data\Tap_Pads_Data"
    )   
    analyser.run()
    analyser.plot_scatter_by_sample_id("Last block: max force per trial (fast vs slow)")
    # analyser = VelocityCalibrationAnalyser(
    #     data_dir=r"C:\Users\dm-potts-admin\Documents\Postdoc\UWE\Outside_Interactions\Object_Characterisation_Study\Study_Program\Twintig_Study_Scrapbook\Logged_Data\Velocity_Calibration_Force_Data\velocityCal")

    # analyser.run()

    # summary = analyser.get_summary()
    # print(summary)

    # data_by_condition = analyser.get_max_force_by_condition()
    # # print(data_by_condition[("tap", "slow")])  # e.g. [0.64, 0.54, ...]
    # # print(np.mean(data_by_condition[("tap", "slow")]))

    # # print(data_by_condition[("tap", "fast")])  # e.g. [0.64, 0.54, ...]
    # # print(np.mean(data_by_condition[("tap", "fast")]))

    # print(data_by_condition[("pat", "fast")])
    # print("-----------------")
    # print(data_by_condition[("pat", "slow")])

    # labels = []
    # means = []
    # stds = []

    # for (gesture, velocity), values in data_by_condition.items():
    #     values = np.asarray(values, dtype=float)

    #     labels.append(f"{gesture} | {velocity}")
    #     means.append(np.nanmean(values))
    #     stds.append(np.nanstd(values))

    # means = np.array(means)
    # stds = np.array(stds)

    # # Plot
    # fig, ax = plt.subplots(figsize=(10, 5))

    # x = np.arange(len(labels))
    # ax.bar(x, means, yerr=stds, capsize=5)

    # ax.set_xticks(x)
    # ax.set_xticklabels(labels, rotation=45, ha="right")
    # ax.set_ylabel("Max Force")
    # ax.set_title("Mean ± SD of Max Force by Condition")

    # ax.grid(axis="y", alpha=0.3)

    # plt.tight_layout()
    # plt.show()

if __name__ == "__main__":
    main()