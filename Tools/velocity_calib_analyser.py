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

# FORCE READINGS DATA CLASSES FOR VELOCITY CONTROL
@dataclass
class MetricStats:
    mean: float
    sd: float

@dataclass
class ConditionMetrics:
    max_force: MetricStats
    min_force: MetricStats
    rise_time_us: MetricStats
    contact_duration_us: MetricStats
    peak_slope: MetricStats

@dataclass
class VelocityCondition:
    n_trials: int
    metrics: ConditionMetrics
    trials: List[dict] = field(default_factory=list)

@dataclass
class VelocityCalibrationForceMetrics:
    conditions: Dict[Gestures, Dict[Velocity, VelocityCondition]]
###############################################################



@dataclass(frozen=True)
class VelocityCalibrationTrialResult:
    trial_index: int
    gesture: str
    velocity: str
    start_ts_us: int
    end_ts_us: int
    n_samples: int
    max_force: float
    min_force: float
    rise_time_us: Optional[float] = None
    contact_duration_us: Optional[float] = None
    peak_slope: Optional[float] = None

class VelocityCalibrationAnalyser:

    def __init__(
            self,
            data_dir: str | Path,
            folder_prefix: str = "Twintig Tap Pads",
            notification_filename: str = "Notification.csv",
            data_filename: str = "SerialAccessory.csv",
            force_channel_index: int = 2,  # CH1 = first channel after timestamp
            begin_detection_enabled: bool = True,
            begin_highpass_threshold: float = -0.02,
            begin_holdoff_ms: float = 250.0,):

            self.data_dir = Path(data_dir)
            self.folder_prefix = folder_prefix
            self.notification_filename = notification_filename
            self.data_filename = data_filename
            self.force_channel_index = force_channel_index

            self.begin_detection_enabled = begin_detection_enabled
            self.begin_highpass_threshold = begin_highpass_threshold
            self.begin_holdoff_ms = begin_holdoff_ms

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


    # need to change experiment controller to not use this, but 'get_metrics_by_condition' instead
    def get_max_force_by_condition(self) -> Dict[Tuple[str, str], List[float]]:
        out: Dict[Tuple[str, str], List[float]] = {}
        for t in self._trials:
            out.setdefault((t.gesture, t.velocity), []).append(t.max_force)
        return out
    
    def get_metrics_by_condition(self) -> Dict[Tuple[str, str], List[dict]]:
        result: Dict[Tuple[str, str], List[dict]] = {}
        for t in self._trials:
            result.setdefault((t.gesture, t.velocity), []).append({
                "trial_index": t.trial_index,
                "min_force": t.min_force,
                "max_force": t.max_force,
                "rise_time_us": t.rise_time_us,
                "contact_duration_us": t.contact_duration_us,
                "peak_slope": t.peak_slope,
            })
        return result

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

        timestamp_us = serial_df["Timestamp (us)"].to_numpy()
        force = serial_df[f"CH{self.force_channel_index}"].to_numpy()

        results: List[VelocityCalibrationTrialResult] = []
        previous_end_timestamp = int(timestamp_us.min()) - 1

        for _, row in markers.iterrows():
            end_timestamp = int(row["Timestamp (us)"])
            trial_index, gesture, velocity = self.__parse_trial_end(row["String"])

            mask = (timestamp_us > previous_end_timestamp) & (timestamp_us <= end_timestamp)
            seg_ts_us = timestamp_us[mask]
            seg_force = force[mask]
            
            if self.begin_detection_enabled and seg_force.size:
                begin_index = self.__detect_begin_index_highpass(seg_ts_us, seg_force)
                if begin_index is not None:
                    seg_ts_us = seg_ts_us[begin_index:]
                    seg_force = seg_force[begin_index:]

            # Now define trial-relative time from the (possibly trimmed) begin
            if seg_ts_us.size == 0:
                previous_end_timestamp = end_timestamp
                continue

            relative_time_us = seg_ts_us - seg_ts_us[0]

            max_force = float(np.nanmax(seg_force)) if seg_force.size else float("nan")
            min_force = float(np.nanmin(seg_force)) if seg_force.size else float("nan")

            rise_time_us = self.__compute_rise_time_inverted(relative_time_us, seg_force) if seg_force.size else None
            contact_duration_us = self.__compute_contact_duration_inverted(relative_time_us, seg_force) if seg_force.size else None
            peak_slope = self.__compute_peak_slope_inverted(relative_time_us, seg_force) if seg_force.size else None            

            results.append(
                VelocityCalibrationTrialResult(
                    trial_index=trial_index,
                    gesture=gesture,
                    velocity=velocity,
                    #start_ts_us=int(previous_end_timestamp + 1),
                    start_ts_us=int(seg_ts_us[0]),
                    end_ts_us=end_timestamp,
                    n_samples=int(seg_force.size),
                    max_force=max_force,
                    min_force=min_force,
                    rise_time_us=rise_time_us,
                    contact_duration_us=contact_duration_us,
                    peak_slope=peak_slope,
                )
            )
            previous_end_timestamp = end_timestamp

        return results

    def __compute_summary(
        self,
        trials: List[VelocityCalibrationTrialResult]
    ) -> pd.DataFrame:

        df = pd.DataFrame([t.__dict__ for t in trials])

        if df.empty:
            return df

        # Ensure numeric columns (important if some values are None)
        numeric_columns = [
            "max_force",
            "min_force",
            "rise_time_us", 
            "contact_duration_us",
            "peak_slope",
        ]

        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # --- GROUP AND AGGREGATE ---
        grouped = df.groupby(["gesture", "velocity"])

        summary = grouped.agg(
            trial_count=("trial_index", "count"),

            # Force metrics
            max_force_mean=("max_force", "mean"),
            max_force_std=("max_force", "std"),
            min_force_mean=("min_force", "mean"),
            min_force_std=("min_force", "std"),

            # Rise time (microseconds)
            rise_time_mean_us=("rise_time_us", "mean"),
            rise_time_std_us=("rise_time_us", "std"),
            rise_time_median_us=("rise_time_us", "median"),
            rise_time_min_us=("rise_time_us", "min"),
            rise_time_max_us=("rise_time_us", "max"),

            # Contact duration (microseconds)
            contact_duration_mean_us=("contact_duration_us", "mean"),
            contact_duration_std_us=("contact_duration_us", "std"),
            contact_duration_median_us=("contact_duration_us", "median"),
            contact_duration_min_us=("contact_duration_us", "min"),
            contact_duration_max_us=("contact_duration_us", "max"),

            # Peak slope (force per microsecond)
            peak_slope_mean=("peak_slope", "mean"),
            peak_slope_std=("peak_slope", "std"),
            peak_slope_median=("peak_slope", "median"),
            peak_slope_min=("peak_slope", "min"),
            peak_slope_max=("peak_slope", "max"),
        ).reset_index()

        return summary
    
    def __detect_begin_index_highpass(
        self,
        timestamps_us: np.ndarray,
        force_values: np.ndarray,) -> Optional[int]:

        if timestamps_us.size < 3 or force_values.size < 3:
            return None

        # Estimate sampling interval to convert holdoff ms -> samples
        timestamp_deltas_us = np.diff(timestamps_us)
        valid_deltas = timestamp_deltas_us[timestamp_deltas_us > 0]
        if valid_deltas.size == 0:
            return None

        estimated_sample_period_us = float(np.median(valid_deltas))
        holdoff_us = float(self.begin_holdoff_ms) * 1000.0
        holdoff_samples = int(max(1, round(holdoff_us / estimated_sample_period_us)))

        # High-pass proxy: first difference
        high_pass_proxy = np.diff(force_values, prepend=force_values[0])

        trigger_indices = np.where(high_pass_proxy <= float(self.begin_highpass_threshold))[0]
        if trigger_indices.size == 0:
            return None

        # First trigger index is our begin
        begin_index = int(trigger_indices[0])

        # (Optional) clamp begin index so we still have baseline samples after trimming
        # e.g., ensure at least 10 samples exist
        if (force_values.size - begin_index) < 10:
            return None

        return begin_index
        pass
    
    def __baseline(self, force_values: np.ndarray) -> Optional[float]:
        if force_values.size < 10:
            return None
        
        first_10_percent = max(5, int(0.10 * force_values.size))  # first 10% as baseline
        baseline = float(np.nanmedian(force_values[:first_10_percent]))

        return baseline if np.isfinite(baseline) else None
    
    # expects baseline to be high, and press to be lower. 
    def __compute_rise_time_inverted(self, timestamps_us_relative_to_trial_start: np.ndarray, force_values_raw: np.ndarray) -> Optional[float]:
        if timestamps_us_relative_to_trial_start.size < 10 or force_values_raw.size < 10:
            return None

        number_of_baseline_samples = min(
            force_values_raw.size,
            max(5, int(0.10 * force_values_raw.size))
        )
        baseline_force_estimate = float(np.nanmedian(force_values_raw[:number_of_baseline_samples]))
        trough_force_value = float(np.nanmin(force_values_raw))

        if not (np.isfinite(baseline_force_estimate) and np.isfinite(trough_force_value)):
            return None

        press_amplitude = baseline_force_estimate - trough_force_value
        if press_amplitude <= 1e-6:
            return None

        normalized_press_depth = (baseline_force_estimate - force_values_raw) / press_amplitude

        def interpolated_crossing_time_us(threshold_depth: float) -> Optional[float]:
            indices_above = np.where(normalized_press_depth >= threshold_depth)[0]
            if indices_above.size == 0:
                return None

            crossing_index = int(indices_above[0])
            if crossing_index == 0:
                return float(timestamps_us_relative_to_trial_start[0])

            previous_index = crossing_index - 1

            y0 = float(normalized_press_depth[previous_index])
            y1 = float(normalized_press_depth[crossing_index])
            t0 = float(timestamps_us_relative_to_trial_start[previous_index])
            t1 = float(timestamps_us_relative_to_trial_start[crossing_index])

            if not (np.isfinite(y0) and np.isfinite(y1) and np.isfinite(t0) and np.isfinite(t1)):
                return None
            if y1 == y0:
                return t1

            fraction = (threshold_depth - y0) / (y1 - y0)
            return float(t0 + fraction * (t1 - t0))

        time_at_10_percent_us = interpolated_crossing_time_us(0.10)
        time_at_90_percent_us = interpolated_crossing_time_us(0.90)

        if time_at_10_percent_us is None or time_at_90_percent_us is None:
            return None
        if time_at_90_percent_us <= time_at_10_percent_us:
            return None

        return float(time_at_90_percent_us - time_at_10_percent_us)
    

    def __compute_contact_duration_inverted(self, timestamps: np.ndarray, force_values_raw: np.ndarray, press_depth_threshold: float = 0.05) -> Optional[float]:
        if timestamps.size < 10 or force_values_raw.size < 10:
            return None
        
        baseline_force_value = self.__baseline(force_values_raw)
        trough_force_value = float(np.nanmin(force_values_raw))

        press_amplitude = baseline_force_value - trough_force_value

        normalized_press_depth = (baseline_force_value - force_values_raw) / press_amplitude
        index_contact_active = np.where(normalized_press_depth >= press_depth_threshold)[0]

        if(index_contact_active.size < 2):
            return None
        
        first_contact_index = int(index_contact_active[0])
        last_contact_index = int(index_contact_active[-1])

        contact_duration_seconds = (timestamps[last_contact_index] - timestamps[first_contact_index])
        
        return float(contact_duration_seconds)

    def __compute_peak_slope_inverted(self, timestamps: np.ndarray, force_values_raw: np.ndarray) -> Optional[float]:
        if timestamps.size < 5 or force_values_raw.size < 5:
            return None
        
        delta_time = np.diff(timestamps)
        delta_force = np.diff(force_values_raw)

        valid_time_steps_mask = delta_time > 0

        if not np.any(valid_time_steps_mask):
            return None
        
        slopes_force_per_us = (delta_force[valid_time_steps_mask] / delta_time[valid_time_steps_mask])

        if slopes_force_per_us.size == 0:
            return None
        
        most_negative_slope = float(np.nanmin(slopes_force_per_us))

        peak_downward_slope_magnitude = abs(most_negative_slope)
        
        return float(peak_downward_slope_magnitude)
    
    def _plot_summary_metric(
        self,
        mean_column: str,
        std_column: str,
        y_label: str,
        title: str,
        velocity_order=("slow", "normal", "fast"), # added NORMAL speed
    ):
        if self._summary is None or self._summary.empty:
            raise ValueError("Summary data not available. Run analyser first.")

        df = self._summary.copy()

        gestures = sorted(df["gesture"].unique())
        x_positions = np.arange(len(gestures), dtype=float)
        bar_width = 0.35

        fig, ax = plt.subplots()

        for i, velocity in enumerate(velocity_order):
            df_vel = df[df["velocity"] == velocity].set_index("gesture")

            means = np.array(
                [df_vel.loc[g, mean_column] if g in df_vel.index else np.nan for g in gestures]
            )
            stds = np.array(
                [df_vel.loc[g, std_column] if g in df_vel.index else np.nan for g in gestures]
            )

            offset = (i - 0.5) * bar_width
            ax.bar(
                x_positions + offset,
                means,
                width=bar_width,
                yerr=stds,
                capsize=4,
                label=str(velocity),
            )

        ax.set_xticks(x_positions)
        ax.set_xticklabels(gestures, rotation=30, ha="right")
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()

        return fig, ax

    def plot_rise_time_summary(self):
        return self._plot_summary_metric(
            mean_column="rise_time_mean_us",
            std_column="rise_time_std_us",
            y_label="Rise time (µs)",
            title="Rise Time by Gesture and Velocity",
        )

    def plot_contact_duration_summary(self):
        return self._plot_summary_metric(
            mean_column="contact_duration_mean_us",
            std_column="contact_duration_std_us",
            y_label="Contact duration (µs)",
            title="Contact Duration by Gesture and Velocity",
        )

    def plot_peak_slope_summary(self):
        return self._plot_summary_metric(
            mean_column="peak_slope_mean",
            std_column="peak_slope_std",
            y_label="Peak slope (force / µs)",
            title="Peak Slope by Gesture and Velocity",
        )
    
    def plot_metric_trials_scatter(
        self,
        metric: str,
        y_label: str | None = None,
        title: str | None = None,
        velocity_order: Tuple[str, ...] = ("slow", "normal", "fast"),
        jitter: float = 0.05,
        alpha: float = 0.7,
        figsize: Tuple[float, float] = (12, 6),
        show_legend: bool = True,
        seed: int = 0,
        annotate: bool = True,
        text_offset: Tuple[float, float] = (0.01, 0.0),
        fontsize: int = 8,
    ):

        trials = self.get_trials()
        if not trials:
            raise ValueError("No trials available. Run analyser first.")

        allowed = {
            "min_force",
            "max_force",
            "rise_time_us",
            "contact_duration_us",
            "peak_slope",
        }
        if metric not in allowed:
            raise ValueError(f"Unknown metric '{metric}'. Allowed: {sorted(allowed)}")

        df = pd.DataFrame([t.__dict__ for t in trials])
        if df.empty:
            raise ValueError("No trial data available.")

        df[metric] = pd.to_numeric(df[metric], errors="coerce")
        df = df[np.isfinite(df[metric].to_numpy())].copy()
        if df.empty:
            raise ValueError(f"Metric '{metric}' has no finite values to plot.")

        gestures = sorted(df["gesture"].dropna().unique().tolist())
        x_positions = np.arange(len(gestures), dtype=float)
        x_map = {g: x for g, x in zip(gestures, x_positions)}

        vel_count = max(1, len(velocity_order))
        vel_spacing = 0.30 / max(1, vel_count)
        vel_offsets = {
            v: (i - (vel_count - 1) / 2.0) * vel_spacing
            for i, v in enumerate(velocity_order)
        }

        rng = np.random.default_rng(seed)

        fig, ax = plt.subplots(figsize=figsize)

        for velocity in velocity_order:
            df_vel = df[df["velocity"] == velocity]
            if df_vel.empty:
                continue

            base_x = df_vel["gesture"].map(x_map).astype(float).to_numpy()
            offset = float(vel_offsets.get(velocity, 0.0))
            x_vals = base_x + offset + rng.uniform(-jitter, jitter, size=base_x.size)
            y_vals = df_vel[metric].to_numpy()

            ax.scatter(x_vals, y_vals, alpha=alpha, label=str(velocity))

            # -------- Add annotations --------
            if annotate:
                for x, y, (_, row) in zip(x_vals, y_vals, df_vel.iterrows()):
                    label = f"{row['trial_index']} | {row['gesture']} | {row['velocity']}"
                    ax.text(
                        x + text_offset[0],
                        y + text_offset[1],
                        label,
                        fontsize=fontsize,
                        alpha=0.75,
                    )

        ax.set_xticks(x_positions)
        ax.set_xticklabels(gestures, rotation=30, ha="right")
        ax.set_ylabel(y_label or metric)
        ax.set_xlabel("Gesture")
        ax.set_title(title or f"{metric} per trial by gesture and velocity")
        ax.grid(True, axis="y", alpha=0.3)

        if show_legend:
            ax.legend(title="Velocity")

        fig.tight_layout()
        return fig, ax
    
    def plot_rise_time_trials(self, **kwargs):
        return self.plot_metric_trials_scatter(
            metric="rise_time_us",
            y_label="Rise time (µs)",
            title="Rise Time per Trial by Gesture and Velocity",
            **kwargs,
        )

    def plot_contact_duration_trials(self, **kwargs):
        return self.plot_metric_trials_scatter(
            metric="contact_duration_us",
            y_label="Contact duration (µs)",
            title="Contact Duration per Trial by Gesture and Velocity",
            **kwargs,
        )

    def plot_peak_slope_trials(self, **kwargs):
        return self.plot_metric_trials_scatter(
            metric="peak_slope",
            y_label="Peak slope (force / µs)",
            title="Peak Slope per Trial by Gesture and Velocity",
            **kwargs,
        )

    def plot_max_force_trials(self, **kwargs):
        return self.plot_metric_trials_scatter(
            metric="max_force",
            y_label="Max force",
            title="Max Force per Trial by Gesture and Velocity",
            **kwargs,
        )

    def plot_min_force_trials(self, **kwargs):
        return self.plot_metric_trials_scatter(
            metric="min_force",
            y_label="Min force",
            title="Min Force per Trial by Gesture and Velocity",
            **kwargs,
        )

############################################################################    

def main():
    # analyser = TrialBlockForceAnalyser(
    #     r"C:\Users\dm-potts-admin\Documents\Postdoc\UWE\Outside_Interactions\Object_Characterisation_Study\Study_Program\Twintig_Study_Scrapbook\Logged_Data\Trial_Force_Data\Tap_Pads_Data"
    # )   
    # analyser.run()
    # analyser.plot_scatter_by_sample_id("Last block: min force per trial (fast vs slow)")
    # analyser.plot_raw_trials_side_by_side()

    calAnalyser = VelocityCalibrationAnalyser(r"C:\Users\dm-potts-admin\Documents\Postdoc\UWE\Outside_Interactions\Object_Characterisation_Study\Study_Program\Twintig_Study_Scrapbook\Logged_Data\Velocity_Calibration_Force_Data\velocityCal")
    calAnalyser.run()
    # df = calAnalyser.get_summary()
    # print(df)

    # change these plots to scatters for each trial.
    calAnalyser.plot_peak_slope_summary()
    calAnalyser.plot_contact_duration_summary()
    calAnalyser.plot_rise_time_summary()

    # print("rise time mean")
    # print(df["rise_time_mean_us"])
    # print("rise time sd")
    # print(df["rise_time_std_us"])
    # print(df[(df.gesture=="pat") & (df.velocity=="fast")]["rise_time_mean_us"].unique())
    calAnalyser.plot_peak_slope_trials(fontsize=6, text_offset=(0.005, 0))
    calAnalyser.plot_contact_duration_trials(fontsize=6, text_offset=(0.005, 0))
    calAnalyser.plot_rise_time_trials(fontsize=6, text_offset=(0.005, 0))

    plt.show()

if __name__ == "__main__":
    main()