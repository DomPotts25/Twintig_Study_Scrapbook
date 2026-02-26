from __future__ import annotations

import csv
import fnmatch
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "../")
from experiment_factors import Gestures, Velocity


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
        begin_holdoff_ms: float = 250.0,
    ):
        self.data_dir = Path(data_dir)
        self.folder_prefix = folder_prefix
        self.notification_filename = notification_filename
        self.data_filename = data_filename
        self.force_channel_index = int(force_channel_index)

        self.begin_detection_enabled = bool(begin_detection_enabled)
        self.begin_highpass_threshold = float(begin_highpass_threshold)
        self.begin_holdoff_ms = float(begin_holdoff_ms)

        self._folder: Optional[Path] = None
        self._trials: List[dict] = []
        self._metrics: Optional[VelocityCalibrationForceMetrics] = None

    # -------------------- Public API --------------------

    def run(self) -> None:
        folder = self.__find_data_folder()
        markers = self.__load_notification_markers(folder)
        timestamps_us, force = self.__load_serial_force(folder)

        self._folder = folder
        self._trials = self.__segment_trials(markers, timestamps_us, force)
        self._metrics = self.__compute_force_metrics(self._trials)

    def get_folder(self) -> Optional[Path]:
        return self._folder

    def get_trials(self) -> List[dict]:
        return self._trials

    def get_force_metrics(self) -> Optional[VelocityCalibrationForceMetrics]:
        return self._metrics

    def get_calibration(self) -> VelocityCalibrationForceMetrics:
        """Convenience: returns force metrics, raising if run() hasn't been called."""
        if self._metrics is None:
            raise ValueError("No calibration metrics available. Call run() first.")
        return self._metrics

    def get_metrics_by_condition(self) -> Dict[Tuple[Gestures, Velocity], List[dict]]:
        """Raw trial dicts grouped by (gesture, velocity). Includes timestamps_us + force arrays."""
        result: Dict[Tuple[Gestures, Velocity], List[dict]] = {}
        for t in self._trials:
            result.setdefault((t["gesture"], t["velocity"]), []).append(t)
        return result

    # -------------------- CSV Loading (no pandas) --------------------

    def __find_data_folder(self) -> Path:
        return next(
            p
            for p in self.data_dir.iterdir()
            if p.is_dir() and p.name.startswith(self.folder_prefix)
        )

    def __load_notification_markers(self, folder: Path) -> List[Tuple[int, str]]:
        """Return sorted list of (timestamp_us, string) for TRIAL END markers."""
        path = folder / self.notification_filename
        markers: List[Tuple[int, str]] = []

        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            # Expect columns: "Timestamp (us)", "String" (case-sensitive in your export)
            for row in reader:
                try:
                    ts = int(row.get("Timestamp (us)", ""))
                except Exception:
                    continue
                s = str(row.get("String", ""))
                if self.__is_trial_end_marker(s):
                    markers.append((ts, s))

        markers.sort(key=lambda x: x[0])
        return markers

    def __load_serial_force(self, folder: Path) -> Tuple[np.ndarray, np.ndarray]:
        path = folder / self.data_filename
        timestamps: List[int] = []
        forces: List[float] = []

        # force_channel_index is 1..8 for CH1..CH8
        ch = int(self.force_channel_index)
        if ch < 1 or ch > 8:
            raise ValueError(f"force_channel_index must be 1..8, got {self.force_channel_index}")

        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header line

            for row in reader:
                # Expect at least 1 (timestamp) + 8 channels = 9 columns
                if not row or len(row) < 9:
                    continue

                # timestamp
                try:
                    ts = int(float(row[0].strip()))
                except Exception:
                    continue

                # channel value at position ch (CH1 is row[1], CH8 is row[8])
                raw = row[ch].strip() if row[ch] is not None else ""
                raw = raw.replace(",", ".")  # harmless if '.' already used

                try:
                    val = float(raw) if raw != "" else float("nan")
                except Exception:
                    val = float("nan")

                timestamps.append(ts)
                forces.append(val)

        return np.asarray(timestamps, dtype=np.int64), np.asarray(forces, dtype=float)

    # -------------------- Trial Segmentation --------------------

    def __is_trial_end_marker(self, s: str) -> bool:
        return fnmatch.fnmatch(s.strip(), "TRIAL * END; gesture: *; velocity: *")

    def __parse_trial_end(self, s: str) -> Tuple[int, Gestures, Velocity]:
        # split into parts based on ;
        trial_part, gesture_part, velocity_part = [p.strip() for p in s.split(";")[:3]]

        # trial_index: "TRIAL X END"
        trial_index = int(trial_part.split()[1])

        gesture_str = gesture_part.split(":", 1)[1].strip()
        velocity_str = velocity_part.split(":", 1)[1].strip()

        # Prefer Enum(value) (matches "tap"/"slow"), fall back to Enum["NAME"] (matches "TAP"/"SLOW")
        try:
            gesture = Gestures(gesture_str)
        except Exception:
            gesture = Gestures[gesture_str.upper()]

        try:
            velocity = Velocity(velocity_str)
        except Exception:
            velocity = Velocity[velocity_str.upper()]

        return trial_index, gesture, velocity

    def __segment_trials(
        self,
        markers: List[Tuple[int, str]],
        timestamp_us: np.ndarray,
        force: np.ndarray,
    ) -> List[dict]:
        if timestamp_us.size == 0:
            return []

        results: List[dict] = []
        previous_end_timestamp = int(np.min(timestamp_us)) - 1

        for end_timestamp, marker_str in markers:
            trial_index, gesture, velocity = self.__parse_trial_end(marker_str)

            mask = (timestamp_us > previous_end_timestamp) & (
                timestamp_us <= int(end_timestamp)
            )
            seg_ts_us = timestamp_us[mask]
            seg_force = force[mask]

            if self.begin_detection_enabled and seg_force.size:
                begin_index = self.__detect_begin_index_highpass(seg_ts_us, seg_force)
                if begin_index is not None:
                    seg_ts_us = seg_ts_us[begin_index:]
                    seg_force = seg_force[begin_index:]

            if seg_ts_us.size == 0:
                previous_end_timestamp = int(end_timestamp)
                continue

            # trial-relative time from the (possibly trimmed) begin
            relative_time_us = seg_ts_us - seg_ts_us[0]

            max_force = float(np.nanmax(seg_force)) if seg_force.size else float("nan")
            min_force = float(np.nanmin(seg_force)) if seg_force.size else float("nan")

            rise_time_us = (
                self.__compute_rise_time_inverted(relative_time_us, seg_force)
                if seg_force.size
                else None
            )
            contact_duration_us = (
                self.__compute_contact_duration_inverted(relative_time_us, seg_force)
                if seg_force.size
                else None
            )
            peak_slope = (
                self.__compute_peak_slope_inverted(relative_time_us, seg_force)
                if seg_force.size
                else None
            )

            results.append(
                dict(
                    trial_index=int(trial_index),
                    gesture=gesture,
                    velocity=velocity,
                    start_ts_us=int(seg_ts_us[0]),
                    end_ts_us=int(end_timestamp),
                    timestamps_us=seg_ts_us.copy(),
                    force=seg_force.copy(),
                    n_samples=int(seg_force.size),
                    max_force=max_force,
                    min_force=min_force,
                    rise_time_us=rise_time_us,
                    contact_duration_us=contact_duration_us,
                    peak_slope=peak_slope,
                )
            )

            previous_end_timestamp = int(end_timestamp)

        return results

    # -------------------- Aggregate metrics (no pandas) --------------------

    def __compute_force_metrics(
        self, trials: List[dict]
    ) -> VelocityCalibrationForceMetrics:
        """Compute per-(gesture, velocity) aggregate stats, while keeping raw per-trial arrays."""
        conditions: Dict[Gestures, Dict[Velocity, VelocityCondition]] = {}

        def _stats(values: Iterable[Optional[float]]) -> MetricStats:
            arr = np.asarray([v for v in values if v is not None], dtype=float)
            if arr.size == 0:
                return MetricStats(mean=float("nan"), sd=float("nan"))

            mean = float(np.nanmean(arr))
            finite = arr[np.isfinite(arr)]
            if finite.size <= 1:
                sd = 0.0 if finite.size == 1 else float("nan")
            else:
                sd = float(np.std(finite, ddof=1))
            return MetricStats(mean=mean, sd=sd)

        by_cond: Dict[Tuple[Gestures, Velocity], List[dict]] = {}
        for t in trials:
            by_cond.setdefault((t["gesture"], t["velocity"]), []).append(t)

        for (gesture, velocity), cond_trials in by_cond.items():
            max_force_vals = [float(x["max_force"]) for x in cond_trials]
            min_force_vals = [float(x["min_force"]) for x in cond_trials]
            rise_time_vals = [x.get("rise_time_us") for x in cond_trials]
            contact_vals = [x.get("contact_duration_us") for x in cond_trials]
            slope_vals = [x.get("peak_slope") for x in cond_trials]

            cm = ConditionMetrics(
                max_force=_stats(max_force_vals),
                min_force=_stats(min_force_vals),
                rise_time_us=_stats(rise_time_vals),
                contact_duration_us=_stats(contact_vals),
                peak_slope=_stats(slope_vals),
            )

            vc = VelocityCondition(
                n_trials=len(cond_trials),
                metrics=cm,
                trials=cond_trials,
            )

            conditions.setdefault(gesture, {})[velocity] = vc

        return VelocityCalibrationForceMetrics(conditions=conditions)

    # -------------------- Plotting (no pandas) --------------------

    def _plot_summary_metric(
        self,
        metric_attr: str,
        y_label: str,
        title: str,
        velocity_order: Tuple[Velocity, ...] | None = None,
    ):
        if self._metrics is None:
            raise ValueError("Summary data not available. Run analyser first.")

        if velocity_order is None:
            velocity_order = (Velocity.SLOW, Velocity.NORMAL, Velocity.FAST)

        gestures = sorted(self._metrics.conditions.keys(), key=lambda g: g.value)
        x_positions = np.arange(len(gestures), dtype=float)
        bar_width = 0.35

        fig, ax = plt.subplots()

        for i, velocity in enumerate(velocity_order):
            means: List[float] = []
            sds: List[float] = []
            for g in gestures:
                cond = self._metrics.conditions.get(g, {}).get(velocity)
                if cond is None:
                    means.append(float("nan"))
                    sds.append(float("nan"))
                    continue
                stats_obj: MetricStats = getattr(cond.metrics, metric_attr)
                means.append(stats_obj.mean)
                sds.append(stats_obj.sd)

            offset = (i - 0.5) * bar_width
            ax.bar(
                x_positions + offset,
                np.asarray(means, dtype=float),
                width=bar_width,
                yerr=np.asarray(sds, dtype=float),
                capsize=4,
                label=str(velocity.value),
            )

        ax.set_xticks(x_positions)
        ax.set_xticklabels([g.value for g in gestures], rotation=30, ha="right")
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.legend(title="Velocity")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()

        return fig, ax

    def plot_rise_time_summary(self):
        return self._plot_summary_metric(
            metric_attr="rise_time_us",
            y_label="Rise time (µs)",
            title="Rise Time by Gesture and Velocity",
        )

    def plot_contact_duration_summary(self):
        return self._plot_summary_metric(
            metric_attr="contact_duration_us",
            y_label="Contact duration (µs)",
            title="Contact Duration by Gesture and Velocity",
        )

    def plot_peak_slope_summary(self):
        return self._plot_summary_metric(
            metric_attr="peak_slope",
            y_label="Peak slope (force / µs)",
            title="Peak Slope by Gesture and Velocity",
        )

    def __plot_metric_trials_scatter(
        self,
        metric: str,
        y_label: str | None = None,
        title: str | None = None,
        velocity_order: Tuple[Velocity, ...] | None = None,
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

        if velocity_order is None:
            velocity_order = (Velocity.SLOW, Velocity.NORMAL, Velocity.FAST)

        gestures = sorted({t["gesture"] for t in trials}, key=lambda g: g.value)
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
            cond_trials = [t for t in trials if t["velocity"] == velocity]
            if not cond_trials:
                continue

            ys_raw = [t.get(metric) for t in cond_trials]
            finite_mask = [y is not None and np.isfinite(float(y)) for y in ys_raw]
            cond_trials = [t for t, ok in zip(cond_trials, finite_mask) if ok]
            ys = [float(y) for y, ok in zip(ys_raw, finite_mask) if ok]
            if not cond_trials:
                continue

            base_x = np.asarray([x_map[t["gesture"]] for t in cond_trials], dtype=float)
            offset = float(vel_offsets.get(velocity, 0.0))
            x_vals = base_x + offset + rng.uniform(-jitter, jitter, size=base_x.size)
            y_vals = np.asarray(ys, dtype=float)

            ax.scatter(x_vals, y_vals, alpha=alpha, label=str(velocity.value))

            if annotate:
                for x, y, t in zip(x_vals, y_vals, cond_trials):
                    label = f"{t['trial_index']} | {t['gesture'].value} | {t['velocity'].value}"
                    ax.text(
                        x + text_offset[0],
                        y + text_offset[1],
                        label,
                        fontsize=fontsize,
                        alpha=0.75,
                    )

        ax.set_xticks(x_positions)
        ax.set_xticklabels([g.value for g in gestures], rotation=30, ha="right")
        ax.set_ylabel(y_label or metric)
        ax.set_xlabel("Gesture")
        ax.set_title(title or f"{metric} per trial by gesture and velocity")
        ax.grid(True, axis="y", alpha=0.3)
        if show_legend:
            ax.legend(title="Velocity")
        fig.tight_layout()
        return fig, ax

    def plot_rise_time_trials(self, **kwargs):
        return self.__plot_metric_trials_scatter(
            metric="rise_time_us",
            y_label="Rise time (µs)",
            title="Rise Time per Trial by Gesture and Velocity",
            **kwargs,
        )

    def plot_contact_duration_trials(self, **kwargs):
        return self.__plot_metric_trials_scatter(
            metric="contact_duration_us",
            y_label="Contact duration (µs)",
            title="Contact Duration per Trial by Gesture and Velocity",
            **kwargs,
        )

    def plot_peak_slope_trials(self, **kwargs):
        return self.__plot_metric_trials_scatter(
            metric="peak_slope",
            y_label="Peak slope (force / µs)",
            title="Peak Slope per Trial by Gesture and Velocity",
            **kwargs,
        )

    def plot_max_force_trials(self, **kwargs):
        return self.__plot_metric_trials_scatter(
            metric="max_force",
            y_label="Max force",
            title="Max Force per Trial by Gesture and Velocity",
            **kwargs,
        )

    def plot_min_force_trials(self, **kwargs):
        return self.__plot_metric_trials_scatter(
            metric="min_force",
            y_label="Min force",
            title="Min Force per Trial by Gesture and Velocity",
            **kwargs,
        )

    # -------------------- Begin detection + metric helpers --------------------

    def __detect_begin_index_highpass(
        self,
        timestamps_us: np.ndarray,
        force_values: np.ndarray,
    ) -> Optional[int]:
        if timestamps_us.size < 3 or force_values.size < 3:
            return None

        timestamp_deltas_us = np.diff(timestamps_us)
        valid_deltas = timestamp_deltas_us[timestamp_deltas_us > 0]
        if valid_deltas.size == 0:
            return None

        estimated_sample_period_us = float(np.median(valid_deltas))
        holdoff_us = float(self.begin_holdoff_ms) * 1000.0
        holdoff_samples = int(max(1, round(holdoff_us / estimated_sample_period_us)))

        high_pass_proxy = np.diff(force_values, prepend=force_values[0])
        trigger_indices = np.where(
            high_pass_proxy <= float(self.begin_highpass_threshold)
        )[0]
        if trigger_indices.size == 0:
            return None

        begin_index = int(trigger_indices[0])
        if (force_values.size - begin_index) < max(10, holdoff_samples):
            return None

        return begin_index

    # expects baseline to be high, and press to be lower.
    def __compute_rise_time_inverted(
        self,
        timestamps_us_relative_to_trial_start: np.ndarray,
        force_values_raw: np.ndarray,
    ) -> Optional[float]:
        if (
            timestamps_us_relative_to_trial_start.size < 10
            or force_values_raw.size < 10
        ):
            return None

        number_of_baseline_samples = min(
            force_values_raw.size,
            max(5, int(0.10 * force_values_raw.size)),
        )
        baseline_force_estimate = float(
            np.nanmedian(force_values_raw[:number_of_baseline_samples])
        )
        trough_force_value = float(np.nanmin(force_values_raw))

        if not (
            np.isfinite(baseline_force_estimate) and np.isfinite(trough_force_value)
        ):
            return None

        press_amplitude = baseline_force_estimate - trough_force_value
        if press_amplitude <= 1e-6:
            return None

        normalized_press_depth = (
            baseline_force_estimate - force_values_raw
        ) / press_amplitude

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

            if not (
                np.isfinite(y0)
                and np.isfinite(y1)
                and np.isfinite(t0)
                and np.isfinite(t1)
            ):
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

    def __compute_contact_duration_inverted(
        self,
        timestamps_us_relative_to_trial_start: np.ndarray,
        force_values_raw: np.ndarray,
        press_depth_threshold: float = 0.10,
    ) -> Optional[float]:
        if (
            timestamps_us_relative_to_trial_start.size < 10
            or force_values_raw.size < 10
        ):
            return None

        number_of_baseline_samples = min(
            force_values_raw.size,
            max(5, int(0.10 * force_values_raw.size)),
        )
        baseline_force_value = float(
            np.nanmedian(force_values_raw[:number_of_baseline_samples])
        )
        trough_force_value = float(np.nanmin(force_values_raw))
        if not (np.isfinite(baseline_force_value) and np.isfinite(trough_force_value)):
            return None

        press_amplitude = baseline_force_value - trough_force_value
        if press_amplitude <= 1e-6:
            return None

        normalized_press_depth = (
            baseline_force_value - force_values_raw
        ) / press_amplitude
        index_contact_active = np.where(
            normalized_press_depth >= press_depth_threshold
        )[0]
        if index_contact_active.size < 2:
            return None

        first_contact_index = int(index_contact_active[0])
        last_contact_index = int(index_contact_active[-1])
        return float(
            timestamps_us_relative_to_trial_start[last_contact_index]
            - timestamps_us_relative_to_trial_start[first_contact_index]
        )

    def __compute_peak_slope_inverted(
        self,
        timestamps_us_relative_to_trial_start: np.ndarray,
        force_values_raw: np.ndarray,
    ) -> Optional[float]:
        if timestamps_us_relative_to_trial_start.size < 5 or force_values_raw.size < 5:
            return None

        delta_time = np.diff(timestamps_us_relative_to_trial_start)
        delta_force = np.diff(force_values_raw)
        valid_time_steps_mask = delta_time > 0
        if not np.any(valid_time_steps_mask):
            return None

        slopes_force_per_us = (
            delta_force[valid_time_steps_mask] / delta_time[valid_time_steps_mask]
        )
        if slopes_force_per_us.size == 0:
            return None

        most_negative_slope = float(np.nanmin(slopes_force_per_us))
        peak_downward_slope_magnitude = abs(most_negative_slope)
        return float(peak_downward_slope_magnitude)


############################################################################


def main():
    # analyser = TrialBlockForceAnalyser(
    #     r"C:\Users\dm-potts-admin\Documents\Postdoc\UWE\Outside_Interactions\Object_Characterisation_Study\Study_Program\Twintig_Study_Scrapbook\Logged_Data\Trial_Force_Data\Tap_Pads_Data"
    # )
    # analyser.run()
    # analyser.plot_scatter_by_sample_id("Last block: min force per trial (fast vs slow)")
    # analyser.plot_raw_trials_side_by_side()

    calAnalyser = VelocityCalibrationAnalyser(
        r"C:\Users\dm-potts-admin\Documents\Postdoc\UWE\Outside_Interactions\Object_Characterisation_Study\Study_Program\Twintig_Study_Scrapbook\Logged_Data\Velocity_Calibration_Force_Data\velocityCal"
    )
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
