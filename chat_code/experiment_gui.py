"""
Two-window GUI for FSR Study (Participant + Experimenter)
---------------------------------------------------------

- Participant Window: big "powerbar" that shows live FSR magnitude vs. a target
  threshold derived from calibration for the *current channel* and *requested intensity*
  (soft/medium/hard). Text prompt at top: what to do next.

- Experimenter Window: controls to Start/Pause/Stop logging, pick the active channel,
  run guided calibration (collect 10 soft, 10 medium, 10 hard taps), then start the trial.

Calibration logic
- Baseline is estimated per-channel from a running median filter.
- Taps are detected using a simple adaptive threshold above baseline.
- For each intensity bucket we store the peak amplitudes (peak-baseline) of 10 taps.
- The threshold used for trials is the median amplitude of the corresponding bucket.

Requirements
- pip install PySide6 ximu3

Run
- python experiment_gui.py

Note
- The GUI uses the `TwintigLogger` library (twintig_logger.py). Keep both files together
  or install the library on PYTHONPATH.
"""

from __future__ import annotations

import math
import sys
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from PySide6 import QtCore, QtGui, QtWidgets
from twintig_logger import FSRPacket, TwintigLogger, get_logger

# ---------------------------- Calibration Model ---------------------------- #


class Intensity(str, Enum):
    SOFT = "soft"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class ChannelCalib:
    baseline: float = 0.0
    baseline_alpha: float = 0.02  # IIR baseline smoothing
    # buffers of detected tap peak amplitudes (peak - baseline)
    soft_peaks: List[float] = field(default_factory=list)
    med_peaks: List[float] = field(default_factory=list)
    hard_peaks: List[float] = field(default_factory=list)

    def update_baseline(self, sample: float) -> None:
        # IIR low-pass to track baseline drift
        self.baseline = (1 - self.baseline_alpha) * self.baseline + self.baseline_alpha * sample

    def add_peak(self, intensity: Intensity, amp: float) -> None:
        if intensity == Intensity.SOFT:
            self.soft_peaks.append(amp)
        elif intensity == Intensity.MEDIUM:
            self.med_peaks.append(amp)
        else:
            self.hard_peaks.append(amp)

    def bucket(self, intensity: Intensity) -> List[float]:
        return self.soft_peaks if intensity == Intensity.SOFT else self.med_peaks if intensity == Intensity.MEDIUM else self.hard_peaks

    def threshold(self, intensity: Intensity) -> Optional[float]:
        data = self.bucket(intensity)
        if len(data) < 3:
            return None
        data_sorted = sorted(data)
        mid = len(data_sorted) // 2
        return data_sorted[mid]  # median amplitude


class CalibrationManager(QtCore.QObject):
    calib_updated = QtCore.Signal()  # emitted when thresholds progress

    def __init__(self, n_channels: int = 8) -> None:
        super().__init__()
        self.channels: List[ChannelCalib] = [ChannelCalib() for _ in range(n_channels)]
        self.current_channel = 0
        self.target_intensity: Optional[Intensity] = None
        self.required_count = 10
        self.min_intertap_s = 0.20
        self.refractory_s = 0.25
        self._last_tap_time = 0.0
        self._armed = False
        self._arm_threshold_mult = 3.0  # SDs above baseline to arm
        self._running_mean = [0.0] * n_channels
        self._running_var = [1.0] * n_channels
        self._mean_alpha = 0.05
        # Calibration session state
        self.active: bool = False
        # CSV logging of calibration samples
        self._csv_path: Optional[str] = None
        self._csv_file = None
        self._csv_writer = None

    def set_channel(self, idx: int) -> None:
        self.current_channel = max(0, min(idx, len(self.channels) - 1))

    def set_csv_path(self, path: Optional[str]) -> None:
        # Close any previous file
        try:
            if self._csv_file:
                self._csv_file.close()
        except Exception:
            pass
        self._csv_file = None
        self._csv_writer = None
        self._csv_path = path
        if path:
            import csv
            import os

            os.makedirs(os.path.dirname(path), exist_ok=True)
            self._csv_file = open(path, "a", newline="", encoding="utf-8")
            self._csv_writer = csv.writer(self._csv_file)
            # write header only if empty file
            try:
                if self._csv_file.tell() == 0:
                    self._csv_writer.writerow(["timestamp_us", "channel", "fsr_value", "intensity"])
                    self._csv_file.flush()
            except Exception:
                pass

    def start_bucket(self, intensity: Intensity) -> None:
        self.target_intensity = intensity
        self._armed = False
        self._last_tap_time = 0.0
        self.active = True

    def stop_calibration(self) -> None:
        self.active = False
        self._armed = False

    def is_done(self) -> bool:
        if not self.active or self.target_intensity is None:
            return False
        return len(self.channels[self.current_channel].bucket(self.target_intensity)) >= self.required_count

    def _update_stats(self, ch: int, x: float) -> Tuple[float, float]:
        # Welford-esque IIR stats for simple z-scoring
        m = self._running_mean[ch]
        v = self._running_var[ch]
        m2 = (1 - self._mean_alpha) * m + self._mean_alpha * x
        v2 = (1 - self._mean_alpha) * v + self._mean_alpha * (x - m2) ** 2
        self._running_mean[ch], self._running_var[ch] = m2, max(v2, 1e-6)
        return m2, math.sqrt(self._running_var[ch])

    def ingest(self, pkt: FSRPacket) -> None:
        ch = self.current_channel
        x = pkt.values[ch]
        ch_state = self.channels[ch]

        # Update baseline slowly regardless of phase
        ch_state.update_baseline(x)

        # Maintain quick stats for arming
        mean, std = self._update_stats(ch, x)
        z = (x - mean) / (std or 1.0)

        if not self.active or self.target_intensity is None:
            return

        now = time.monotonic()
        # Arm when signal rises notably
        if not self._armed and z > self._arm_threshold_mult:
            self._armed = True
            self._candidate_peak = x
        if self._armed:
            self._candidate_peak = max(self._candidate_peak, x)
            # Disarm if we fall below mean (peak passed)
            if x < mean:
                if now - self._last_tap_time > self.min_intertap_s:
                    amp = max(0.0, self._candidate_peak - ch_state.baseline)
                    if (now - self._last_tap_time) > self.refractory_s:
                        ch_state.add_peak(self.target_intensity, amp)
                        # Write CSV row with raw peak value
                        self._write_csv(
                            pkt.timestamp_us,
                            ch,
                            self._candidate_peak,
                            self.target_intensity,
                        )
                        self._last_tap_time = now
                        self.calib_updated.emit()
                self._armed = False

    def _write_csv(self, ts_us: int, ch: int, value: float, intensity: Intensity) -> None:
        if self._csv_writer is None:
            return
        try:
            self._csv_writer.writerow([int(ts_us), int(ch), float(value), intensity.value])
            self._csv_file.flush()
        except Exception:
            pass

    def threshold_for(self, channel: int, intensity: Intensity) -> Optional[float]:
        return self.channels[channel].threshold(intensity)

    def baseline_for(self, channel: int) -> float:
        return self.channels[channel].baseline
        return self.channels[channel].baseline


# ------------------------------- UI Widgets ------------------------------- #
class PowerBar(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._value = 0.0
        self._target = None  # absolute target in same units as value
        self.setMinimumSize(200, 40)

    def set_value(self, value: float) -> None:
        self._value = max(0.0, value)
        self.update()

    def set_target(self, target: Optional[float]) -> None:
        self._target = target
        self.update()

    def paintEvent(self, e):
        p = QtGui.QPainter(self)
        r = self.rect()
        p.fillRect(r, self.palette().window())

        # Draw frame
        p.setPen(QtGui.QPen(QtCore.Qt.black, 2))
        p.drawRect(r.adjusted(1, 1, -1, -1))

        # Determine scale
        max_val = max(self._target or 0.0, self._value, 1.0)
        fill_w = int((self._value / max_val) * (r.width() - 4))
        # Fill current value
        p.fillRect(2, 2, fill_w, r.height() - 4, self.palette().highlight())

        # Draw target marker
        if self._target is not None and max_val > 0:
            x = 2 + int((self._target / max_val) * (r.width() - 4))
            p.setPen(QtGui.QPen(QtCore.Qt.red, 2))
            p.drawLine(x, 2, x, r.height() - 2)


class ParticipantWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Participant")
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        self.prompt = QtWidgets.QLabel("Waiting…")
        self.prompt.setAlignment(QtCore.Qt.AlignCenter)
        self.prompt.setStyleSheet("font-size: 28px; font-weight: 600;")

        self.bar = PowerBar()
        self.bar.setFixedHeight(80)

        layout.addWidget(self.prompt)
        layout.addWidget(self.bar)

    def set_prompt(self, text: str) -> None:
        self.prompt.setText(text)

    def set_bar(self, value: float, target: Optional[float]):
        self.bar.set_value(value)
        self.bar.set_target(target)


class ExperimenterWindow(QtWidgets.QMainWindow):
    startCalib = QtCore.Signal()
    pauseLog = QtCore.Signal()
    resumeLog = QtCore.Signal()
    stopLog = QtCore.Signal()
    startLog = QtCore.Signal()
    channelChanged = QtCore.Signal(int)
    intensitySelected = QtCore.Signal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Experimenter")
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        # Logging controls
        logRow = QtWidgets.QHBoxLayout()
        self.btnStart = QtWidgets.QPushButton("Start Logging")
        self.btnPause = QtWidgets.QPushButton("Pause")
        self.btnResume = QtWidgets.QPushButton("Resume")
        self.btnStop = QtWidgets.QPushButton("Stop")
        logRow.addWidget(self.btnStart)
        logRow.addWidget(self.btnPause)
        logRow.addWidget(self.btnResume)
        logRow.addWidget(self.btnStop)

        # Log status label
        self.lblLogStatus = QtWidgets.QLabel("Logging: stopped")

        # Channel select
        chRow = QtWidgets.QHBoxLayout()
        chRow.addWidget(QtWidgets.QLabel("Channel:"))
        self.chSpin = QtWidgets.QSpinBox()
        self.chSpin.setRange(0, 7)
        chRow.addWidget(self.chSpin)

        # Calibration
        calibRow = QtWidgets.QHBoxLayout()
        self.btnCalib = QtWidgets.QPushButton("Start Calibration")
        self.cmbIntensity = QtWidgets.QComboBox()
        self.cmbIntensity.addItems([i.value for i in Intensity])
        calibRow.addWidget(self.btnCalib)
        calibRow.addWidget(QtWidgets.QLabel("Bucket:"))
        calibRow.addWidget(self.cmbIntensity)

        # Progress
        self.lblProg = QtWidgets.QLabel("Calib: 0/10")

        layout.addLayout(logRow)
        layout.addWidget(self.lblLogStatus)
        layout.addLayout(chRow)
        layout.addLayout(calibRow)
        layout.addWidget(self.lblProg)

        # Wire signals
        self.btnStart.clicked.connect(self.startLog.emit)
        self.btnCalib.clicked.connect(self.startCalib.emit)
        self.btnPause.clicked.connect(self.pauseLog.emit)
        self.btnResume.clicked.connect(self.resumeLog.emit)
        self.btnStop.clicked.connect(self.stopLog.emit)
        self.chSpin.valueChanged.connect(self.channelChanged.emit)
        self.cmbIntensity.currentTextChanged.connect(self.intensitySelected.emit)

    def set_progress(self, n: int, total: int):
        self.lblProg.setText(f"Calib: {n}/{total}")

    def set_log_status(self, text: str):
        self.lblLogStatus.setText(text)


# ------------------------------- Controller ------------------------------- #
class Controller(QtCore.QObject):
    def __init__(self, app: QtWidgets.QApplication):
        super().__init__()
        self.app = app
        self.logger = get_logger()
        # Use a timestamped session name to avoid "Entity already exists" between runs
        try:
            import time as _time

            self.logger.log_name = _time.strftime("Logged Data %Y-%m-%d_%H-%M-%S")
        except Exception:
            pass

        self.calib = CalibrationManager(n_channels=8)

        # Windows
        self.wPart = ParticipantWindow()
        self.wExp = ExperimenterWindow()

        # Place on two displays if present
        screens = app.screens()
        if len(screens) >= 2:
            self.wPart.setGeometry(screens[0].geometry())
            self.wExp.setGeometry(screens[1].geometry())
        else:
            # Tile side by side
            g = screens[0].availableGeometry()
            half = g.width() // 2
            self.wPart.setGeometry(g.x(), g.y(), half, g.height())
            self.wExp.setGeometry(g.x() + half, g.y(), g.width() - half, g.height())

        # Connect model updates
        self.calib.calib_updated.connect(self._on_calib_progress)

        # Logger callback -> Qt thread via signal
        self._fsr_signal = QtCore.SignalInstance = type("_Sig", (QtCore.QObject,), {"sig": QtCore.Signal(object)})()
        self._fsr_signal.sig.connect(self._on_fsr)
        self.logger.add_fsr_callback(lambda pkt: self._fsr_signal.sig.emit(pkt))

        # Wire experimenter controls
        self.wExp.startLog.connect(self._start_logging)
        self.wExp.startCalib.connect(self._start_calib)
        self.wExp.pauseLog.connect(self._pause_logging)
        self.wExp.resumeLog.connect(self._resume_logging)
        self.wExp.stopLog.connect(self._stop_logging)
        self.wExp.channelChanged.connect(self._set_channel)
        self.wExp.intensitySelected.connect(self._set_intensity)

        # Initial UI state: no calibration active
        self.calib.active = False
        self.calib.target_intensity = None
        self.wExp.set_log_status("Logging: stopped")
        self.wPart.set_prompt("Waiting…")

        # Show as normal windows (keep title bar with minimize/maximize/close)
        self.wPart.show()
        self.wExp.show()

    # ---------------------------- Event Handlers ---------------------------- #

    def _start_logging(self):
        try:
            self.logger.start_logging(delete_existing=False)
            self.wExp.set_log_status(f"Logging: recording → {self.logger.log_name}")
        except Exception as e:
            self.wExp.set_log_status(f"Logging: error — {e}")

    def _pause_logging(self):
        self.logger.pause_logging()
        self.wExp.set_log_status("Logging: paused")

    def _resume_logging(self):
        self.logger.resume_logging()
        self.wExp.set_log_status(f"Logging: recording → {self.logger.log_name}")

    def _stop_logging(self):
        self.logger.stop_logging()
        self.wExp.set_log_status("Logging: stopped")

    # ---------------------------- Event Handlers ---------------------------- #

    def _on_fsr(self, pkt: FSRPacket):
        # Feed calibration detector (only records taps when active)
        self.calib.ingest(pkt)

        # Compute value for current channel
        ch = self.calib.current_channel
        baseline = self.calib.baseline_for(ch)
        cur_val = max(0.0, pkt.values[ch] - baseline)

        if self.calib.active and self.calib.target_intensity is not None:
            intensity = self.calib.target_intensity
            target = self.calib.threshold_for(ch, intensity)
            label = f"Tap {intensity.value} on channel {ch}"
        else:
            target = None
            label = "Waiting…"

        # Update participant view
        self.wPart.set_prompt(label)
        self.wPart.set_bar(cur_val, target)

    def _on_calib_progress(self):
        intensity = self.calib.target_intensity or Intensity.SOFT
        n = len(self.calib.channels[self.calib.current_channel].bucket(intensity))
        self.wExp.set_progress(n, self.calib.required_count)

    def _start_calib(self):
        intensity = Intensity(self.wExp.cmbIntensity.currentText())
        self.calib.start_bucket(intensity)
        # Set CSV path for this run under the logger's session folder
        import os
        import time as _time

        cal_csv = os.path.join(self.logger.log_destination, self.logger.log_name, "calibration_samples.csv")
        self.calib.set_csv_path(cal_csv)
        self._on_calib_progress()

    def _set_channel(self, idx: int):
        self.calib.set_channel(idx)

    def _set_intensity(self, name: str):
        # Do not auto-start calibration; only change the intended bucket if active
        try:
            inten = Intensity(name)
        except Exception:
            inten = Intensity.SOFT
        if self.calib.active:
            self.calib.target_intensity = inten


# --------------------------------- main --------------------------------- #


def main():
    app = QtWidgets.QApplication(sys.argv)
    ctrl = Controller(app)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
