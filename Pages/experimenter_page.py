from collections import deque
from typing import Deque

from PySide6 import QtCore, QtGui, QtStateMachine, QtWidgets

from twintig_logger import FSRPacket, TwintigLogger

import time


QState = QtStateMachine.QState
QStateMachine = QtStateMachine.QStateMachine


class LogBus(QtCore.QObject):
    message = QtCore.Signal(str)

    def __init__(self):
        super().__init__()
        self.history: list[str] = []

    def log(self, text: str):
        self.history.append(text)
        self.message.emit(text)


# ------------------ Base page with black theme + nav bar ------------------
class ExperimenterPage(QtWidgets.QWidget):
    # Global control signals (hook in MainWindow)
    connectRequested = QtCore.Signal()
    disconnectRequested = QtCore.Signal()
    pauseToggled = QtCore.Signal(bool)  # True when paused
    recordingToggled = QtCore.Signal(bool)

    # Navigation signals (wired by controller)
    backRequested = QtCore.Signal()  # fixed back (MAIN_BACK)
    navRequested = QtCore.Signal(str)  # go to target page
    studyPhaseRequested = QtCore.Signal(object)  # will carry a StudyPhases value

    def __init__(self, name: str, log_bus: LogBus):
        super().__init__()
        self.page_name = name
        self.log_bus = log_bus
        self.nav_buttons = {}
        self.back_button = None

        self._logger: TwintigLogger | None = None
        self._devices_connected: bool = False
        self._recording: bool = False
        self._msg_rate_hz: float = 0.0

        self._poll_timer = QtCore.QTimer(self)
        self._poll_timer.setInterval(250)  # 4 Hz UI refresh is plenty
        self._poll_timer.timeout.connect(self._poll_logger_state)

        # rolling window for Hz (last 2 seconds)
        self._ts_window_us: Deque[int] = deque(maxlen=512)

        # Internal status
        self._devices_connected = False
        self._paused = False
        self._recording = False
        self._msg_rate_hz = 0.0
        self._participant_id: str | None = None
        self._study_phase = None
        self._sample_group = None
        self._sample_id: int | None = None
        self._sample_name: str | None = None
        self._gesture = None
        self._trial_id: int | None = None

        # Dark theme
        self.setStyleSheet("""
            QWidget { background-color: #000; color: #f0f0f0; font-family: Segoe UI, sans-serif; }
            QPushButton { background-color: #222; color: #f0f0f0; border: 1px solid #555; border-radius: 6px; padding: 6px 12px; }
            QPushButton:hover { background-color: #333; }
            QLabel { color: #f0f0f0; }
            QPlainTextEdit, QLineEdit, QSpinBox, QComboBox, QCheckBox { background-color: #111; color: #f0f0f0; }
            QProgressBar { border: 1px solid #444; text-align: center; background: #111; color: #fff; }
            QProgressBar::chunk { background-color: #0f7; }
        """)

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(10)

        # Title
        title = QtWidgets.QLabel(name)
        title.setAlignment(QtCore.Qt.AlignCenter)
        title.setStyleSheet("font-size: 26px; font-weight: 600;")
        # root.addWidget(title)

        # small pill-style labels ("chips")
        chip_css = """
            QLabel {
                color: #ddd; background: #111; border: 1px solid #444;
                border-radius: 10px; padding: 4px 8px; font-size: 12px;
            }
        """

        self._lbl_participant = QtWidgets.QLabel("Participant: —")

        self._lbl_phase = QtWidgets.QLabel("Phase: —")
        self._lbl_sample_group = QtWidgets.QLabel("Sample_Group: —")
        self._lbl_sample_id = QtWidgets.QLabel("Sample_ID: —")
        self._lbl_sample_name = QtWidgets.QLabel("Sample_Name: —")
        self._lbl_gesture = QtWidgets.QLabel("Gesture: —")
        self._lbl_trial = QtWidgets.QLabel("Trial: —")

        for lbl in (
            self._lbl_participant,
            self._lbl_phase,
            self._lbl_sample_group,
            self._lbl_sample_id,
            self._lbl_sample_name,
            self._lbl_gesture,
            self._lbl_trial,
        ):
            lbl.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            lbl.setStyleSheet(chip_css)

        # Put title + chip on one row
        title_row = QtWidgets.QHBoxLayout()
        title_row.addStretch(1)
        title_row.addWidget(title, 0, QtCore.Qt.AlignCenter)
        title_row.addStretch(1)

        chips_row = QtWidgets.QHBoxLayout()
        chips_row.setSpacing(6)
        chips_row.addWidget(self._lbl_phase)
        chips_row.addWidget(self._lbl_gesture)
        chips_row.addWidget(self._lbl_sample_group)
        chips_row.addWidget(self._lbl_sample_id)
        chips_row.addWidget(self._lbl_sample_name)
        chips_row.addWidget(self._lbl_trial)
        chips_row.addWidget(self._lbl_participant)

        title_row.addLayout(chips_row)
        root.addLayout(title_row)

        # Page hint
        self.status_label = QtWidgets.QLabel("")
        self.status_label.setAlignment(QtCore.Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #aaa;")
        root.addWidget(self.status_label)

        # --------- Shared status + log panel ----------
        panel = QtWidgets.QGroupBox("System Status & Log")
        panel_layout = QtWidgets.QVBoxLayout(panel)

        ind_row = QtWidgets.QHBoxLayout()

        def make_led(label_text):
            dot = QtWidgets.QLabel("●")
            dot.setFixedWidth(14)
            dot.setAlignment(QtCore.Qt.AlignCenter)
            txt = QtWidgets.QLabel(label_text)
            wrap = QtWidgets.QWidget()
            h = QtWidgets.QHBoxLayout(wrap)
            h.setSpacing(6)
            h.addWidget(dot)
            h.addWidget(txt)
            return wrap, dot, txt

        self._led_devices_wrap, self._led_devices, self._lbl_devices = make_led(
            "Devices: Disconnected"
        )
        self._led_msg_wrap, self._led_msg, self._lbl_msg = make_led("Msg rate: 0.0 Hz")
        self._led_rec_wrap, self._led_rec, self._lbl_rec = make_led("Recording: Off")

        ind_row.addWidget(self._led_devices_wrap)
        ind_row.addSpacing(12)
        ind_row.addWidget(self._led_msg_wrap)
        ind_row.addSpacing(12)
        ind_row.addWidget(self._led_rec_wrap)
        ind_row.addStretch(1)
        panel_layout.addLayout(ind_row)

        # Log view — we’ll attach a shared QTextDocument from the controller
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("System log output…")
        self.log.setMinimumHeight(120)
        panel_layout.addWidget(self.log)

        # Controls
        ctrl_row = QtWidgets.QHBoxLayout()
        self.btn_connect = QtWidgets.QPushButton("Connect Devices")
        self.btn_disconnect = QtWidgets.QPushButton("Disconnect Devices")
        self.btn_pause_resume = QtWidgets.QPushButton("Pause Experiment")
        self.btn_connect.clicked.connect(self._on_connect_clicked)
        self.btn_disconnect.clicked.connect(self._on_disconnect_clicked)
        self.btn_pause_resume.clicked.connect(self._on_pause_resume_clicked)
        ctrl_row.addWidget(self.btn_connect)
        ctrl_row.addWidget(self.btn_disconnect)
        ctrl_row.addStretch(1)
        ctrl_row.addWidget(self.btn_pause_resume)
        panel_layout.addLayout(ctrl_row)

        root.addWidget(panel)

        # Page-specific content area
        self.content = QtWidgets.QWidget()
        self.content_layout = QtWidgets.QVBoxLayout(self.content)
        root.addWidget(self.content, 1)

        # Main nav bar
        self.nav_bar = QtWidgets.QHBoxLayout()
        root.addLayout(self.nav_bar)

        self._refresh_indicators()

        # --- hook to the log bus ---
        for line in self.log_bus.history:  # replay past messages
            self._append_to_log_widget(line)
        self.log_bus.message.connect(self._append_to_log_widget)

    # ---------- NEW: Participant ID API ----------
    def set_participant_id(self, pid: str | None):
        """Called by controller to propagate the active participant ID."""
        self._participant_id = pid
        self._lbl_participant.setText(
            f"Participant: {pid}" if pid not in (None, "") else "Participant: —"
        )

    # ---------- Experiment context chips API ----------
    def set_study_phase(self, phase):
        """phase can be a StudyPhases enum, string, or None."""
        self._study_phase = phase
        txt = "—"
        if phase not in (None, ""):
            # enums will usually have .name, but we fall back to str()
            txt = getattr(phase, "name", str(phase))
        self._lbl_phase.setText(f"Phase: {txt}")

    def set_sample_group(self, group):
        """group can be a SampleGroup enum, string, or None."""
        self._sample_group = group
        txt = "—"
        if group not in (None, ""):
            txt = getattr(group, "name", str(group))
        self._lbl_sample_group.setText(f"Group: {txt}")

    def set_sample_id(self, sample_id):
        """sample_id is typically an int, -1/None means 'no sample'."""
        self._sample_id = sample_id
        if sample_id in (None, -1, ""):
            txt = "—"
        else:
            txt = str(sample_id)
        self._lbl_sample_id.setText(f"Sample: {txt}")

    def set_sample_name(self, name):
        """Human-readable name for the sample."""
        self._sample_name = name
        txt = "—" if not name else str(name)
        self._lbl_sample_name.setText(f"Name: {txt}")

    def set_trial_id(self, trial_id):
        """Current trial index / ID, -1/None means 'no trial'."""
        self._trial_id = trial_id
        if trial_id in (None, -1, ""):
            txt = "—"
        else:
            txt = str(trial_id)
        self._lbl_trial.setText(f"Trial: {txt}")

    def set_gesture(self, gesture):
        """Current trial index / ID, -1/None means 'no trial'."""
        self._gesture = gesture
        txt = "—"
        if gesture not in (None, ""):
            txt = getattr(gesture, "name", str(gesture))
        self._lbl_gesture.setText(f"Group: {txt}")

    # All logging should go through the bus so it appears on every page
    def append_log(self, text: str):
        self.log_bus.log(text)

    # local helper to render a line (don’t call this directly from app code)
    def _append_to_log_widget(self, text: str):
        self.log.appendPlainText(text)
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

        # ------------ new: DI for the shared logger ------------

    def set_logger(self, logger: TwintigLogger):
        self._logger = logger
        # If already open (another page connected), reflect that in our LEDs/labels
        self._devices_connected = logger.is_open
        self._recording = logger.is_logging
        self._msg_rate_hz = logger.get_msg_rate_hz(2.0)
        self._refresh_indicators()

    # ------------ internal: FSR → message-rate ------------
    def _fsr_rx(self, pkt: FSRPacket):
        # compute Hz over a short rolling time window
        self._ts_window_us.append(pkt.timestamp_us)
        if len(self._ts_window_us) >= 2:
            dt_us = self._ts_window_us[-1] - self._ts_window_us[0]
            if dt_us > 0:
                hz = (len(self._ts_window_us) - 1) / (dt_us / 1_000_000.0)
                self.set_msg_rate(hz)

    # ------------ device lifecycle handlers ------------
    @QtCore.Slot()
    def _on_connect_clicked(self):
        if not self._logger:
            QtWidgets.QMessageBox.critical(self, "Error", "Logger not initialised.")
            return
        try:
            self._logger.open()
            self.append_log("Devices connected.")
        except Exception as e:
            self.append_log(f"Device open failed: {e}")
            QtWidgets.QMessageBox.critical(self, "Connection failed", str(e))

    @QtCore.Slot()
    def _on_disconnect_clicked(self):
        if not self._logger:
            return
        try:
            self._logger.close()
            self.append_log("Devices disconnected.")
        except Exception as e:
            self.append_log(f"Close error: {e}")

    @QtCore.Slot()
    def _on_start_rec_clicked(self):
        if not self._logger:
            return
        if not self._logger.is_open:
            # auto-open if user presses record first
            try:
                self._logger.open()
            except Exception as e:
                self.log_bus.log(f"Open before record failed: {e}")
                QtWidgets.QMessageBox.critical(self, "Record failed", str(e))
                return
            self._logger.add_fsr_callback(self._fsr_rx)
            self._devices_connected = True

        # Optional: make per-participant folder names
        log_name = "Logged Data"
        pid = getattr(self, "participant_id", lambda: None)()
        if pid:
            log_name = f"PID_{pid}"

        try:
            # You can also pass delete_existing / make_unique_on_conflict here
            self._logger.log_name = log_name
            self._logger.start_logging()
        except Exception as e:
            self.log_bus.log(f"Start logging failed: {e}")
            QtWidgets.QMessageBox.critical(self, "Record failed", str(e))
            return

        self._recording = True
        self.log_bus.log(f"Recording started → {self._logger.log_name}")
        self._refresh_indicators()

    @QtCore.Slot()
    def _on_stop_rec_clicked(self):
        if not self._logger:
            return
        try:
            self._logger.stop_logging()
        except Exception as e:
            self.log_bus.log(f"Stop logging error: {e}")
        self._recording = False
        self.log_bus.log("Recording stopped.")
        self._refresh_indicators()

    # ---------- Public status helpers ----------
    def set_status(self, text: str):
        self.status_label.setText(text)

    def add_content_widget(self, w: QtWidgets.QWidget):
        self.content_layout.addWidget(w)

    def set_devices_connected(self, connected: bool):
        self._devices_connected = connected
        self._refresh_indicators()

    def set_msg_rate(self, hz: float):
        self._msg_rate_hz = max(0.0, float(hz))
        self._refresh_indicators()

    def set_recording(self, recording: bool):
        self._recording = recording
        self._refresh_indicators()
        self.recordingToggled.emit(recording)

    def set_paused(self, paused: bool):
        self._paused = paused
        self.btn_pause_resume.setText(
            "Resume Experiment" if paused else "Pause Experiment"
        )

    # ---------- Build simple nav (fixed back target only) ----------
    def build_nav(self, main_targets: list[str], back_target: str | None):
        for i in reversed(range(self.nav_bar.count())):
            item = self.nav_bar.itemAt(i)
            w = item.widget()
            if w:
                w.setParent(None)
        self.nav_buttons.clear()
        self.back_button = None

        if back_target:
            btn = QtWidgets.QPushButton("⬅ Back")
            btn.clicked.connect(self.backRequested.emit)
            self.nav_bar.addWidget(btn)
            self.back_button = btn

        self.nav_bar.addStretch(1)

        for t in main_targets:
            b = QtWidgets.QPushButton(f"Go to {t} ➜")
            b.clicked.connect(lambda _, target=t: self.navRequested.emit(target))
            self.nav_bar.addWidget(b)
            self.nav_buttons[t] = b

    def _on_pause_resume_clicked(self):
        self._paused = not self._paused
        self.btn_pause_resume.setText(
            "Resume Experiment" if self._paused else "Pause Experiment"
        )
        self.append_log(f"[ui] {'Paused' if self._paused else 'Resumed'} experiment.")
        self.pauseToggled.emit(self._paused)

    def _refresh_indicators(self):
        def set_led(
            led_label: QtWidgets.QLabel, on: bool, on_color="#0f7", off_color="#444"
        ):
            led_label.setStyleSheet(
                f"color: {on_color if on else off_color}; font-size: 18px;"
            )

        set_led(self._led_devices, self._devices_connected)
        self._lbl_devices.setText(
            f"Devices: {'Connected' if self._devices_connected else 'Disconnected'}"
        )
        set_led(self._led_msg, self._msg_rate_hz > 0.1)
        self._lbl_msg.setText(f"Msg rate: {self._msg_rate_hz:.1f} Hz")
        set_led(self._led_rec, self._recording, on_color="#fa0")
        self._lbl_rec.setText(f"Recording: {'On' if self._recording else 'Off'}")

    def showEvent(self, e: QtGui.QShowEvent):
        super().showEvent(e)
        if self._logger:
            self._poll_timer.start()
            self._poll_logger_state()

    def hideEvent(self, e: QtGui.QHideEvent):
        self._poll_timer.stop()
        super().hideEvent(e)

    def _poll_logger_state(self):
        if not self._logger:
            return
        self._devices_connected = self._logger.is_open
        self._recording = self._logger.is_logging
        self._msg_rate_hz = self._logger.get_msg_rate_hz(
            2.0
        )  # total across counted streams
        self._refresh_indicators()


    