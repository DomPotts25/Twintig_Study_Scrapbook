from PySide6 import QtCore, QtGui, QtStateMachine, QtWidgets

QState = QtStateMachine.QState
QStateMachine = QtStateMachine.QStateMachine

from experiment_factors import Intensity


# ------------------ 1) Full directed graph (legal transitions) ------------------
# ------------------ 1) Full directed graph (legal transitions) ------------------
def add_edge(g, a, b):
    g.setdefault(a, []).append(b)


GRAPH = {}

# Setup1 <-> Setup2 / StillCalib / ROM_1_Calib / ROM_2_Calib / Midair_Calib
for tgt in ["Setup2", "StillCalib", "ROM_1_Calib", "ROM_2_Calib", "Midair_Calib"]:
    add_edge(GRAPH, "Setup1", tgt)
    add_edge(GRAPH, tgt, "Setup1")

# Setup2 <-> Setup3 / Velocity_Calib
for tgt in ["Setup3", "Velocity_Calib"]:
    add_edge(GRAPH, "Setup2", tgt)
    add_edge(GRAPH, tgt, "Setup2")

# Setup3 <-> TrialCheck
add_edge(GRAPH, "Setup3", "TrialCheck")
add_edge(GRAPH, "TrialCheck", "Setup3")

# TrialCheck -> RunTrials (one-way)
add_edge(GRAPH, "TrialCheck", "RunTrials")

# RunTrials -> GestureChange / SampleChange / EndTrials (controller-triggered; no user nav)
for tgt in ["GestureChange", "SampleChange", "EndTrials"]:
    add_edge(GRAPH, "RunTrials", tgt)

# GestureChange -> RunTrials (one-way, user nav)
add_edge(GRAPH, "GestureChange", "RunTrials")
# SampleChange -> RunTrials (one-way, user nav)
add_edge(GRAPH, "SampleChange", "RunTrials")

ALL_PAGES = sorted({*GRAPH.keys(), *(n for outs in GRAPH.values() for n in outs)})


# ----------- 2) Navigation policy (what appears in the main nav bar) -----------
# Buttons shown per page (subset of GRAPH[page])
MAIN_NAV = {
    "Setup1": ["Setup2"],  # only Setup2 in main nav
    "Setup2": ["Setup3"],  # only Setup3 in main nav
    "Setup3": ["TrialCheck"],  # only TrialCheck in main nav
    "TrialCheck": ["RunTrials"],  # only RunTrials in main nav
    "RunTrials": [],  # controller-driven only
    "GestureChange": ["RunTrials"],
    "SampleChange": ["RunTrials"],
    # Others (calibration & EndTrials) show no forward nav
    "StillCalib": [],
    "ROM_1_Calib": [],
    "ROM_2_Calib": [],
    "Midair_Calib": [],
    "Velocity_Calib": [],
    "EndTrials": [],
}

# Fixed back buttons (no history). Added Setup3->Setup2, Setup2->Setup1. No back for Setup1.
MAIN_BACK = {
    "Setup3": "Setup2",
    "Setup2": "Setup1",
    "StillCalib": "Setup1",
    "ROM_1_Calib": "Setup1",
    "ROM_2_Calib": "Setup1",
    "Midair_Calib": "Setup1",
    "Velocity_Calib": "Setup2",
    # others: no back
}


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

    def __init__(self, name: str, log_bus: LogBus):
        super().__init__()
        self.page_name = name
        self.log_bus = log_bus
        self.nav_buttons = {}
        self.back_button = None

        # Internal status
        self._devices_connected = False
        self._paused = False
        self._recording = False
        self._msg_rate_hz = 0.0
        self._participant_id: str | None = None  # <<< NEW: central store on every page

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
        root.addWidget(title)

        # NEW: a small chip to show current participant
        self._lbl_participant = QtWidgets.QLabel("Participant: —")
        self._lbl_participant.setAlignment(
            QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter
        )
        self._lbl_participant.setStyleSheet("""
            QLabel {
                color: #ddd; background: #111; border: 1px solid #444;
                border-radius: 10px; padding: 4px 8px; font-size: 12px;
            }
        """)

        # Put title + chip on one row
        title_row = QtWidgets.QHBoxLayout()
        title_row.addStretch(1)
        title_row.addWidget(title, 0, QtCore.Qt.AlignCenter)
        title_row.addStretch(1)
        title_row.addWidget(self._lbl_participant, 0, QtCore.Qt.AlignRight)
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

    def participant_id(self) -> str | None:
        """Optional getter for pages that need to read it."""
        return self._participant_id

    # All logging should go through the bus so it appears on every page
    def append_log(self, text: str):
        self.log_bus.log(text)

    # local helper to render a line (don’t call this directly from app code)
    def _append_to_log_widget(self, text: str):
        self.log.appendPlainText(text)
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

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

    # ---------- Internals ----------
    def _on_connect_clicked(self):
        self.append_log("[ui] Connect requested.")
        self.connectRequested.emit()

    def _on_disconnect_clicked(self):
        self.append_log("[ui] Disconnect requested.")
        self.disconnectRequested.emit()

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


# ------------------ Example page subclasses ------------------
class Setup1Page(ExperimenterPage):
    participantIdCommitted = QtCore.Signal(str)

    def __init__(self, name, log_bus):
        super().__init__(name, log_bus)
        self.set_status("Please fit twintig and connect calibration samples.")

        # --- Participant Info Group ---
        input_group = QtWidgets.QGroupBox("Participant Information")
        input_form = QtWidgets.QFormLayout(input_group)

        self.pid_input = QtWidgets.QLineEdit()
        self.pid_input.setPlaceholderText("e.g. 42")
        self.pid_input.setValidator(QtGui.QIntValidator(0, 1000))

        self.pid_set_btn = QtWidgets.QPushButton("Set")
        row = QtWidgets.QWidget()
        row_h = QtWidgets.QHBoxLayout(row)
        row_h.setContentsMargins(0, 0, 0, 0)
        row_h.addWidget(self.pid_input)
        row_h.addWidget(self.pid_set_btn)
        input_form.addRow("Participant ID:", row)

        self.pid_set_btn.clicked.connect(self._commit_pid)
        self.pid_input.returnPressed.connect(self._commit_pid)

        # --- Calibrations Group (button + status dot) ---
        cal_group = QtWidgets.QGroupBox("Calibrations")
        cal_grid = QtWidgets.QGridLayout(cal_group)
        cal_grid.setColumnStretch(0, 0)  # buttons column
        cal_grid.setColumnStretch(1, 1)  # status column can grow a bit but stays small

        # Keep references for later status updates
        self.cal_status_labels = {}

        # Helper to add a row
        def add_cal_row(row: int, label_text: str, page_name: str):
            btn = QtWidgets.QPushButton(f"Open {label_text}")
            # Keep buttons compact, left-aligned
            btn.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
            btn.clicked.connect(lambda: self.navRequested.emit(page_name))

            # Status dot label
            dot = QtWidgets.QLabel("●")
            dot.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
            dot.setStyleSheet("QLabel { color: #d32f2f; font-size: 16px; }")  # red
            dot.setToolTip(f"{label_text} not completed")

            self.cal_status_labels[page_name] = dot

            cal_grid.addWidget(btn, row, 0, 1, 1, alignment=QtCore.Qt.AlignLeft)
            cal_grid.addWidget(dot, row, 1, 1, 1, alignment=QtCore.Qt.AlignLeft)

        add_cal_row(0, "StillCalib", "StillCalib")
        add_cal_row(1, "ROM_1_Calib", "ROM_1_Calib")
        add_cal_row(2, "ROM_2_Calib", "ROM_2_Calib")
        add_cal_row(3, "Midair_Calib", "Midair_Calib")

        # --- Overall layout ---
        wrap = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(wrap)
        v.addSpacing(10)
        v.addWidget(input_group)
        v.addSpacing(10)
        v.addWidget(cal_group)
        v.addStretch()
        self.add_content_widget(wrap)

    @QtCore.Slot()
    def _commit_pid(self):
        text = self.pid_input.text().strip()

        # If a validator is set, this checks Acceptable directly
        if not self.pid_input.hasAcceptableInput():
            self.log_bus.log("Invalid Participant ID. Enter an integer 0–1000.")
            return

        # Optional: normalize to int, then back to str if you want
        pid_int = int(text)
        self.log_bus.log(f"Participant ID set to {pid_int}.")
        self.participantIdCommitted.emit(str(pid_int))  # or emit the int if you prefer

        # (Optional) lock after setting
        # self.pid_input.setReadOnly(True)
        # self.pid_set_btn.setEnabled(False)

    # ---- Public helper to flip a dot red/green ----
    def set_cal_status(self, page_name: str, ok: bool):
        label = self.cal_status_labels.get(page_name)
        if not label:
            return
        if ok:
            label.setStyleSheet("QLabel { color: #2e7d32; font-size: 16px; }")  # green
            label.setToolTip(f"{page_name} completed")
        else:
            label.setStyleSheet("QLabel { color: #d32f2f; font-size: 16px; }")  # red
            label.setToolTip(f"{page_name} not completed")

    # ---- Example slot you can connect to your app’s signal ----
    @QtCore.Slot(str)
    def on_calibration_done(self, page_name: str):
        self.set_cal_status(page_name, True)


class Setup2Page(ExperimenterPage):
    def __init__(self, name, log_bus):
        super().__init__(name, log_bus)
        self.set_status("Stage 2 setup — configure environment & sensors.")

        # Example setup controls unique to Setup2
        grid = QtWidgets.QGridLayout()
        grid.addWidget(QtWidgets.QLabel("Environment preset:"), 0, 0)
        cmb = QtWidgets.QComboBox()
        cmb.addItems(["Lab A", "Lab B", "Field"])
        grid.addWidget(cmb, 0, 1)

        grid.addWidget(QtWidgets.QLabel("Noise filter:"), 1, 0)
        spin = QtWidgets.QSpinBox()
        spin.setRange(0, 100)
        spin.setValue(30)
        grid.addWidget(spin, 1, 1)

        btn_apply = QtWidgets.QPushButton("Apply Settings")
        grid.addWidget(btn_apply, 2, 0, 1, 2)

        # --- Page-specific button (NOT main nav) ---
        # Velocity calibration is reachable from Setup2 but should show Back in its own page.
        btn_vel = QtWidgets.QPushButton("Open Velocity_Calib")
        btn_vel.clicked.connect(lambda: self.navRequested.emit("Velocity_Calib"))

        # Optional: quick link to Setup3 even though it's also in main nav
        # (remove this if you want it strictly in the nav bar only)
        # btn_setup3 = QtWidgets.QPushButton("Go to Setup3")
        # btn_setup3.clicked.connect(lambda: self.navRequested.emit("Setup3"))

        wrap = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(wrap)
        v.addLayout(grid)
        v.addSpacing(12)
        v.addWidget(btn_vel)
        # v.addWidget(btn_setup3)
        v.addStretch(1)

        self.add_content_widget(wrap)


class TrialCheckPage(ExperimenterPage):
    def __init__(self, name, log_bus):
        super().__init__(name, log_bus)
        self.set_status("Verify subject ready & parameters valid.")
        self.chk_ok = QtWidgets.QCheckBox("All checks passed")
        self.add_content_widget(self.chk_ok)


class RunTrialsPage(ExperimenterPage):
    # expose a signal that a controller can use to trigger state transitions
    requestTransition = QtCore.Signal(
        str
    )  # "GestureChange" / "SampleChange" / "EndTrials"

    def __init__(self, name, log_bus):
        super().__init__(name, log_bus)
        self.set_status("Configure and run trials; controller may trigger transitions.")
        row = QtWidgets.QHBoxLayout()
        self.spin_trials = QtWidgets.QSpinBox()
        self.spin_trials.setRange(1, 999)
        self.spin_trials.setValue(10)
        self.btn_start = QtWidgets.QPushButton("Start")
        self.btn_stop = QtWidgets.QPushButton("Stop")
        row.addWidget(QtWidgets.QLabel("Trials:"))
        row.addWidget(self.spin_trials)
        row.addStretch(1)
        row.addWidget(self.btn_start)
        row.addWidget(self.btn_stop)
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("Trial log...")
        wrap = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(wrap)
        v.addLayout(row)
        v.addWidget(self.log, 1)
        self.add_content_widget(wrap)

        # Demo/testing only: hidden controller triggers (you'll call requestTransition from real logic)
        demo = QtWidgets.QHBoxLayout()
        for label in ("GestureChange", "SampleChange", "EndTrials"):
            b = QtWidgets.QPushButton(f"[demo] {label}")
            b.clicked.connect(lambda _, t=label: self.requestTransition.emit(t))
            demo.addWidget(b)
        self.add_content_widget(
            QtWidgets.QLabel("Developer demo triggers below (remove in production):")
        )
        demo_wrap = QtWidgets.QWidget()
        demo_wrap.setLayout(demo)
        self.add_content_widget(demo_wrap)


class GenericPage(ExperimenterPage):
    def __init__(self, name, log_bus):
        super().__init__(name, log_bus)
        lbl = QtWidgets.QLabel("Custom UI for this page goes here.")
        lbl.setAlignment(QtCore.Qt.AlignCenter)
        self.add_content_widget(lbl)


class StillCalibPage(ExperimenterPage):
    calibrationDone = QtCore.Signal(str)  # emits the page key

    def __init__(self, name, log_bus):
        super().__init__(name, log_bus)

        wrap = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(wrap)

        lbl = QtWidgets.QLabel("Run Still Calibration")
        lbl.setAlignment(QtCore.Qt.AlignCenter)

        self.run_btn = QtWidgets.QPushButton("Run Calibration")
        self.run_btn.clicked.connect(self._run_calibration)

        v.addWidget(lbl)
        v.addSpacing(12)
        v.addWidget(self.run_btn, alignment=QtCore.Qt.AlignCenter)
        v.addStretch()

        self.add_content_widget(wrap)

    @QtCore.Slot()
    def _run_calibration(self):
        # Placeholder “work”: disable button, show status, then finish after 1.2s
        self.run_btn.setEnabled(False)
        self.log_bus.log("Running still calibration…")
        QtCore.QTimer.singleShot(1200, self._finish_calibration)

    def _finish_calibration(self):
        self.log_bus.log("Still calibration complete.")
        self.run_btn.setEnabled(True)
        self.calibrationDone.emit("StillCalib")  # key matches Setup1’s map


PAGE_CLASS = {
    "Setup1": Setup1Page,
    "Setup2": Setup2Page,
    "TrialCheck": TrialCheckPage,
    "RunTrials": RunTrialsPage,
    "StillCalib": StillCalibPage,
}


# ---------- Controller ----------
class ExperimenterWindow(QtWidgets.QMainWindow):
    def __init__(self, start_page="Setup1"):
        super().__init__()
        self.setWindowTitle("Twintig Experimenter Window")
        self.resize(1000, 640)

        self.stack = QtWidgets.QStackedWidget()
        self.setCentralWidget(self.stack)
        self.participant_id: str | None = None

        self.log_bus = LogBus()  # <<< one bus for the whole app

        # Create pages
        self.pages = {}
        for name in ALL_PAGES:
            cls = PAGE_CLASS.get(name, GenericPage)
            page = cls(name, self.log_bus)
            page.build_nav(MAIN_NAV.get(name, []), MAIN_BACK.get(name))
            self.pages[name] = page
            self.stack.addWidget(page)

        setup1 = self.pages.get("Setup1")
        if isinstance(setup1, Setup1Page):
            setup1.participantIdCommitted.connect(self.set_participant_id)

        still = self.pages.get("StillCalib")
        if isinstance(still, StillCalibPage) and hasattr(setup1, "on_calibration_done"):
            still.calibrationDone.connect(setup1.on_calibration_done)

        # Build state machine
        self.machine = QStateMachine(self)
        self.states = {}
        for name, page in self.pages.items():
            st = QState()
            st.assignProperty(self.stack, "currentIndex", self.stack.indexOf(page))
            self.states[name] = st
            self.machine.addState(st)

        # Edge emitters (signals per edge)
        class EdgeEmitter(QtCore.QObject):
            fired = QtCore.Signal()

        self._edge_emitters = {}
        for src, outs in GRAPH.items():
            for tgt in outs:
                emitter = EdgeEmitter()
                self._edge_emitters[(src, tgt)] = emitter
                self.states[src].addTransition(emitter.fired, self.states[tgt])

        # Wire visible nav buttons
        for src, page in self.pages.items():
            page.navRequested.connect(lambda target, s=src: self._emit_edge(s, target))
            if src in MAIN_BACK and page.back_button:
                back_tgt = MAIN_BACK[src]
                page.backRequested.connect(
                    lambda s=src, t=back_tgt: self._emit_edge(s, t)
                )

        # Controller-driven transitions from RunTrials
        run_trials = self.pages.get("RunTrials")
        if isinstance(run_trials, RunTrialsPage):
            run_trials.requestTransition.connect(
                lambda tgt: self._emit_edge("RunTrials", tgt)
            )

        self.machine.setInitialState(self.states[start_page])
        self.machine.start()

        # ---- Example controller wiring for the shared controls ----
        def handle_connect():
            self.log_bus.log("[ctrl] Connecting devices…")
            for p in self.pages.values():
                p.set_devices_connected(True)
            self.log_bus.log("[ctrl] Devices connected.")

        def handle_disconnect():
            self.log_bus.log("[ctrl] Disconnecting devices…")
            for p in self.pages.values():
                p.set_devices_connected(False)
                p.set_recording(False)
            self.log_bus.log("[ctrl] Devices disconnected.")

        def handle_pause(paused: bool):
            self.log_bus.log(f"[ctrl] Pipeline {'paused' if paused else 'resumed'}.")

        for page in self.pages.values():
            page.connectRequested.connect(handle_connect)
            page.disconnectRequested.connect(handle_disconnect)
            page.pauseToggled.connect(handle_pause)

        # Demo telemetry (remove in production)
        self._demo_timer = QtCore.QTimer(self)
        self._demo_timer.timeout.connect(self._demo_tick)
        self._demo_timer.start(1000)

    @QtCore.Slot(str)
    def set_participant_id(self, pid_text: str):
        self.participant_id = pid_text

        # Push to all pages (all inherit ExperimenterPage, so they have the setter)
        for p in self.pages.values():
            p.set_participant_id(pid_text)

    def _emit_edge(self, src: str, tgt: str):
        if (src, tgt) in self._edge_emitters:
            self._edge_emitters[(src, tgt)].fired.emit()

    def _demo_tick(self):
        import random

        # Simple fake rate: nonzero only if connected
        connected = any(p._devices_connected for p in self.pages.values())
        hz = random.uniform(5.0, 30.0) if connected else 0.0
        for p in self.pages.values():
            p.set_msg_rate(hz)
