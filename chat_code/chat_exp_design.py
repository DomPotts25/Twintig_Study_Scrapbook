# experiment_flow_nav_policy.py
# Works with PySide6 (Qt6) or PyQt5 (Qt5)
from PySide6 import QtCore, QtStateMachine, QtWidgets

QState = QtStateMachine.QState
QStateMachine = QtStateMachine.QStateMachine

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

# RunTrials -> GestureChange / SampleChange / EndTrials (controller triggered, not user nav)
for tgt in ["GestureChange", "SampleChange", "EndTrials"]:
    add_edge(GRAPH, "RunTrials", tgt)

# GestureChange -> RunTrials (one-way, main nav)
add_edge(GRAPH, "GestureChange", "RunTrials")

# SampleChange -> RunTrials (one-way, main nav)
add_edge(GRAPH, "SampleChange", "RunTrials")

ALL_PAGES = sorted({*GRAPH.keys(), *(n for outs in GRAPH.values() for n in outs)})


# ----------- 2) Navigation policy (what appears in the main nav bar) -----------
# Buttons shown per page (subset of GRAPH[page])
MAIN_NAV = {
    # Setup1 main nav shows only Setup2
    "Setup1": ["Setup2"],
    # Setup2 main nav shows Setup3
    "Setup2": ["Setup3"],
    # Setup3 main nav shows TrialCheck
    "Setup3": ["TrialCheck"],
    # TrialCheck main nav shows RunTrials
    "TrialCheck": ["RunTrials"],
    # RunTrials: no main-nav buttons for controller-driven transitions
    "RunTrials": [],
    # GestureChange/SampleChange: show RunTrials in main nav
    "GestureChange": ["RunTrials"],
    "SampleChange": ["RunTrials"],
    # Calibration pages & EndTrials have no forward nav in main bar
    "StillCalib": [],
    "ROM_1_Calib": [],
    "ROM_2_Calib": [],
    "Midair_Calib": [],
    "Velocity_Calib": [],
    "EndTrials": [],
}

# Back button destination when present on a page (appears in main nav)
MAIN_BACK = {
    # On these pages, show a Back button to the parent in your plan
    "StillCalib": "Setup1",
    "ROM_1_Calib": "Setup1",
    "ROM_2_Calib": "Setup1",
    "Midair_Calib": "Setup1",
    "Velocity_Calib": "Setup2",
    # (If you later want a back on other pages, add them here)
}


# ------------------ Base page with black theme + nav bar ------------------
class ExperimenterPage(QtWidgets.QWidget):
    # Signals your controller can handle centrally
    connectRequested    = QtCore.Signal()
    disconnectRequested = QtCore.Signal()
    pauseToggled        = QtCore.Signal(bool)   # True = paused, False = resumed
    recordingToggled    = QtCore.Signal(bool)   # optional: emit when you change recording state

    # Emitted by nav buttons (already used in your app)
    backRequested = QtCore.Signal()
    navRequested  = QtCore.Signal(str)

    def __init__(self, name: str):
        super().__init__()
        self.page_name = name
        self.nav_buttons = {}
        self.back_button = None

        # Internal status state
        self._devices_connected = False
        self._paused = False
        self._recording = False
        self._msg_rate_hz = 0.0

        # Dark theme (matches your app)
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
        root.setContentsMargins(16,16,16,16)
        root.setSpacing(10)

        # Title
        title = QtWidgets.QLabel(name)
        title.setAlignment(QtCore.Qt.AlignCenter)
        title.setStyleSheet("font-size: 26px; font-weight: 600;")
        root.addWidget(title)

        # Page status (you can set per-page hints here)
        self.status_label = QtWidgets.QLabel("")
        self.status_label.setAlignment(QtCore.Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #aaa;")
        root.addWidget(self.status_label)

        # --------- UNIVERSAL STATUS + LOG PANEL (shared on all pages) ----------
        panel = QtWidgets.QGroupBox("System Status & Log")
        panel_layout = QtWidgets.QVBoxLayout(panel)

        # Row of indicators
        ind_row = QtWidgets.QHBoxLayout()

        # Tiny "LED" helpers
        def make_led(label_text):
            dot = QtWidgets.QLabel("●")
            dot.setFixedWidth(14)
            dot.setAlignment(QtCore.Qt.AlignCenter)
            txt = QtWidgets.QLabel(label_text)
            wrap = QtWidgets.QHBoxLayout()
            w = QtWidgets.QWidget()
            wrap.addWidget(dot)
            wrap.addWidget(txt)
            wrap.setSpacing(6)
            w.setLayout(wrap)
            return w, dot, txt

        self._led_devices_wrap, self._led_devices, self._lbl_devices = make_led("Devices: Disconnected")
        self._led_msg_wrap,     self._led_msg,     self._lbl_msg     = make_led("Msg rate: 0.0 Hz")
        self._led_rec_wrap,     self._led_rec,     self._lbl_rec     = make_led("Recording: Off")

        ind_row.addWidget(self._led_devices_wrap)
        ind_row.addSpacing(12)
        ind_row.addWidget(self._led_msg_wrap)
        ind_row.addSpacing(12)
        ind_row.addWidget(self._led_rec_wrap)
        ind_row.addStretch(1)
        panel_layout.addLayout(ind_row)

        # Log output
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("System log output…")
        self.log.setMinimumHeight(120)
        panel_layout.addWidget(self.log)

        # Controls row (Connect / Disconnect / Pause-Resume)
        ctrl_row = QtWidgets.QHBoxLayout()
        self.btn_connect = QtWidgets.QPushButton("Connect Devices")
        self.btn_disconnect = QtWidgets.QPushButton("Disconnect Devices")
        self.btn_pause_resume = QtWidgets.QPushButton("Pause Experiment")  # toggles to Resume

        self.btn_connect.clicked.connect(self._on_connect_clicked)
        self.btn_disconnect.clicked.connect(self._on_disconnect_clicked)
        self.btn_pause_resume.clicked.connect(self._on_pause_resume_clicked)

        ctrl_row.addWidget(self.btn_connect)
        ctrl_row.addWidget(self.btn_disconnect)
        ctrl_row.addStretch(1)
        ctrl_row.addWidget(self.btn_pause_resume)

        panel_layout.addLayout(ctrl_row)

        root.addWidget(panel)  # <-- panel lives above page-specific content

        # Content area (unique per subclass)
        self.content = QtWidgets.QWidget()
        self.content_layout = QtWidgets.QVBoxLayout(self.content)
        root.addWidget(self.content, 1)

        # Nav bar (your policy-driven buttons live here)
        self.nav_bar = QtWidgets.QHBoxLayout()
        root.addLayout(self.nav_bar)

        # Initialize indicators
        self._refresh_indicators()

    # --------------- Public helpers you can call from anywhere ----------------
    def set_status(self, text: str): self.status_label.setText(text)

    def add_content_widget(self, w: QtWidgets.QWidget): self.content_layout.addWidget(w)

    def append_log(self, text: str):
        self.log.appendPlainText(text)
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

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
        self.btn_pause_resume.setText("Resume Experiment" if paused else "Pause Experiment")
        # (You might also want to gray out certain page widgets when paused)
        # e.g., self.content.setEnabled(not paused)
        # We don't emit here, because set_paused is generally a controller-driven update.

    # Called by controller to build policy-driven nav
    def build_nav(self, main_targets: list[str], back_target: str | None):
        # Clear
        for i in reversed(range(self.nav_bar.count())):
            item = self.nav_bar.itemAt(i)
            w = item.widget()
            if w: w.setParent(None)
        self.nav_buttons.clear()
        self.back_button = None

        # Back
        if back_target:
            btn = QtWidgets.QPushButton("⬅ Back")
            btn.clicked.connect(self.backRequested.emit)
            self.nav_bar.addWidget(btn)
            self.back_button = btn

        self.nav_bar.addStretch(1)

        # Forward
        for t in main_targets:
            b = QtWidgets.QPushButton(f"Go to {t} ➜")
            b.clicked.connect(lambda _, target=t: self.navRequested.emit(target))
            self.nav_bar.addWidget(b)
            self.nav_buttons[t] = b

    # -------------------------- Internal handlers -----------------------------
    def _on_connect_clicked(self):
        self.append_log("[ui] Connect requested.")
        self.connectRequested.emit()

    def _on_disconnect_clicked(self):
        self.append_log("[ui] Disconnect requested.")
        self.disconnectRequested.emit()

    def _on_pause_resume_clicked(self):
        self._paused = not self._paused
        self.btn_pause_resume.setText("Resume Experiment" if self._paused else "Pause Experiment")
        self.append_log(f"[ui] {'Paused' if self._paused else 'Resumed'} experiment.")
        self.pauseToggled.emit(self._paused)

    def _refresh_indicators(self):
        # LED colors
        def set_led(led_label: QtWidgets.QLabel, on: bool, on_color="#0f7", off_color="#444"):
            led_label.setStyleSheet(f"color: {on_color if on else off_color}; font-size: 18px;")
        # Devices
        set_led(self._led_devices, self._devices_connected)
        self._lbl_devices.setText(f"Devices: {'Connected' if self._devices_connected else 'Disconnected'}")
        # Msg rate
        set_led(self._led_msg, self._msg_rate_hz > 0.1)
        self._lbl_msg.setText(f"Msg rate: {self._msg_rate_hz:.1f} Hz")
        # Recording
        set_led(self._led_rec, self._recording, on_color="#fa0")
        self._lbl_rec.setText(f"Recording: {'On' if self._recording else 'Off'}")


# ------------------ Example page subclasses ------------------
class Setup1Page(ExperimenterPage):
    def __init__(self, name):
        super().__init__(name)
        self.set_status("Connect and verify experiment hardware.")

        # --- Page-specific buttons (NOT main nav) ---
        # These will navigate directly to calibration pages.
        cal_group = QtWidgets.QGroupBox("Calibrations")
        cal_v = QtWidgets.QVBoxLayout(cal_group)

        btn_still = QtWidgets.QPushButton("Open StillCalib")
        btn_rom1  = QtWidgets.QPushButton("Open ROM_1_Calib")
        btn_rom2  = QtWidgets.QPushButton("Open ROM_2_Calib")
        btn_mid   = QtWidgets.QPushButton("Open Midair_Calib")

        # Hook to base-class signal; controller already listens for this
        btn_still.clicked.connect(lambda: self.navRequested.emit("StillCalib"))
        btn_rom1.clicked.connect(lambda: self.navRequested.emit("ROM_1_Calib"))
        btn_rom2.clicked.connect(lambda: self.navRequested.emit("ROM_2_Calib"))
        btn_mid.clicked.connect(lambda: self.navRequested.emit("Midair_Calib"))

        cal_v.addWidget(btn_still)
        cal_v.addWidget(btn_rom1)
        cal_v.addWidget(btn_rom2)
        cal_v.addWidget(btn_mid)

        # --- Layout ---
        wrap = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(wrap)
        v.addSpacing(12)
        v.addWidget(cal_group)

        self.add_content_widget(wrap)

class Setup2Page(ExperimenterPage):
    def __init__(self, name):
        super().__init__(name)
        self.set_status("Stage 2 setup — configure environment & sensors.")

        # Example setup controls unique to Setup2
        grid = QtWidgets.QGridLayout()
        grid.addWidget(QtWidgets.QLabel("Environment preset:"), 0, 0)
        cmb = QtWidgets.QComboBox(); cmb.addItems(["Lab A", "Lab B", "Field"])
        grid.addWidget(cmb, 0, 1)

        grid.addWidget(QtWidgets.QLabel("Noise filter:"), 1, 0)
        spin = QtWidgets.QSpinBox(); spin.setRange(0, 100); spin.setValue(30)
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
    def __init__(self, name):
        super().__init__(name)
        self.set_status("Verify subject ready & parameters valid.")
        self.chk_ok = QtWidgets.QCheckBox("All checks passed")
        self.add_content_widget(self.chk_ok)


class RunTrialsPage(ExperimenterPage):
    # expose a signal that a controller can use to trigger state transitions
    requestTransition = QtCore.Signal(
        str
    )  # "GestureChange" / "SampleChange" / "EndTrials"

    def __init__(self, name):
        super().__init__(name)
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
    def __init__(self, name):
        super().__init__(name)
        lbl = QtWidgets.QLabel("Custom UI for this page goes here.")
        lbl.setAlignment(QtCore.Qt.AlignCenter)
        self.add_content_widget(lbl)


# Map specialized pages
PAGE_CLASS = {
    "Setup1": Setup1Page,
    "Setup2": Setup2Page,
    "TrialCheck": TrialCheckPage,
    "RunTrials": RunTrialsPage,
    # Add more concrete subclasses as you flesh them out
}


# ------------------ Controller (states, policy, and wiring) ------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, start_page="Setup1"):
        super().__init__()
        self.setWindowTitle("Experiment Flow — Policy-driven Nav")
        self.resize(1000, 640)

        self.stack = QtWidgets.QStackedWidget()
        self.setCentralWidget(self.stack)

        # Create pages
        self.pages = {}
        for name in ALL_PAGES:
            cls = PAGE_CLASS.get(name, GenericPage)
            page = cls(name)
            # Build nav per policy
            page.build_nav(MAIN_NAV.get(name, []), MAIN_BACK.get(name))
            self.pages[name] = page
            self.stack.addWidget(page)

        def handle_connect():
            # TODO: your real connect logic
            for p in self.pages.values():
                p.append_log("[ctrl] Connecting devices…")
            # Simulate success
            for p in self.pages.values():
                p.set_devices_connected(True)
                p.append_log("[ctrl] Devices connected.")

        def handle_disconnect():
            # TODO: your real disconnect logic
            for p in self.pages.values():
                p.append_log("[ctrl] Disconnecting devices…")
                p.set_devices_connected(False)
                p.set_recording(False)
            for p in self.pages.values():
                p.append_log("[ctrl] Devices disconnected.")

        def handle_pause(paused: bool):
            # TODO: pause/resume your pipelines here
            for p in self.pages.values():
                p.append_log(f"[ctrl] Pipeline {'paused' if paused else 'resumed'}.")

        # Subscribe every page once
        for page in self.pages.values():
            page.connectRequested.connect(handle_connect)
            page.disconnectRequested.connect(handle_disconnect)
            page.pauseToggled.connect(handle_pause)

        # Build state machine
        self.machine = QStateMachine(self)
        self.states = {}
        for name, page in self.pages.items():
            st = QState()
            st.assignProperty(self.stack, "currentIndex", self.stack.indexOf(page))
            self.states[name] = st
            self.machine.addState(st)

        # 2a) Wire MAIN NAV buttons (visible)
        for src, page in self.pages.items():
            page.navRequested.connect(lambda target, s=src: self._go(s, target))
            if page.back_button and src in MAIN_BACK:
                page.backRequested.connect(lambda s=src: self._go(s, MAIN_BACK[s]))

        # 2b) Wire hidden/controller transitions (no buttons)
        for src, outs in GRAPH.items():
            for tgt in outs:
                # we add a transition object anyway; triggers will be custom signals
                # Create an "internal" transition signal per page pair
                signal_name = f"__sig_{src}_TO_{tgt}"
                if not hasattr(self, signal_name):
                    setattr(self, signal_name, QtCore.Signal)
        # Instead of dynamic signals, we’ll route via a helper method that calls .postEvent on the machine.
        # Simpler approach: use addTransition on a dummy QObject signal per edge:
        self._edge_emitters = {}  # (src,tgt) -> QObject with 'fired' signal

        class EdgeEmitter(QtCore.QObject):
            fired = QtCore.Signal()

        for src, outs in GRAPH.items():
            for tgt in outs:
                emitter = EdgeEmitter()
                self._edge_emitters[(src, tgt)] = emitter
                self.states[src].addTransition(emitter.fired, self.states[tgt])

        # Hook the visible buttons into emitters
        for src, page in self.pages.items():
            for tgt in MAIN_NAV.get(src, []):
                btn = page.nav_buttons.get(tgt)
                if btn:
                    btn.clicked.connect(self._edge_emitters[(src, tgt)].fired)

            # Back button routed to its target
            if src in MAIN_BACK:
                back_tgt = MAIN_BACK[src]
                if page.back_button:
                    page.backRequested.connect(
                        self._edge_emitters[(src, back_tgt)].fired
                    )

        # Hook controller-driven transitions (RunTrials → X)
        run_trials = self.pages.get("RunTrials")
        if isinstance(run_trials, RunTrialsPage):

            def on_request(tgt):
                if ("RunTrials", tgt) in self._edge_emitters:
                    self._edge_emitters[("RunTrials", tgt)].fired.emit()

            run_trials.requestTransition.connect(on_request)

        # For pages like GestureChange/SampleChange that have a visible "RunTrials" nav:
        for src in ("GestureChange", "SampleChange"):
            btn = self.pages[src].nav_buttons.get("RunTrials")
            if btn:
                btn.clicked.connect(self._edge_emitters[(src, "RunTrials")].fired)

        self.machine.setInitialState(self.states[start_page])
        self.machine.start()

    # Optional: helper to do guard checks before moving (not strictly required here)
    def _go(self, src, tgt):
        # Example of a guard: require TrialCheck checkbox before allowing RunTrials
        if src == "TrialCheck" and tgt == "RunTrials":
            page = self.pages["TrialCheck"]
            if isinstance(page, TrialCheckPage) and not page.chk_ok.isChecked():
                QtWidgets.QMessageBox.warning(
                    self,
                    "Blocked",
                    "Please tick 'All checks passed' before continuing.",
                )
                return
        # If passed, emit the transition
        self._edge_emitters[(src, tgt)].fired.emit()


def main():
    import sys

    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow(start_page="Setup1")
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
