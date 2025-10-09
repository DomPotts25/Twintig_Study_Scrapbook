from PySide6 import QtCore, QtGui, QtStateMachine, QtWidgets

from experiment_factors import Gestures, SampleGroup, StudyPhases, Velocity
from Pages.setup_page import SetupPage, TrialCheckPage
from Pages.experimenter_page import ExperimenterPage, LogBus
from Pages.trial_pages import RunTrialsPage
from Pages.calibration_pages import StillCalibPage, GenericPage

QState = QtStateMachine.QState
QStateMachine = QtStateMachine.QStateMachine

# ------------------ 1) Full directed graph (legal transitions) ------------------
def add_edge(g, a, b):
    g.setdefault(a, []).append(b)


GRAPH = {}

# Setup <-> TrialCheck / StillCalib / ROM_1_Calib / ROM_2_Calib / Midair_Calib
for tgt in ["TrialCheck", "StillCalib", "ROM_1_Calib", "ROM_2_Calib", "Midair_Calib", "Velocity_Calib"]:
    add_edge(GRAPH, "Setup", tgt)
    add_edge(GRAPH, tgt, "Setup")

# Setup3 <-> TrialCheck
# add_edge(GRAPH, "Setup3", "TrialCheck")
# add_edge(GRAPH, "TrialCheck", "Setup3")

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
    "Setup": ["TrialCheck"],  # only Setup2 in main nav
    #"Setup2": ["Setup3"],  # only Setup3 in main nav
    #"Setup3": ["TrialCheck"],  # only TrialCheck in main nav
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

# Fixed back buttons (no history). Added Setup3->Setup2, Setup2->Setup. No back for Setup.
MAIN_BACK = {
    #"Setup3": "Setup2",
    "TrialCheck": "Setup",
    "StillCalib": "Setup",
    "ROM_1_Calib": "Setup",
    "ROM_2_Calib": "Setup",
    "Midair_Calib": "Setup",
    "Velocity_Calib": "Setup",
    # others: no back
}


PAGE_CLASS = {
    "Setup": SetupPage,
    #"Setup2": Setup2Page,
    "TrialCheck": TrialCheckPage,
    "RunTrials": RunTrialsPage,
    "StillCalib": StillCalibPage,
}


# ---------- Controller ----------
class ExperimenterWindow(QtWidgets.QMainWindow):
    def __init__(self, start_page="Setup"):
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

        setup = self.pages.get("Setup")
        if isinstance(setup, SetupPage):
            setup.participantIdCommitted.connect(self.set_participant_id)

        still = self.pages.get("StillCalib")
        if isinstance(still, StillCalibPage) and hasattr(setup, "on_calibration_done"):
            still.calibrationDone.connect(setup.on_calibration_done)

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
