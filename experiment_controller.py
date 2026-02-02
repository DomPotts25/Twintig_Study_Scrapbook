from PySide6 import QtCore, QtGui, QtStateMachine, QtWidgets

from experiment_factors import Gestures, SampleGroup, StudyPhases, Velocity
from Pages.setup_page import SetupPage, TrialCheckPage
from Pages.experimenter_page import ExperimenterPage, LogBus
from Pages.trial_pages import RunTrialsPage
from Pages.calibration_pages import StillCalibPage, GenericPage
from twintig_interface import TwintigInterface

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
    "TrialCheck": TrialCheckPage,
    "RunTrials": RunTrialsPage,
    "StillCalib": StillCalibPage,
}

# ---------- Controller ----------
class ExperimenterWindow(QtWidgets.QMainWindow):

    pageEntered = QtCore.Signal(str)  # page name

    def __init__(self, start_page="Setup"):
        super().__init__()

        self.logger = TwintigInterface()
        self.setWindowTitle("Twintig Experimenter Window")
        self.resize(1000, 640)

        self.stack = QtWidgets.QStackedWidget()
        self.setCentralWidget(self.stack)
        
        self.participant_id: str | None = None

        self.curr_study_phase = StudyPhases.SETUP
        self.curr_study_phase_start_ts = 0
        self.prev_study_phase_end_ts = 0

        self.curr_trial_id = -1
        self.curr_sample_group = None
        self.curr_sample_id = -1
        self.curr_sample_name = None
        self.curr_gesture = None
        self.curr_trial_valid = True
        
        self.gesture_sequence: list[Gestures] = []
        self.sample_sequence: list[SampleGroup] = []

        # Experimenter log window output
        self.log_bus = LogBus()

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
            setup.gestureOrderCommitted.connect(self.set_gesture_sequence)
            setup.sampleOrderCommitted.connect(self.set_sample_sequence)

        still = self.pages.get("StillCalib")
        if isinstance(still, StillCalibPage) and hasattr(setup, "on_calibration_done"):
            still.calibrationDone.connect(setup.on_calibration_done)

        for p in self.pages.values():
            if isinstance(p, ExperimenterPage):
                p.set_twintig_interface(self.logger)

        # Build state machine
        self.machine = QStateMachine(self)
        self.states = {}
        for name, page in self.pages.items():
            st = QState()
            st.assignProperty(self.stack, "currentIndex", self.stack.indexOf(page))
            self.states[name] = st
            self.machine.addState(st)

        # connect to currentChanged:
        self.stack.currentChanged.connect(self._on_page_changed)

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
            page.navRequested.connect(
                lambda target, s=src: self._handle_nav_click(s, target)
            )
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
            page.studyPhaseRequested.connect(self.set_study_phase)
        
        for p in self.pages.values():
            if isinstance(p, ExperimenterPage):
                p.set_twintig_interface(self.logger)

        # After pages exist, push initial context into the chips
        self._broadcast_experiment_context()

    @QtCore.Slot(int)
    def _on_page_changed(self, index: int):
        page = self.stack.widget(index)
        if not isinstance(page, ExperimenterPage):
            return

        name = page.page_name  # set in ExperimenterPage.__init__
        self.log_bus.log(f"[ctrl] Entered page: {name}")
        self.pageEntered.emit(name)  # if this needs to be observed elsewhere

        if name == "RunTrials":
            self.set_study_phase(StudyPhases.TRIAL)
        elif name == "StillCalib":
            self.set_study_phase(StudyPhases.SETUP)
        elif name == "Setup":
            self.set_study_phase(StudyPhases.SETUP)
        # TODO add other pages

    def _broadcast_experiment_context(self):
            """Push current controller state into all ExperimenterPage headers."""
            for p in self.pages.values():
                if isinstance(p, ExperimenterPage):
                    p.set_participant_id(self.participant_id)
                    p.set_study_phase(self.curr_study_phase)
                    p.set_sample_group(self.curr_sample_group)
                    p.set_sample_id(self.curr_sample_id)
                    p.set_sample_name(self.curr_sample_name)
                    p.set_trial_id(self.curr_trial_id)
                    p.set_gesture(self.curr_gesture)

    def start_next_trial(self, trial_id: int, sample_group: SampleGroup,
                         sample_id: int, sample_name: str):
        self.set_study_phase(StudyPhases.RUN_TRIALS)
        self.set_trial_id(trial_id)
        self.set_sample_group(sample_group)
        self.set_sample_id(sample_id)
        self.set_sample_name(sample_name)

    @QtCore.Slot(str)
    def set_participant_id(self, pid_text: str):
        self.participant_id = pid_text
        self._broadcast_experiment_context()

    @QtCore.Slot(object)
    def set_study_phase(self, phase: StudyPhases):
        self.curr_study_phase = phase
        self._broadcast_experiment_context()
        self.log_bus.log(f"[ctrl] Study phase -> {phase}")

    def set_sample_group(self, group: SampleGroup):
        self.curr_sample_group = group
        self._broadcast_experiment_context()

    def set_sample_id(self, sample_id: int):
        self.curr_sample_id = sample_id
        self._broadcast_experiment_context()

    def set_sample_name(self, name: str | None):
        self.curr_sample_name = name
        self._broadcast_experiment_context()

    def set_trial_id(self, trial_id: int):
        self.curr_trial_id = trial_id
        self._broadcast_experiment_context()

    def set_gesture(self, gesture: Gestures):
        self.curr_gesture = gesture
        self._broadcast_experiment_context()

    @QtCore.Slot(object)
    def set_gesture_sequence(self, gestures):
        """Store and log the gesture order for this participant."""
        self.gesture_sequence = list(gestures or [])

        labels = []
        for g in self.gesture_sequence:
            if g is None:
                continue
            labels.append(getattr(g, "name", str(g)))

        if labels:
            self.log_bus.log(
                "[ctrl] Gesture order for this participant: " + " → ".join(labels)
            )
        else:
            self.log_bus.log("[ctrl] Gesture order for this participant: (empty)")

        self.set_gesture(self.gesture_sequence[0])

    @QtCore.Slot(object)
    def set_sample_sequence(self, samples):
        """Store and log the sample order for this participant."""
        self.sample_sequence = list(samples or [])

        labels = []
        for s in self.sample_sequence:
            if s is None:
                continue
            labels.append(getattr(s, "name", str(s)))

        if labels:
            self.log_bus.log(
                "[ctrl] Sample order for this participant: " + " → ".join(labels)
            )
        else:
            self.log_bus.log("[ctrl] Sample order for this participant: (empty)")
        
        self.set_sample_group(self.sample_sequence[0])

    def _emit_edge(self, src: str, tgt: str):
        if (src, tgt) in self._edge_emitters:
            self._edge_emitters[(src, tgt)].fired.emit()

    def _handle_nav_click(self, src: str, tgt: str):
        # 1) refresh all header chips so the destination page is up to date
        self._broadcast_experiment_context()

        # 2) perform the actual state transition
        self._emit_edge(src, tgt)
