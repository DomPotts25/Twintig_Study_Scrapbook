from Pages.experimenter_page import ExperimenterPage
from PySide6 import QtCore, QtGui, QtStateMachine, QtWidgets

from experiment_factors import Gestures, SampleGroup, StudyPhases, Velocity
from Tools.velocity_calib_analyser import TrialBlockForceAnalyser

import os

QState = QtStateMachine.QState
QStateMachine = QtStateMachine.QStateMachine

class SetupPage(ExperimenterPage):
    participantIdCommitted = QtCore.Signal(str)
    gestureOrderCommitted = QtCore.Signal(object)  # list[Gestures]
    sampleOrderCommitted = QtCore.Signal(object)   # list[SampleGroup]

    def __init__(self, name, log_bus):
        super().__init__(name, log_bus)
        self.set_status("Please fit twintig and connect calibration samples.")

        # Track calibration completion flags for the 5 required calibrations
        self._cal_done: dict[str, bool] = {
            "StillCalib": True,
            "ROM_1_Calib": True,
            "ROM_2_Calib": True,
            "Midair_Calib": True,
            "Velocity_Calib": True,
        }

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
        def add_cal_row(row: int, label_text: str, page_name: str, _cal_grid):
            btn = QtWidgets.QPushButton(f"Open {label_text}")
            btn.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
            btn.clicked.connect(lambda: self.navRequested.emit(page_name))

            dot = QtWidgets.QLabel("●")
            dot.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
            dot.setStyleSheet("QLabel { color: #d32f2f; font-size: 16px; }")  # red
            dot.setToolTip(f"{label_text} not completed")

            self.cal_status_labels[page_name] = dot

            _cal_grid.addWidget(btn, row, 0, 1, 1, alignment=QtCore.Qt.AlignLeft)
            _cal_grid.addWidget(dot, row, 1, 1, 1, alignment=QtCore.Qt.AlignLeft)

        add_cal_row(0, "StillCalib", "StillCalib", cal_grid)
        add_cal_row(1, "ROM_1_Calib", "ROM_1_Calib", cal_grid)
        add_cal_row(2, "ROM_2_Calib", "ROM_2_Calib", cal_grid)
        add_cal_row(3, "Midair_Calib", "Midair_Calib", cal_grid)
        add_cal_row(4, "Velocity_Calib", "Velocity_Calib", cal_grid)

        # --- Gesture Info Group ---
        gesture_group = QtWidgets.QGroupBox("Gesture Assignments")
        gesture_row = QtWidgets.QHBoxLayout(gesture_group)
        gesture_row.setContentsMargins(9, 9, 9, 9)
        gesture_row.setSpacing(8)

        # Keep references
        self.gesture_combos: list[QtWidgets.QComboBox] = []

        # Build master option list from the Enum
        # Adjust display as you prefer (name vs value). Here we use .name for label, enum in userData
        try:
            self._gesture_options = [(g.name, g) for g in Gestures]
        except Exception:
            # Fallback if your Enum stores strings in .value
            self._gesture_options = [(str(g.value), g) for g in Gestures]

        # Create one combo per enum member (blank initially)
        for _ in self._gesture_options:
            cb = QtWidgets.QComboBox()
            cb.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
            cb.setMinimumWidth(140)

            # Add a blank placeholder as index 0
            cb.addItem("— Select —", userData=None)

            # Add all gestures (temporarily; we'll prune on change)
            for label, enum_member in self._gesture_options:
                cb.addItem(label, userData=enum_member)

            # When changed, rebuild all others to keep selections unique
            cb.currentIndexChanged.connect(self._on_gesture_changed)

            self.gesture_combos.append(cb)
            gesture_row.addWidget(cb)


        # --- Samples --- 
        sample_group = QtWidgets.QGroupBox("Sample Setup")
        sample_group_row = QtWidgets.QHBoxLayout(sample_group)

        sample_group_row.setContentsMargins(9, 9, 9, 9)
        sample_group_row.setSpacing(8)

        # same functionality as gestures but for samples
        self.sample_combos: list[QtWidgets.QComboBox] = []

        try:
            self._sample_options = [(s.name, s) for s in SampleGroup]
        except Exception:
            self._sample_options = [(s.name, s) for s in SampleGroup]

        for _ in self._sample_options:
            cb = QtWidgets.QComboBox()
            cb.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
            cb.setMinimumWidth(140)
            cb.addItem("— Select —", userData=None)

            for label, enum_member in self._sample_options:
                cb.addItem(label, userData=enum_member)
            
            cb.currentIndexChanged.connect(self._on_sample_combo_box_changed)
            self.sample_combos.append(cb)
            sample_group_row.addWidget(cb)

        # --- Overall layout ---
        wrap = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(wrap)
        v.addSpacing(10)
        v.addWidget(input_group)
        v.addSpacing(10)
        v.addWidget(cal_group)
        v.addSpacing(10)
        v.addWidget(gesture_group)
        v.addSpacing(10)
        v.addWidget(sample_group)
        # v.addSpacing(10)
        # v.addWidget(vel_cal_group)
        v.addStretch()
        self.add_content_widget(wrap)

    # ---- NAV OVERRIDE: intercept “Go to …” from Setup ----
    def build_nav(self, main_targets: list[str], back_target: str | None):
        super().build_nav(main_targets, back_target)
        # Intercept all forward navs from Setup; only allow if ready
        for tgt, btn in self.nav_buttons.items():
            try:
                btn.clicked.disconnect()  # remove default “emit navRequested”
            except TypeError:
                pass
            btn.clicked.connect(lambda _, t=tgt: self._attempt_nav(t))
    
    def _attempt_nav(self, target: str):
        ok, issues = self._is_ready_to_proceed()
        if ok:
            # Only push assignments when leaving Setup towards TrialCheck
            if target == "TrialCheck":
                # Gestures: list[Gestures]
                gesture_order = [cb.currentData() for cb in self.gesture_combos]
                gesture_order = [g for g in gesture_order if g is not None]

                # Samples: list[SampleGroup]
                sample_order = [cb.currentData() for cb in self.sample_combos]
                sample_order = [s for s in sample_order if s is not None]

                self.gestureOrderCommitted.emit(gesture_order)
                self.sampleOrderCommitted.emit(sample_order)

            # Proceed with normal navigation
            self.navRequested.emit(target)
        else:
            QtWidgets.QMessageBox.warning(
                self,
                "Setup incomplete",
                "You can’t proceed yet:\n\n• " + "\n• ".join(issues),
                QtWidgets.QMessageBox.Ok,
            )

        # ---- Readiness checks ----
    def _is_ready_to_proceed(self) -> tuple[bool, list[str]]:
        issues: list[str] = []

        # 1) Devices connected
        if not self._devices_connected:
            issues.append("Devices are not connected.")

        # 2) Participant ID is an integer (we store it globally on every page)
        pid = self._participant_id
        if pid is None or not pid.isdigit():
            issues.append("Participant ID is not set to an integer.")

        # 3) All calibration pages completed
        missing_cals = [name for name, done in self._cal_done.items() if not done]
        if missing_cals:
            issues.append("Calibrations incomplete: " + ", ".join(missing_cals))

        # 4) All gesture & sample comboboxes must have a selection
        if any(cb.currentData() is None for cb in self.gesture_combos):
            issues.append("All gesture assignments must be selected.")

        if any(cb.currentData() is None for cb in self.sample_combos):
            issues.append("All sample assignments must be selected.")

        return (len(issues) == 0, issues)

    def _current_gesture_selections(self) -> set:
        """Return the set of enum members currently selected across all combos (excluding blanks)."""
        selected = set()
        for cb in self.gesture_combos:
            data = cb.currentData()
            if data is not None:
                selected.add(data)
        return selected

    @QtCore.Slot()
    def _on_gesture_changed(self):
        """Rebuild each combo's items so already-chosen gestures are unavailable in others."""
        selected = self._current_gesture_selections()

        # Rebuild each combo while preserving its current choice
        for cb in self.gesture_combos:
            keep = cb.currentData()  # enum member or None

            cb.blockSignals(True)
            try:
                current_enum = keep
                cb.clear()
                cb.addItem("— Select —", userData=None)

                for label, enum_member in self._gesture_options:
                    if enum_member == current_enum or enum_member not in selected:
                        cb.addItem(label, userData=enum_member)

                # Restore current selection
                if current_enum is None:
                    cb.setCurrentIndex(0)
                else:
                    # Find index whose userData matches current_enum
                    for i in range(cb.count()):
                        if cb.itemData(i) == current_enum:
                            cb.setCurrentIndex(i)
                            break
            finally:
                cb.blockSignals(False)

    def _current_sample_selections(self) -> set:
        """Return the set of enum members currently selected across all combos (excluding blanks)."""
        selected = set()
        for cb in self.sample_combos:
            data = cb.currentData()
            if data is not None:
                selected.add(data)
        return selected
    
    @QtCore.Slot()
    def _on_sample_combo_box_changed(self):
        """Rebuild each combo's items so already-chosen samples are unavailable in others."""
        selected = self._current_sample_selections()

        # Rebuild each combo while preserving its current choice and blocking signals
        for cb in self.sample_combos:
            keep = cb.currentData()  # enum member or None

            cb.blockSignals(True)
            try:
                # Capture current text to restore index after rebuild
                current_enum = keep

                # Wipe and re-add placeholder
                cb.clear()
                cb.addItem("— Select —", userData=None)

                for label, enum_member in self._sample_options:
                    if enum_member == current_enum or enum_member not in selected:
                        cb.addItem(label, userData=enum_member)

                # Restore current selection
                if current_enum is None:
                    cb.setCurrentIndex(0)
                else:
                    # Find index whose userData matches current_enum
                    for i in range(cb.count()):
                        if cb.itemData(i) == current_enum:
                            cb.setCurrentIndex(i)
                            break
            finally:
                cb.blockSignals(False)

    @QtCore.Slot()
    def _commit_pid(self):
        text = self.pid_input.text().strip()

        if not self.pid_input.hasAcceptableInput():
            self.log_bus.log("Invalid Participant ID. Enter an integer 0–1000.")
            return
        
        pid_int = int(text)
        self.log_bus.log(f"Participant ID set to {pid_int}.")
        self.participantIdCommitted.emit(str(pid_int))

        # lock after setting
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

        # Track the boolean for readiness check
        if page_name in self._cal_done:
            self._cal_done[page_name] = bool(ok)

    @QtCore.Slot(str)
    def on_calibration_done(self, page_name: str):
        self.set_cal_status(page_name, True)

class TrialCheckPage(ExperimenterPage):
    def __init__(self, name, log_bus):
        super().__init__(name, log_bus)
        self.set_status("Verify experiment/participant parameters, run some test trials")




class EndTrialsPage(ExperimenterPage):
    def __init__(self, name, log_bus):
        super().__init__(name, log_bus)
        lbl = QtWidgets.QLabel("Experiment End")
        lbl.setAlignment(QtCore.Qt.AlignCenter)
        self.add_content_widget(lbl)

        btn_inspect_all_trials = QtWidgets.QPushButton("Inspect force data of last block")
        btn_inspect_all_trials.clicked.connect(self._on_inspect_max_force_clicked)
        self.add_content_widget(btn_inspect_all_trials)

        btn_inspect_trial_by_trial = QtWidgets.QPushButton("Trial by Trial Force Data")
        btn_inspect_trial_by_trial.clicked.connect(self._on_inspect_trial_by_trial)
        self.add_content_widget(btn_inspect_trial_by_trial)

        self.__trial_analyser : TrialBlockForceAnalyser | None = None

    def check_analyser(self)-> None:
        if(self.__trial_analyser is None):
            self.__trial_analyser = TrialBlockForceAnalyser(os.getcwd()+"/Logged_Data/Trial_Force_Data/Tap_Pads_Data")
            self.__trial_analyser.run()

    def _on_inspect_max_force_clicked(self):
        self.check_analyser()
        self.__trial_analyser.plot_scatter_by_sample_id(title="Last block: min force per trial (fast vs slow)")

    def _on_inspect_trial_by_trial(self):
        self.check_analyser()
        self.__trial_analyser.plot_raw_trials_side_by_side()


