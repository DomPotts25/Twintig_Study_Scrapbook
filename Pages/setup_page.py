from Pages.experimenter_page import ExperimenterPage
from PySide6 import QtCore, QtGui, QtStateMachine, QtWidgets

from experiment_factors import Gestures, SampleGroup, StudyPhases, Velocity

QState = QtStateMachine.QState
QStateMachine = QtStateMachine.QStateMachine


# ------------------ Example page subclasses ------------------
class SetupPage(ExperimenterPage):
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
        def add_cal_row(row: int, label_text: str, page_name: str, _cal_grid):
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

            _cal_grid.addWidget(btn, row, 0, 1, 1, alignment=QtCore.Qt.AlignLeft)
            _cal_grid.addWidget(dot, row, 1, 1, 1, alignment=QtCore.Qt.AlignLeft)

        add_cal_row(0, "StillCalib", "StillCalib", cal_grid)
        add_cal_row(1, "ROM_1_Calib", "ROM_1_Calib", cal_grid)
        add_cal_row(2, "ROM_2_Calib", "ROM_2_Calib", cal_grid)
        add_cal_row(3, "Midair_Calib", "Midair_Calib", cal_grid)

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

        vel_cal_group = QtWidgets.QGroupBox("Velocity Calibration")
        vel_cal_grid = QtWidgets.QGridLayout(vel_cal_group)
        vel_cal_grid.setColumnStretch(0, 0)  # buttons column
        vel_cal_grid.setColumnStretch(1, 1)  # status column can grow a bit but stays small
        add_cal_row(0, "Velocity_Calib", "Velocity_Calib", vel_cal_grid)

        # --- Samples --- 
        sample_group = QtWidgets.QGroupBox("Sample Setup")
        sample_group_row = QtWidgets.QHBoxLayout(sample_group)
        sample_combos = [(s.name, s) for s in SampleGroup]

        sample_cb = QtWidgets.QComboBox()
        sample_cb.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        sample_cb.setMinimumWidth(140)
        sample_cb.addItem("— Select —", userData=None)

        for label, enum_member in sample_combos:
                sample_cb.addItem(label, userData=enum_member)
        
        sample_check = QtWidgets.QCheckBox(" -> Sample Inserted")
        sample_group_row.addWidget(sample_cb)
        sample_group_row.addWidget(sample_check)

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
        v.addWidget(vel_cal_group)
        v.addSpacing(10)
        v.addWidget(sample_group)
        v.addStretch()
        self.add_content_widget(wrap)

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

        # Rebuild each combo while preserving its current choice and blocking signals
        for cb in self.gesture_combos:
            keep = cb.currentData()  # enum member or None

            cb.blockSignals(True)
            try:
                # Capture current text to restore index after rebuild
                current_enum = keep

                # Wipe and re-add placeholder
                cb.clear()
                cb.addItem("— Select —", userData=None)

                # Refill with allowed options:
                # show everything that's NOT chosen by others,
                # plus the one this combo already has (so it doesn't vanish)
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


# class Setup2Page(ExperimenterPage):
#     def __init__(self, name, log_bus):
#         super().__init__(name, log_bus)
#         self.set_status("Stage 2 setup — configure environment & sensors.")

#         # Example setup controls unique to Setup2
#         grid = QtWidgets.QGridLayout()
#         grid.addWidget(QtWidgets.QLabel("Environment preset:"), 0, 0)
#         cmb = QtWidgets.QComboBox()
#         cmb.addItems(["Lab A", "Lab B", "Field"])
#         grid.addWidget(cmb, 0, 1)

#         grid.addWidget(QtWidgets.QLabel("Noise filter:"), 1, 0)
#         spin = QtWidgets.QSpinBox()
#         spin.setRange(0, 100)
#         spin.setValue(30)
#         grid.addWidget(spin, 1, 1)

#         btn_apply = QtWidgets.QPushButton("Apply Settings")
#         grid.addWidget(btn_apply, 2, 0, 1, 2)

#         # --- Page-specific button (NOT main nav) ---
#         # Velocity calibration is reachable from Setup2 but should show Back in its own page.
#         btn_vel = QtWidgets.QPushButton("Open Velocity_Calib")
#         btn_vel.clicked.connect(lambda: self.navRequested.emit("Velocity_Calib"))

#         wrap = QtWidgets.QWidget()
#         v = QtWidgets.QVBoxLayout(wrap)
#         v.addLayout(grid)
#         v.addSpacing(12)
#         v.addWidget(btn_vel)
#         v.addStretch(1)

#         self.add_content_widget(wrap)


class TrialCheckPage(ExperimenterPage):
    def __init__(self, name, log_bus):
        super().__init__(name, log_bus)
        self.set_status("Verify subject ready & parameters valid.")
        self.chk_ok = QtWidgets.QCheckBox("All checks passed")
        self.add_content_widget(self.chk_ok)
