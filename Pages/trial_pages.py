from __future__ import annotations

import random
from typing import List, Tuple

from Pages.experimenter_page import ExperimenterPage
from PySide6 import QtCore, QtWidgets
from experiment_factors import Gestures, SampleGroup, StudyPhases, Velocity

class RunTrialsPage(ExperimenterPage):

    requestTransition = QtCore.Signal(str)  # "GestureChange" / "SampleChange" / "EndTrials"

    def __init__(self, name, log_bus):
        super().__init__(name, log_bus)
        self.set_status("Run trials.")

        # ---------------- UI ----------------
        self.lbl_trial = QtWidgets.QLabel("Press Next trial to begin.")
        self.lbl_trial.setWordWrap(True)

        self.btn_next = QtWidgets.QPushButton("Next trial")
        self.btn_end = QtWidgets.QPushButton("End trials")

        row = QtWidgets.QHBoxLayout()
        
        row.addWidget(self.btn_end)
        row.addStretch(1)
        row.addWidget(self.btn_next)

        wrap = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(wrap)
        v.addWidget(self.lbl_trial)
        v.addLayout(row)
        self.add_content_widget(wrap)

        self.btn_next.clicked.connect(self._on_next_trial)
        self.btn_end.clicked.connect(lambda: self.requestTransition.emit("EndTrials"))

        self._trial_id: int = 0
        self._sample_group_idx: int = 0
        self._gesture_idx: int = 0
        self._pending_trials: List[Tuple[int, Velocity, int]] = []  # (sample_id, velocity, repetition)

        self.studyPhaseRequested.emit(StudyPhases.TRIAL)

    def _controller(self):
        # ExperimenterWindow is the QMainWindow; this page is inside a QStackedWidget
        return self.window()

    def _gesture_sequence(self) -> List[Gestures]:
        ctrl = self._controller()
        return list(getattr(ctrl, "gesture_sequence", []) or [])

    def _sample_group_sequence(self) -> List[SampleGroup]:
        ctrl = self._controller()
        return list(getattr(ctrl, "sample_group_sequence", []) or [])

    def _make_trials_for_current_gesture(self):
        """
        Build and shuffle only (sample_id, velocity, repetition).
        Gesture + sample_group order is fixed by indices.
        """
        combos: List[Tuple[int, Velocity, int]] = []
        for sample_id in (0, 1, 2, 3):
            for velocity in (Velocity.SLOW, Velocity.FAST):
                for rep in (0, 1, 2):
                    combos.append((sample_id, velocity, rep))
        random.shuffle(combos)
        self._pending_trials = combos

    def _show_sample_group_complete_popup(self):
        QtWidgets.QMessageBox.information(self, "Complete", "Sample group complete!")

    # ---------- main action ----------
    @QtCore.Slot()
    def _on_next_trial(self):
        ctrl = self._controller()
        gesture_seq = self._gesture_sequence()
        sample_group_seq = self._sample_group_sequence()

        if not gesture_seq or not sample_group_seq:
            self.lbl_trial.setText(
                "No gesture/sample-group sequence set yet. Go to Setup and commit them first."
            )
            return

        # If current block has no remaining randomized permutations, create or advance
        if not self._pending_trials:
            # first time OR we just finished a gesture block -> create list for this gesture
            self._make_trials_for_current_gesture()

        # Pop next randomized permutation
        sample_id, velocity, rep = self._pending_trials.pop()

        # Current fixed-order factors
        sample_group = sample_group_seq[self._sample_group_idx]
        gesture = gesture_seq[self._gesture_idx]

        # Update controller state (best-effort)
        if hasattr(ctrl, "set_study_phase"):
            ctrl.set_study_phase(StudyPhases.TRIAL)
        if hasattr(ctrl, "set_sample_group"):
            ctrl.set_sample_group(sample_group)
        if hasattr(ctrl, "set_gesture"):
            ctrl.set_gesture(gesture)
        if hasattr(ctrl, "set_trial_id"):
            ctrl.set_trial_id(self._trial_id)
        if hasattr(ctrl, "set_sample_id"):
            ctrl.set_sample_id(sample_id)
        if hasattr(ctrl, "set_sample_name"):
            # You currently don't have a mapping from (sample_group, sample_id) -> name,
            # so we keep this simple and just store the id string.
            ctrl.set_sample_name(f"{sample_group}:{sample_id}")
        if hasattr(ctrl, "set_velocity"):
            ctrl.set_velocity(velocity)

        # Display
        self.lbl_trial.setText(
            f"Trial {self._trial_id} | sample_group={sample_group} | gesture={gesture} "
            f"| sample_id={sample_id} | velocity={velocity} | repetition={rep}"
        )
        self.log_bus.log(
            f"[trial] id={self._trial_id} group={sample_group} gesture={gesture} "
            f"sample_id={sample_id} velocity={velocity} rep={rep}"
        )

        self._trial_id += 1

        # If we just consumed the last permutation for this (sample_group, gesture) block:
        if not self._pending_trials:
            # advance gesture (fixed order)
            self._gesture_idx += 1

            if self._gesture_idx < len(gesture_seq):
                # next gesture -> controller should show GestureChange page
                self.requestTransition.emit("GestureChange")
                return

            # gesture sequence finished for this sample group
            self._gesture_idx = 0
            self._sample_group_idx += 1

            # popup before progressing to next sample group
            self._show_sample_group_complete_popup()

            if self._sample_group_idx < len(sample_group_seq):
                # controller should show SampleChange page
                self.requestTransition.emit("SampleChange")
            else:
                # experiment finished
                self.requestTransition.emit("EndTrials")
