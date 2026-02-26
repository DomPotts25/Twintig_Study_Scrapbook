from __future__ import annotations

import os
import random
import shutil
from typing import List, Tuple

import ximu3
from PySide6 import QtCore, QtWidgets
from ximu3 import DataLogger

from experiment_factors import Gestures, SampleGroup, StudyPhases, Velocity
from Pages.experimenter_page import ExperimenterPage
from Tools.trial_force_analyser import TrialBlockForceAnalyser


class RunTrialsPage(ExperimenterPage):
    requestTransition = QtCore.Signal(
        str
    )  # "GestureChange" / "SampleChange" / "EndTrials"

    def __init__(self, name, log_bus):
        super().__init__(name, log_bus)
        self.set_status("Run trials.")

        self.lbl_trial = QtWidgets.QLabel("Press Next trial to begin.")
        self.lbl_trial.setWordWrap(True)
        self.lbl_next_gesture = QtWidgets.QLabel("")
        self.lbl_next_gesture.setWordWrap(True)

        self.btn_begin = QtWidgets.QPushButton("Begin trials")
        self.btn_next = QtWidgets.QPushButton("Next trial")
        self.btn_end = QtWidgets.QPushButton("End trials")

        self.btn_next.setEnabled(False)
        self.btn_begin.setEnabled(True)

        row = QtWidgets.QHBoxLayout()

        row.addWidget(self.btn_end)
        row.addStretch(1)
        row.addWidget(self.btn_begin)
        row.addWidget(self.btn_next)

        wrap = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(wrap)
        v.addWidget(self.lbl_next_gesture)
        v.addWidget(self.lbl_trial)
        v.addLayout(row)
        self.add_content_widget(wrap)

        self.btn_begin.clicked.connect(self._on_begin_trials)
        self.btn_next.clicked.connect(self._on_next_trial)
        self.btn_end.clicked.connect(self._on_end_trials_clicked)
        self.__sample_channels = [0, 1, 2, 3]
        self.__reps = [0, 1, 2]
        self.__trial_id: int = 0
        self._sample_group_idx: int = 0
        self.__gesture_idx: int = 0
        self.__pending_trials: List[
            Tuple[int, Velocity, int]
        ] = []  # (sample_id, velocity, repetition)

        self.__forceDataLogger = None
        self.__current_trial_ctx = None  # dict or None

    def _controller(self):
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
        for sample_id in self.__sample_channels:
            for velocity in (Velocity.SLOW, Velocity.FAST):
                for rep in self.__reps:
                    combos.append((sample_id, velocity, rep))
        random.shuffle(combos)
        self.__pending_trials = combos

    def _show_sample_group_complete_popup(self):
        QtWidgets.QMessageBox.information(self, "Complete", "Sample group complete!")

    def _update_next_gesture_label(self):
        gesture_seq = self._gesture_sequence()
        if not gesture_seq:
            self.lbl_next_gesture.setText("Next gesture: (not set)")
            return

        # If you’re between blocks and about to start the next gesture,
        # this index already points at the upcoming gesture.
        if 0 <= self.__gesture_idx < len(gesture_seq):
            g = gesture_seq[self.__gesture_idx]
            self.lbl_next_gesture.setText(f"Next gesture: {g}")
        else:
            self.lbl_next_gesture.setText("Next gesture: (complete)")

    def _load_one_trial(self) -> bool:
        ctrl = self._controller()
        gesture_seq = self._gesture_sequence()
        sample_group_seq = self._sample_group_sequence()

        if not gesture_seq or not sample_group_seq:
            self.lbl_trial.setText(
                "No gesture/sample-group sequence set yet. Go to Setup and commit them first."
            )
            return False

        # If this block has no remaining randomized permutations, create them now
        if not self.__pending_trials:
            self._make_trials_for_current_gesture()

        # Pull next randomized permutation
        sample_id, velocity, rep = self.__pending_trials.pop()

        # Fixed-order factors
        sample_group = sample_group_seq[self._sample_group_idx]
        gesture = gesture_seq[self.__gesture_idx]

        # Update controller state (best-effort)
        if hasattr(ctrl, "set_study_phase"):
            ctrl.set_study_phase(StudyPhases.TRIAL)
        if hasattr(ctrl, "set_sample_group"):
            ctrl.set_sample_group(sample_group)
        if hasattr(ctrl, "set_gesture"):
            ctrl.set_gesture(gesture)
        if hasattr(ctrl, "set_trial_id"):
            ctrl.set_trial_id(self.__trial_id)
        if hasattr(ctrl, "set_sample_id"):
            ctrl.set_sample_id(sample_id)
        if hasattr(ctrl, "set_sample_name"):
            ctrl.set_sample_name(f"{sample_group}:{sample_id}")
        if hasattr(ctrl, "set_velocity"):
            ctrl.set_velocity(velocity)

        # Display
        self.lbl_trial.setText(
            f"Trial {self.__trial_id} | sample_group={sample_group} | gesture={gesture} | sample_id={sample_id} | velocity={velocity} | repetition={rep}"
        )
        self.log_bus.log(
            f"[trial] id={self.__trial_id} group={sample_group} gesture={gesture} sample_id={sample_id} velocity={velocity} rep={rep}"
        )

        self._current_trial_ctx = {
            "trial_id": self.__trial_id,
            "sample_group": sample_group,
            "gesture": gesture,
            "sample_id": sample_id,
            "velocity": velocity,
            "repetition": rep,
        }

        self.get_participant_page().set_prompt(
            f"Please {str(self._current_trial_ctx['gesture']).upper()} Sample {str(self._current_trial_ctx['sample_id'] + 1).upper()} \n {str(self._current_trial_ctx['velocity']).split('.')[1]} \n \n {len(self.__pending_trials)} Trials remaining\n"
        )

        # Send LED command!

        self.__trial_id += 1
        return True

    def _send_trial_end_note(self):
        context = self._current_trial_ctx
        if not context:
            return
        self._twintig_interface.send_command_to_tap_pads(
            f'{{"note":"TRIAL {context["trial_id"]} END; gesture: {context["gesture"]}; sample_id: {context["sample_id"]}; velocity: {context["velocity"]}; repetition: {context["repetition"]};"}}'
        )

    @QtCore.Slot()
    def _on_begin_trials(self):
        self.studyPhaseRequested.emit(StudyPhases.TRIAL)
        # Load the first trial of the current block, then enable Next.
        ok = self._load_one_trial()
        if not ok:
            return

        # create data logger for force data
        out_dir = os.path.join(
            os.getcwd(), "Logged_Data", "Trial_Force_Data", "Tap_Pads_Data"
        )
        try:
            shutil.rmtree(out_dir)
        except Exception as e:
            print(e)

        self.__forceDataLogger = DataLogger(
            os.path.join(os.getcwd(), "Logged_Data", "Trial_Force_Data"),
            "Tap_Pads_Data",
            self._twintig_interface.get_tap_pads_connection_as_list(),
        )

        context = self._current_trial_ctx
        self._twintig_interface.send_command_to_tap_pads(
            f'{{"note":"TRIAL {context["trial_id"]} BEGIN; gesture: {context["gesture"]}; sample_id: {context["sample_id"]}; velocity: {context["velocity"]}; repetition: {context["repetition"]};"}}'
        )

        self.btn_begin.setEnabled(False)
        self.btn_next.setEnabled(True)

    def __evaluate_next_trial(self):
        # print("hello!")
        # If there is no next trial to load, we just ended the final trial of the block.
        if not self.__pending_trials:
            gesture_seq = self._gesture_sequence()
            sample_group_seq = self._sample_group_sequence()

            # tear down logger AFTER ending the final trial
            del self.__forceDataLogger

            # advance gesture/sample_group and transition
            self.__gesture_idx += 1

            if self.__gesture_idx < len(gesture_seq):
                self.requestTransition.emit("GestureChange")
                self.get_participant_page().set_prompt(
                    "Trial Block Completed! \n \n \n Get ready for the next Gesture..."
                )
                self.studyPhaseRequested.emit(StudyPhases.GESTURE_SWITCH)
                return

            self.__gesture_idx = 0
            self._sample_group_idx += 1
            self._show_sample_group_complete_popup()

            if self._sample_group_idx < len(sample_group_seq):
                self.requestTransition.emit("SampleChange")
                self.get_participant_page().set_prompt(
                    "Trial Block Completed! \n \n \n The experimenter will switch the samples..."
                )
                self.studyPhaseRequested.emit(StudyPhases.SAMPLE_SWITCH)
            else:
                self.requestTransition.emit("EndTrials")
                self.get_participant_page().set_prompt(
                    "Experiment Complete! \n \n \n  you for participating :^) "
                )
                self.studyPhaseRequested.emit(StudyPhases.END_EXPERIMENT)
            return

        # Otherwise, load the next trial and continue within the block
        ok = self._load_one_trial()
        if not ok:
            return

    # ---------- main action ----------
    @QtCore.Slot()
    def _on_next_trial(self):
        # End the currently active trial
        self._send_trial_end_note()
        self.get_participant_page().set_prompt(
            f"Trial Complete! \n \n \n {len(self.__pending_trials)} Trials remaining\n"
        )

        QtCore.QTimer.singleShot(1000, self.__evaluate_next_trial)

    def _on_end_trials_clicked(self):
        del self.__forceDataLogger
        self.requestTransition.emit("EndTrials")
        self.studyPhaseRequested.emit(StudyPhases.END_EXPERIMENT)
        self.get_participant_page().set_prompt(
            "Experiment Complete! \n \n \n  you for participating :^)"
        )

    def showEvent(self, event):
        super().showEvent(event)
        self._update_next_gesture_label()
        self._reset_for_entry()

    def _reset_for_entry(self):
        # wipe old text + require begin again
        self.lbl_trial.setText("Ready. Press Begin trials to load the first trial.")
        self.studyPhaseRequested.emit(StudyPhases.PRE_TRIAL)
        self.get_participant_page().set_prompt("Ready to begin next trials? \n")
        self.btn_next.setEnabled(False)
        self.btn_begin.setEnabled(True)


class GestureChangeReviewPage(ExperimenterPage):
    def __init__(self, name, log_bus):
        super().__init__(name, log_bus)
        lbl = QtWidgets.QLabel("Gesture Block Complete")
        lbl.setAlignment(QtCore.Qt.AlignCenter)
        self.add_content_widget(lbl)

        btn_inspect_all_trials = QtWidgets.QPushButton(
            "Inspect force data of last block"
        )
        btn_inspect_all_trials.clicked.connect(self._on_inspect_max_force_clicked)
        self.add_content_widget(btn_inspect_all_trials)

        btn_inspect_trial_by_trial = QtWidgets.QPushButton("Trial by Trial Force Data")
        btn_inspect_trial_by_trial.clicked.connect(self._on_inspect_trial_by_trial)
        self.add_content_widget(btn_inspect_trial_by_trial)

        self.__trial_analyser: TrialBlockForceAnalyser | None = None

    def check_analyser(self) -> None:
        if self.__trial_analyser is None:
            self.__trial_analyser = TrialBlockForceAnalyser(
                os.getcwd() + "/Logged_Data/Trial_Force_Data/Tap_Pads_Data"
            )
            self.__trial_analyser.run()

    def _on_inspect_max_force_clicked(self):
        self.check_analyser()
        self.__trial_analyser.plot_scatter_by_sample_id(
            title="Last block: min force per trial (fast vs slow)"
        )

    def _on_inspect_trial_by_trial(self):
        self.check_analyser()
        self.__trial_analyser.plot_raw_trials_side_by_side()


class SampleChangeReviewPage(ExperimenterPage):
    def __init__(self, name, log_bus):
        super().__init__(name, log_bus)
        lbl = QtWidgets.QLabel("Sample Block Complete")
        lbl.setAlignment(QtCore.Qt.AlignCenter)
        self.add_content_widget(lbl)

        btn_inspect_all_trials = QtWidgets.QPushButton(
            "Inspect force data of last block"
        )
        btn_inspect_all_trials.clicked.connect(self._on_inspect_max_force_clicked)
        self.add_content_widget(btn_inspect_all_trials)

        btn_inspect_trial_by_trial = QtWidgets.QPushButton("Trial by Trial Force Data")
        btn_inspect_trial_by_trial.clicked.connect(self._on_inspect_trial_by_trial)
        self.add_content_widget(btn_inspect_trial_by_trial)

        self.__trial_analyser: TrialBlockForceAnalyser | None = None

    def check_analyser(self) -> None:
        if self.__trial_analyser is None:
            self.__trial_analyser = TrialBlockForceAnalyser(
                os.getcwd() + "/Logged_Data/Trial_Force_Data/Tap_Pads_Data"
            )
            self.__trial_analyser.run()

    def _on_inspect_max_force_clicked(self):
        self.check_analyser()
        self.__trial_analyser.plot_scatter_by_sample_id(
            title="Last block: min force per trial (fast vs slow)"
        )

    def _on_inspect_trial_by_trial(self):
        self.check_analyser()
        self.__trial_analyser.plot_raw_trials_side_by_side()


class TestTrialsPage(ExperimenterPage):
    def __init__(self, name, log_bus):
        super().__init__(name, log_bus)
        self.set_status("Run Test Trials.")

        self.lbl_trial = QtWidgets.QLabel("Press Next trial to begin test trials.")
        self.lbl_trial.setWordWrap(True)
        self.lbl_next_gesture = QtWidgets.QLabel("")
        self.lbl_next_gesture.setWordWrap(True)

        self.btn_begin = QtWidgets.QPushButton("Begin test trials")
        self.btn_next = QtWidgets.QPushButton("Next test trial")
        self.btn_end = QtWidgets.QPushButton("End test trials")

        self.btn_next.setEnabled(False)
        self.btn_begin.setEnabled(True)

        row = QtWidgets.QHBoxLayout()

        row.addWidget(self.btn_end)
        row.addStretch(1)
        row.addWidget(self.btn_begin)
        row.addWidget(self.btn_next)

        wrap = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(wrap)
        v.addWidget(self.lbl_next_gesture)
        v.addWidget(self.lbl_trial)
        v.addLayout(row)
        self.add_content_widget(wrap)

        # self.btn_begin.clicked.connect(self._on_begin_trials)
        # self.btn_next.clicked.connect(self._on_next_trial)
        # self.btn_end.clicked.connect(self._on_end_trials_clicked)
        # self.__sample_channels = [0,1,2,3]
        # self.__reps = [0,1,2]
        # self.__trial_id: int = 0
        # self._sample_group_idx: int = 0
        # self.__gesture_idx: int = 0
        # self.__pending_trials: List[Tuple[int, Velocity, int]] = []  # (sample_id, velocity, repetition)

        # self.__forceDataLogger = None
        # self.__current_trial_ctx = None  # dict or None
