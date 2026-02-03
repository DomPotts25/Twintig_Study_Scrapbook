from Pages.experimenter_page import ExperimenterPage
from PySide6 import QtCore, QtGui, QtStateMachine, QtWidgets
from experiment_factors import Gestures, SampleGroup, StudyPhases, Velocity

import ximu3
import shutil
import os

QState = QtStateMachine.QState
QStateMachine = QtStateMachine.QStateMachine


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
        
        ### Placeholder "work": disable button, show status, then finish after 1.2s

        self.run_btn.setEnabled(False)
        self.log_bus.log("Running still calibration…")
        self.studyPhaseRequested.emit(
            StudyPhases.STILL_CALIBRATION
        ) 
        
        QtCore.QTimer.singleShot(1200, self._finish_calibration)

    def _finish_calibration(self):
        self.log_bus.log("Still calibration complete.")
        self.run_btn.setEnabled(True)
        self.studyPhaseRequested.emit(
            StudyPhases.SETUP
        ) 
        self.calibrationDone.emit("StillCalib")  # key matches Setup’s map


class VelocityCalibPage(ExperimenterPage):
    TRIALS_PER_CONDITION = 2

    def __init__(self, name, log_bus):
        super().__init__(name, log_bus)
        self._is_calibrating = False
        self.__velocityCalDataLogger = None
        self.__NAME = "velocityCal"
        
        self._conditions = [(g, v) for g in Gestures for v in Velocity]
        self._condition_index = 0
        self._trial_in_condition = 0

        wrap = QtWidgets.QWidget()
        ctrl_row = QtWidgets.QHBoxLayout(wrap)

        # push content to participant page
        self.btn_start_calib = QtWidgets.QPushButton("Start Velocity Calibration")
        self.btn_next_trial = QtWidgets.QPushButton("Next Velocity Trial")
        self.btn_stop_calib = QtWidgets.QPushButton("STOP Velocity Calibration")

        self.btn_start_calib.clicked.connect(self.__on_start_calib_clicked)
        self.btn_next_trial.clicked.connect(self.__on_next_trial_clicked)
        self.btn_stop_calib.clicked.connect(self._on_stop_velocity_cal)

        ctrl_row.addWidget(self.btn_start_calib) 
        ctrl_row.addWidget(self.btn_next_trial)
        ctrl_row.addWidget(self.btn_stop_calib)
        ctrl_row.setAlignment(QtCore.Qt.AlignCenter)

        self.add_content_widget(wrap)

        self._update_ui_state()
    
    def _current_condition(self):
        return self._conditions[self._condition_index]

    def _update_ui_state(self):
        self.btn_start_calib.setEnabled(not self._is_calibrating)
        self.btn_next_trial.setEnabled(self._is_calibrating)
        self.btn_stop_calib.setEnabled(self._is_calibrating)

        # Optional: change button text to show progress
        if self._is_calibrating:
            gesture, velocity = self._current_condition()
            cond_num = self._condition_index + 1
            cond_total = len(self._conditions)
            trial_num = self._trial_in_condition + 1
            self.btn_next_trial.setText(
                f"Next Trial (Cond {cond_num}/{cond_total}: {gesture.value}-{velocity.value}, "
                f"Trial {trial_num}/{self.TRIALS_PER_CONDITION})"
            )
        else:
            self.btn_next_trial.setText("Next Velocity Trial")


    def __on_start_calib_clicked(self) -> None:
        if not self._twintig_interface.is_open:            
            self.log_bus.log("[vel_Cal]: Please connect devices and insert samples")
            return
        
        self._is_calibrating = True
        self._condition_index = 0
        self._trial_in_condition = 0

        out_dir = os.path.join(os.getcwd(), "Logged_Data", "Velocity_Calibration_Force_Data", self.__NAME)
        try:
            shutil.rmtree(out_dir)
        except Exception as e:
            print(e)

        self.__velocityCalDataLogger = ximu3.DataLogger(
            os.path.join(os.getcwd(), "Logged_Data", "Velocity_Calibration_Force_Data"),
            self.__NAME,
            self._twintig_interface.get_tap_pads_connection_as_list(),
        )

        g, v = self._current_condition()
        self.log_bus.log(f"[vel_Cal]: Calibration begun. Starting {g.value}-{v.value}, trial 1")

        #participant window update

        self._update_ui_state()

    def __on_next_trial_clicked(self) -> None:
        if not self._is_calibrating:
            return

        gesture, velocity = self._current_condition()

        self._twintig_interface.send_command_to_tap_pads(
            f'{{"note":"TRIAL {self._trial_in_condition} END; gesture: {gesture.value}; velocity: {velocity.value}"}}'
            )
        self.log_bus.log(
            f"[vel_Cal]: Completed {gesture.value}-{velocity.value} "
            f"trial {self._trial_in_condition + 1}/{self.TRIALS_PER_CONDITION}"
        )

        self._trial_in_condition += 1
        # Update participant window

        if self._trial_in_condition >= self.TRIALS_PER_CONDITION:
            self._trial_in_condition = 0
            self._condition_index += 1

            # All conditions done -> stop
            if self._condition_index >= len(self._conditions):
                self._on_stop_velocity_cal()
                return

            next_g, next_v = self._current_condition()
            self.log_bus.log(f"[vel_Cal]: Next condition: {next_g.value}-{next_v.value}, trial 1")
        else:
            self.log_bus.log("[vel_Cal]: Next trial...")

        self._update_ui_state()


    def _on_stop_velocity_cal(self) -> None:
        if not self._is_calibrating:
            return

        self._is_calibrating = False

        del self.__velocityCalDataLogger

        self.log_bus.log("[vel_Cal]: Calibration complete")

        # participant window update
        # compute thresholds, etc.

        self._update_ui_state()
