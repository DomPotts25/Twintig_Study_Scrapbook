from Pages.experimenter_page import ExperimenterPage
from PySide6 import QtCore, QtGui, QtStateMachine, QtWidgets

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
        # Placeholder “work”: disable button, show status, then finish after 1.2s
        self.run_btn.setEnabled(False)
        self.log_bus.log("Running still calibration…")
        QtCore.QTimer.singleShot(1200, self._finish_calibration)

    def _finish_calibration(self):
        self.log_bus.log("Still calibration complete.")
        self.run_btn.setEnabled(True)
        self.calibrationDone.emit("StillCalib")  # key matches Setup’s map