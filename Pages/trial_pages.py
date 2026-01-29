from Pages.experimenter_page import ExperimenterPage
from PySide6 import QtCore, QtGui, QtStateMachine, QtWidgets
from experiment_factors import Gestures, SampleGroup, StudyPhases, Velocity

QState = QtStateMachine.QState
QStateMachine = QtStateMachine.QStateMachine


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
        # self.log = QtWidgets.QPlainTextEdit()
        # self.log.setReadOnly(True)
        # self.log.setPlaceholderText("Trial log...")
        wrap = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(wrap)
        v.addLayout(row)
        #v.addWidget(self.log, 1)
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

        self.studyPhaseRequested.emit(
            StudyPhases.TRIAL
        ) 
