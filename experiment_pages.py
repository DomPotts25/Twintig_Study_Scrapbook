from PySide6 import QtCore, QtStateMachine, QtWidgets

QState = QtStateMachine.QState
QStateMachine = QtStateMachine.QStateMachine

from experiment_factors import Intensity



class ExperimenterPage(QtWidgets.QWidget):
    def __init__(self, title: str, color: str):
        super().__init__()
        self.next_btn = QtWidgets.QPushButton("Next ▶")
        self.prev_btn = QtWidgets.QPushButton("◀ Previous")

        title_lbl = QtWidgets.QLabel(title)
        title_lbl.setAlignment(QtCore.Qt.AlignCenter)
        title_lbl.setStyleSheet("font-size: 28px; font-weight: 600;")

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addWidget(self.prev_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(self.next_btn)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addStretch(1)
        layout.addWidget(title_lbl)
        layout.addStretch(1)
        layout.addLayout(btn_row)

        self.setStyleSheet(f"background:{color};")


class ExperimenterWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Twintig Object Characterisation Study - Experimenter Screen")

        self.stack = QtWidgets.QStackedWidget()
        self.setCentralWidget(self.stack)
