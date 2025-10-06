# cycle_pages_statemachine_fixed.py
from PySide6 import QtCore, QtStateMachine, QtWidgets

QState = QtStateMachine.QState
QStateMachine = QtStateMachine.QStateMachine


class Page(QtWidgets.QWidget):
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


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cycling Pages with QStateMachine")

        self.stack = QtWidgets.QStackedWidget()
        self.setCentralWidget(self.stack)

        p1 = Page("Page 1", "#ffe4c4")
        p2 = Page("Page 2", "#e6e6fa")
        p3 = Page("Page 3", "#e0ffff")
        self.stack.addWidget(p1)
        self.stack.addWidget(p2)
        self.stack.addWidget(p3)

        self.machine = QStateMachine(self)

        s0 = QState()
        s0.assignProperty(self.stack, "currentIndex", 0)
        s1 = QState()
        s1.assignProperty(self.stack, "currentIndex", 1)
        s2 = QState()
        s2.assignProperty(self.stack, "currentIndex", 2)

        s0.addTransition(p1.next_btn.clicked, s1)
        s1.addTransition(p2.next_btn.clicked, s2)
        s2.addTransition(p3.next_btn.clicked, s0)

        s0.addTransition(p1.prev_btn.clicked, s2)
        s1.addTransition(p2.prev_btn.clicked, s0)
        s2.addTransition(p3.prev_btn.clicked, s1)

        self.machine.addState(s0)
        self.machine.addState(s1)
        self.machine.addState(s2)
        self.machine.setInitialState(s0)
        self.machine.start()


def main():
    import sys

    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.resize(520, 360)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
