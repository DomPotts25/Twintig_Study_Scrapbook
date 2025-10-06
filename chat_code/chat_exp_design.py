# experiment_flow_pages_black.py
# Works with PySide6 (Qt6) or PyQt5 (Qt5)

try:
    from PySide6 import QtCore, QtWidgets, QtStateMachine
    QState = QtStateMachine.QState
    QStateMachine = QtStateMachine.QStateMachine
except ImportError:
    from PyQt5 import QtCore, QtWidgets, QtStateMachine
    QState = QtStateMachine.QState
    QStateMachine = QtStateMachine.QStateMachine


# ---------- Flow graph (directed) ----------
def add_edge(g, a, b):
    g.setdefault(a, []).append(b)

GRAPH = {}
for tgt in ["Setup2", "StillCalib", "ROM_1_Calib", "ROM_2_Calib", "Midair_Calib"]:
    add_edge(GRAPH, "Setup1", tgt)
    add_edge(GRAPH, tgt, "Setup1")

for tgt in ["Setup3", "Velocity_Calib"]:
    add_edge(GRAPH, "Setup2", tgt)
    add_edge(GRAPH, tgt, "Setup2")

add_edge(GRAPH, "Setup3", "TrialCheck")
add_edge(GRAPH, "TrialCheck", "Setup3")

add_edge(GRAPH, "TrialCheck", "RunTrials")

for tgt in ["GestureChange", "Sample Change"]:
    add_edge(GRAPH, "RunTrials", tgt)
    add_edge(GRAPH, tgt, "RunTrials")
add_edge(GRAPH, "RunTrials", "EndTrials")

ALL_PAGES = sorted(set(list(GRAPH.keys()) + [n for outs in GRAPH.values() for n in outs]))


# ---------- Base page ----------
class ExperimenterPage(QtWidgets.QWidget):
    """Common layout and styling for all experimenter pages."""
    def __init__(self, name: str):
        super().__init__()
        self.page_name = name
        self.nav_buttons = {}

        # Uniform dark theme
        self.setStyleSheet("""
            QWidget {
                background-color: #000000;
                color: #f0f0f0;
                font-family: 'Segoe UI', sans-serif;
            }
            QPushButton {
                background-color: #222;
                color: #f0f0f0;
                border: 1px solid #555;
                border-radius: 6px;
                padding: 6px 12px;
            }
            QPushButton:hover {
                background-color: #333;
            }
            QLabel {
                color: #f0f0f0;
            }
            QProgressBar {
                border: 1px solid #444;
                text-align: center;
                background: #111;
                color: #fff;
            }
            QProgressBar::chunk {
                background-color: #0f7;
            }
        """)

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(10)

        # Title
        title = QtWidgets.QLabel(name)
        title.setAlignment(QtCore.Qt.AlignCenter)
        title.setStyleSheet("font-size: 26px; font-weight: 600;")
        root.addWidget(title)

        # Status
        self.status_label = QtWidgets.QLabel("")
        self.status_label.setAlignment(QtCore.Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #aaa;")
        root.addWidget(self.status_label)

        # Content area (unique per subclass)
        self.content = QtWidgets.QWidget()
        self.content_layout = QtWidgets.QVBoxLayout(self.content)
        root.addWidget(self.content, 1)

        # Nav bar
        self.nav_bar = QtWidgets.QHBoxLayout()
        root.addLayout(self.nav_bar)

    def set_status(self, text: str):
        self.status_label.setText(text)

    def add_content_widget(self, w: QtWidgets.QWidget):
        self.content_layout.addWidget(w)

    def build_nav(self, targets: list[str]):
        # clear existing nav
        for i in reversed(range(self.nav_bar.count())):
            item = self.nav_bar.itemAt(i)
            w = item.widget()
            if w:
                w.setParent(None)
        self.nav_buttons.clear()

        if not targets:
            lbl = QtWidgets.QLabel("No outgoing pages.")
            lbl.setStyleSheet("color:#888;")
            self.nav_bar.addWidget(lbl)
            return

        self.nav_bar.addStretch(1)
        for t in targets:
            btn = QtWidgets.QPushButton(f"Go to {t} ➜")
            btn.setMinimumHeight(34)
            self.nav_bar.addWidget(btn)
            self.nav_buttons[t] = btn


# ---------- Example subclasses ----------
class Setup1Page(ExperimenterPage):
    def __init__(self, name):
        super().__init__(name)
        self.set_status("Connect and verify experiment hardware.")
        form = QtWidgets.QFormLayout()
        self.chk_cam = QtWidgets.QCheckBox("Camera connected")
        self.chk_imu = QtWidgets.QCheckBox("IMU connected")
        self.chk_markers = QtWidgets.QCheckBox("Markers visible")
        form.addRow(self.chk_cam)
        form.addRow(self.chk_imu)
        form.addRow(self.chk_markers)

        row = QtWidgets.QHBoxLayout()
        self.btn_probe = QtWidgets.QPushButton("Probe Devices")
        self.btn_save = QtWidgets.QPushButton("Save Setup")
        row.addWidget(self.btn_probe)
        row.addWidget(self.btn_save)

        block = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(block)
        v.addLayout(form)
        v.addSpacing(8)
        v.addLayout(row)

        self.add_content_widget(block)


class StillCalibPage(ExperimenterPage):
    def __init__(self, name):
        super().__init__(name)
        self.set_status("Hold still to collect baseline samples.")
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        self.btn_collect = QtWidgets.QPushButton("Collect 5s Sample")
        self.add_content_widget(self.progress)
        self.add_content_widget(self.btn_collect)


class RunTrialsPage(ExperimenterPage):
    def __init__(self, name):
        super().__init__(name)
        self.set_status("Configure and run trial sequences.")
        cfg_row = QtWidgets.QHBoxLayout()
        self.spin_trials = QtWidgets.QSpinBox()
        self.spin_trials.setRange(1, 999)
        self.spin_trials.setValue(20)
        self.btn_start = QtWidgets.QPushButton("Start Trials")
        self.btn_stop = QtWidgets.QPushButton("Stop")
        cfg_row.addWidget(QtWidgets.QLabel("Trials:"))
        cfg_row.addWidget(self.spin_trials)
        cfg_row.addStretch(1)
        cfg_row.addWidget(self.btn_start)
        cfg_row.addWidget(self.btn_stop)

        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("Trial log output...")

        wrap = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(wrap)
        v.addLayout(cfg_row)
        v.addWidget(self.log, 1)
        self.add_content_widget(wrap)


class GenericPage(ExperimenterPage):
    def __init__(self, name):
        super().__init__(name)
        lbl = QtWidgets.QLabel("Custom UI for this page goes here.")
        lbl.setAlignment(QtCore.Qt.AlignCenter)
        lbl.setStyleSheet("color:#ccc;")
        self.add_content_widget(lbl)


# Map page names to specific classes
PAGE_CLASS = {
    "Setup1": Setup1Page,
    "StillCalib": StillCalibPage,
    "RunTrials": RunTrialsPage,
}


# ---------- Controller ----------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, start_page="Setup1"):
        super().__init__()
        self.setWindowTitle("Experiment Flow — Black Theme")
        self.resize(960, 600)

        self.stack = QtWidgets.QStackedWidget()
        self.setCentralWidget(self.stack)

        # Create pages
        self.pages = {}
        for name in ALL_PAGES:
            cls = PAGE_CLASS.get(name, GenericPage)
            page = cls(name)
            page.build_nav(GRAPH.get(name, []))
            self.pages[name] = page
            self.stack.addWidget(page)

        # Build QStateMachine
        self.machine = QStateMachine(self)
        states = {}
        for name, page in self.pages.items():
            st = QState()
            st.assignProperty(self.stack, "currentIndex", self.stack.indexOf(page))
            states[name] = st
            self.machine.addState(st)

        # Wire transitions from each page’s nav buttons
        for src, outs in GRAPH.items():
            for tgt in outs:
                btn = self.pages[src].nav_buttons.get(tgt)
                if btn:
                    states[src].addTransition(btn.clicked, states[tgt])

        self.machine.setInitialState(states[start_page])
        self.machine.start()


def main():
    import sys
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow("Setup1")
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
