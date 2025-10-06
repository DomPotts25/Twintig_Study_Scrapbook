from __future__ import annotations

from typing import Optional

from PySide6 import QtCore, QtGui, QtWidgets


class PowerBar(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._value = 0.0
        self._target = None  # absolute target in same units as value
        self.setMinimumSize(200, 40)

    def set_value(self, value: float) -> None:
        self._value = max(0.0, value)
        self.update()

    def set_target(self, target: Optional[float]) -> None:
        self._target = target
        self.update()

    def paintEvent(self, e):
        p = QtGui.QPainter(self)
        r = self.rect()
        p.fillRect(r, self.palette().window())

        # Draw frame
        p.setPen(QtGui.QPen(QtCore.Qt.black, 2))
        p.drawRect(r.adjusted(1, 1, -1, -1))

        # Determine scale
        max_val = max(self._target or 0.0, self._value, 1.0)
        fill_w = int((self._value / max_val) * (r.width() - 4))
        # Fill current value
        p.fillRect(2, 2, fill_w, r.height() - 4, self.palette().highlight())

        # Draw target marker
        if self._target is not None and max_val > 0:
            x = 2 + int((self._target / max_val) * (r.width() - 4))
            p.setPen(QtGui.QPen(QtCore.Qt.red, 2))
            p.drawLine(x, 2, x, r.height() - 2)


class ParticipantWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Participant")
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        self.prompt = QtWidgets.QLabel("Waiting…")
        self.prompt.setAlignment(QtCore.Qt.AlignCenter)
        self.prompt.setStyleSheet("font-size: 28px; font-weight: 600;")

        self.bar = PowerBar()
        self.bar.setFixedHeight(80)

        layout.addWidget(self.prompt)
        layout.addWidget(self.bar)

    def set_prompt(self, text: str) -> None:
        self.prompt.setText(text)

    def set_bar(self, value: float, target: Optional[float]):
        self.bar.set_value(value)
        self.bar.set_target(target)
