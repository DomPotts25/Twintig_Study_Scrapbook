import sys

from PySide6 import QtCore, QtStateMachine, QtWidgets

import experiment_controller
import participant_page

QState = QtStateMachine.QState
QStateMachine = QtStateMachine.QStateMachine

def main():
    app = QtWidgets.QApplication(sys.argv)
    experimenter_window = experiment_controller.ExperimenterWindow()
    participant_window = participant_page.ParticipantWindow()

    # Place on two displays if present
    screens = app.screens()
    if len(screens) >= 2:
        participant_window.setGeometry(screens[0].geometry())
        experimenter_window.setGeometry(screens[1].geometry())
    else:
        # Tile side by side
        g = screens[0].availableGeometry()
        half = g.width() // 2
        participant_window.setGeometry(g.x(), g.y(), half, g.height())
        experimenter_window.setGeometry(
            g.x() + half, g.y(), g.width() - half, g.height()
        )

    experimenter_window.showMaximized()
    participant_window.showFullScreen()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()
