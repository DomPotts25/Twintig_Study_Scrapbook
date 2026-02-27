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
