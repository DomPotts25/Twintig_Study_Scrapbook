from enum import Enum


class Intensity(str, Enum):
    SOFT = "soft"
    MEDIUM = "medium"
    HARD = "hard"


class SampleGroup(str, Enum):
    SHAPE = ""
    STIFF_1 = ""
    STIFF_2 = ""
