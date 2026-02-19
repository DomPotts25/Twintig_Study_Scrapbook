from enum import Enum


class Velocity(str, Enum):
    SLOW = "slow"
    NORMAL = "normal"
    FAST = "fast"


class SampleGroup(str, Enum):
    SHAPE = "shape"
    #STIFF_SOFT = "stiff_soft"
    # STIFF_HARD = "stiff_hard"
    # TEXTURE = "texture"
    # HOLLOWNESS = "hollowness"
    # SIZE = "size"


# class ShapeSamples(str, Enum):
#     FLAT = "flat"
#     CONVEX = "convex"

# class SizeSamples(str, Enum):
#     SMALL = "small"

# class StiffnessSoftSamples(str, Enum):
#     SOFT = "soft"

# class StiffnessHardSamples(str, Enum):
#     MEDIUM = "medium"

# class TextureSamples(str, Enum):
#     COARSE = "coarse"

# class HollowSamples(str, Enum):
#     HOLLOW = "hollow"

class Gestures(str, Enum):
    TAP = "tap"
    PAT = "pat"
    # PINCH = "pinch"
    # GRASP = "grasp"
    # STROKE = "stroke"


class StudyPhases(str, Enum):
    SETUP = "setup"
    STILL_CALIBRATION = "still_calibration"
    ROM_1 = "rom_1"
    ROM_2 = "rom_2"
    MIDAIR_CALIBRATION = "midair_calibration"
    VELOCITY_CALIBRATION = "velocity_calibration"
    PRE_TRIAL = "pre_trial"
    TRIAL = "trial"
    GESTURE_SWITCH = "gesture_switch"
    SAMPLE_SWITCH = "sample_switch"
    BREAK = "break"
    END_EXPERIMENT = "end_experiment"
