import os
import ximu3csv
import matplotlib.pyplot as plt
import numpy as np

devices = ximu3csv.read(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Logged Data"))

devices = {d.device_name: d for d in devices}

for device in devices.values():
    plt.plot(device.inertial.timestamp, np.linalg.norm(device.inertial.accelerometer.xyz, axis=1))

plt.plot(devices["Twintig Tap Pads"].notification.timestamp, np.ones_like(devices["Twintig Tap Pads"].notification.timestamp), "^")

plt.show()
