import os
import shutil
import time

import ximu3


def connect(device_name: str) -> ximu3.Connection:
    devices = ximu3.PortScanner.scan()

    time.sleep(0.1)  # wait for ports to close

    devices = [d for d in devices if d.device_name == device_name]

    if not devices:
        raise Exception(f"Unable to find {device_name}")

    connection = ximu3.Connection(devices[0].connection_info)

    result = connection.open()

    if result != ximu3.RESULT_OK:
        raise Exception(
            f"Unable to open connection {connection.get_info().to_string()}. {ximu3.result_to_string(result)}"
        )

    return connection


def send_timestamp(connection: ximu3.Connection) -> None:
    responses = connection.send_commands(
        [f'{{"timestamp":{time.time_ns() // 1000}}}'], 0, 500
    )

    if not responses:
        raise Exception(f"No response to for {connection.get_info()}")

    command_message = ximu3.CommandMessage.parse(responses[0])

    if command_message.error:
        raise Exception(command_message.error)


# Connect to tap pads and carpus
tap_pads_connection = connect("Twintig Tap Pads")
carpus_connection = connect("Twintig Carpus")

counter = 0


def serial_accessory_callback(message: ximu3.SerialAccessoryMessage):
    global counter
    counter += 1
    # print(counter)

    if counter % 300 != 0:
        return

    # print(timestamp_format(message.timestamp) + string_format(message.string))
    # print(message.string)  # alternative to above
    values = list(map(float, message.string.split(",")))
    print(message.timestamp, " ", values)


tap_pads_connection.add_serial_accessory_callback(serial_accessory_callback)


# Connect to IMUs
connectin_infos = [
    ximu3.MuxConnectionInfo(c, carpus_connection) for c in range(65, 85)
]  # mux channels 0x41 to 0x50

imu_connections = [ximu3.Connection(c) for c in connectin_infos]

for result in [c.open() for c in imu_connections]:
    if result != ximu3.RESULT_OK:
        raise Exception("Unable to open connection")

# Check if logged data already exists
destination = os.path.dirname(os.path.abspath(__file__))
name = "Logged Data"

path = os.path.join(destination, name)

if os.path.isdir(path):
    if input(f"Delete existing {name}?  [Y/N]\n") in ["y", "Y"]:
        shutil.rmtree(path)

# Start data logger
data_logger = ximu3.DataLogger(
    destination, name, [tap_pads_connection] + imu_connections
)

result = data_logger.get_result()

if result != ximu3.RESULT_OK:
    raise Exception(f"Data logger failed. {ximu3.result_to_string(result)}")

# Set timestamps
send_timestamp(tap_pads_connection)
send_timestamp(carpus_connection)

# Wait for user to stop logging
input("Press Enter to stop logging")

