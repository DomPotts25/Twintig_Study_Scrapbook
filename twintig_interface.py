from __future__ import annotations
from typing import Iterable

import os
import shutil
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, List, Optional, Tuple
import json
from fnmatch import fnmatch

import ximu3

def _connect(device_name: str) -> ximu3.Connection:
    devices = ximu3.PortScanner.scan()
    time.sleep(0.1)  # wait for ports to close
    devices = [d for d in devices if d.device_name == device_name]
    
    if not devices:
        raise RuntimeError(f"Unable to find {device_name}")
    
    connection = ximu3.Connection(devices[0].connection_config)
    connection.open()

    return connection

def _send_timestamp(connection: ximu3.Connection) -> None:
    response = connection.send_command(f'{{"timestamp":{time.time_ns() // 1000}}}')

    if not response:
        raise RuntimeError(f"No response to for {connection.get_config()}")

    if response.error:
        raise RuntimeError(response.error)

# ------------------------------ Data Types ------------------------------ #
@dataclass
class FSRPacket:
    timestamp_us: int
    values: List[float]  # len == 8 (channels)


class ImuConnection:
    def __init__(self, config: ximu3.MuxConnectionConfig) -> None:
        self.__connection = ximu3.Connection(config)

        self.__connection.open()

        response = self.__connection.ping()

        if not response:
            raise Exception(f"Ping failed for {self.__connection.get_config()}")

        self.__name = response.device_name

    def close(self) -> None:
        self.__connection.close()

    def send_command(self, command: str) -> None:
        response = self.__connection.send_command(command)

        if not response:
            raise Exception(f"No response. {command} for {self.__connection.get_config()}")

        if response.error:
            raise Exception(f"{response.error}. {command} for {self.__connection.get_config()}")

    @property
    def name(self) -> str:
        return self.__name


# ------------------------------ Main Class ------------------------------ #
class TwintigInterface:
    TAP_PADS_NAME = "Twintig Tap Pads"
    CARPUS_NAME = "Twintig Carpus"

    def __init__(
        self, log_destination: Optional[str] = None, log_name: str = "Logged Data"
    ) -> None:
        self.log_destination = os.path.abspath(log_destination or os.getcwd())
        self.log_name = log_name

        self._tap_conn: Optional[ximu3.Connection] = None
        self._carpus_conn: Optional[ximu3.Connection] = None
        self._data_logger: Optional[ximu3.DataLogger] = None
        self._imu_connections: List[ImuConnection] = []

        self._latest_fsr: Optional[FSRPacket] = None
        self._latest_lock = threading.Lock()

        self._carpus_msg_rate: float = 0.0
        self._tap_pads_msg_rate: float = 0.0

        self._paused = False
        self._open = False

    # ------------------------ Lifecycle & Logging ------------------------ #
    def open(self) -> None:
        if self._open:
            return
        
        # Connect to tap pads + carpus (mux controller)
        self._tap_conn = _connect(self.TAP_PADS_NAME)
        self._carpus_conn = _connect(self.CARPUS_NAME)

        # Route serial accessory callback for FSRs
        self._tap_conn.add_serial_accessory_callback(self._serial_accessory_callback)
        
        self._carpus_conn.add_statistics_callback(self._carpus_stats_callback)
        self._tap_conn.add_statistics_callback(self._tap_pads_stats_callback)

        self._imu_connections = [ImuConnection(c) for c in [ximu3.MuxConnectionConfig(c, self._carpus_conn) for c in range(0x41, 0x55)]]

        # Pre-sync timestamps
        _send_timestamp(self._tap_conn)
        _send_timestamp(self._carpus_conn)

        self._open = True

    def send_commands_to_all(self, command_file_path) -> None:
        if(not self._open):
            return

        with open(command_file_path) as file:
            scripts = json.load(file)

        for script in scripts:
            for imu_connection in [i for i in self._imu_connections if fnmatch(i.name, script["name"])]:
                for command in script["commands"]:
                    imu_connection.send_command(command)


    def send_command_to_all(self, command:str) -> None:
        if(not self._open):
            return
        
        for imu_connection in self._imu_connections:
            imu_connection.send_command(command)


    def close(self) -> None:
        
        try:
            if self._tap_conn:
                self._tap_conn.close()
        finally:
            self._tap_conn = None
        
        try:
            if self._carpus_conn:
                self._carpus_conn.close()
        finally:
            self._carpus_conn = None

        for c in self._imu_connections:
            try:
                c.close()
            except Exception:
                pass
        self._imu_connections.clear()

        with self._latest_lock:
            self._latest_fsr = None
        self._open = False
        self._carpus_msg_rate = 0.0
        self._tap_pads_msg_rate = 0.0

    # ----------------------------- Callbacks ----------------------------- #
    def get_latest_fsr(self) -> Optional[FSRPacket]:
        with self._latest_lock:
            return self._latest_fsr

    def _serial_accessory_callback(self, message: ximu3.SerialAccessoryMessage):
        if self._paused:
            return
        try:
            values = [float(v) for v in message.string.split(",")]
        except Exception:
            return
        
        pkt = FSRPacket(timestamp_us=message.timestamp, values=values)

        with self._latest_lock:
            self._latest_fsr = pkt

    def _carpus_stats_callback(self, message: ximu3.Statistics):
        if self._paused:
            return
        self._carpus_msg_rate = message.message_rate

    def _tap_pads_stats_callback(self, message: ximu3.Statistics):
        if self._paused:
            return
        self._tap_pads_msg_rate = message.message_rate

    @property
    def is_open(self) -> bool:
        return self._open

    @property
    def is_logging(self) -> bool:
        return self._data_logger is not None
    