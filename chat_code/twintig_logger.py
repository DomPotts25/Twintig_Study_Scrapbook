from __future__ import annotations
from typing import Iterable

import os
import shutil
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, List, Optional, Tuple

import ximu3

# --------------------------- Low-level helpers --------------------------- #
def _connect(device_name: str) -> ximu3.Connection:
    devices = ximu3.PortScanner.scan()
    time.sleep(0.1)  # wait for ports to close
    devices = [d for d in devices if d.device_name == device_name]
    if not devices:
        raise RuntimeError(f"Unable to find {device_name}")
    connection = ximu3.Connection(devices[0].connection_info)
    result = connection.open()
    if result != ximu3.RESULT_OK:
        raise RuntimeError(
            f"Unable to open connection {connection.get_info().to_string()}. {ximu3.result_to_string(result)}"
        )
    return connection


def _send_timestamp(connection: ximu3.Connection) -> None:
    responses = connection.send_commands(
        [f'{{"timestamp":{time.time_ns() // 1000}}}'], 0, 500
    )

    if not responses:
        raise RuntimeError(f"No response to for {connection.get_info()}")
    cmd = ximu3.CommandMessage.parse(responses[0])
    if cmd.error:
        raise RuntimeError(cmd.error)




# ------------------------------ Data Types ------------------------------ #


@dataclass
class FSRPacket:
    timestamp_us: int
    values: List[float]  # len == 8 (channels)


# ------------------------------ Main Class ------------------------------ #


class TwintigLogger:
    """High-level API for Twintig devices.

    Example
    -------
    >>> logger = TwintigLogger()
    >>> logger.open()
    >>> logger.start_logging()  # begins DataLogger to disk
    >>> logger.add_fsr_callback(lambda p: print(p.timestamp_us, p.values))
    >>> # ... later
    >>> logger.stop_logging()
    >>> logger.close()
    """

    TAP_PADS_NAME = "Twintig Tap Pads"
    CARPUS_NAME = "Twintig Carpus"

    def __init__(
        self, log_destination: Optional[str] = None, log_name: str = "Logged Data"
    ) -> None:
        self.log_destination = os.path.abspath(log_destination or os.getcwd())
        self.log_name = log_name

        self._tap_conn: Optional[ximu3.Connection] = None
        self._carpus_conn: Optional[ximu3.Connection] = None
        self._imu_conns: List[ximu3.Connection] = []
        self._data_logger: Optional[ximu3.DataLogger] = None

        self._fsr_callbacks: List[Callable[[FSRPacket], None]] = []
        self._latest_fsr: Optional[FSRPacket] = None
        self._latest_lock = threading.Lock()

        self._traffic_ts: Deque[int] = deque(maxlen=4096)

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

        # Connect IMUs via mux (0x41..0x50)
        connect_infos = [
            ximu3.MuxConnectionInfo(c, self._carpus_conn) for c in range(65, 85)
        ]
        self._imu_conns = [ximu3.Connection(ci) for ci in connect_infos]
        for c in self._imu_conns:
            result = c.open()
            if result != ximu3.RESULT_OK:
                raise RuntimeError("Unable to open IMU mux connection")

            try:
                c.add_inertial_callback(self._imu_inertial_callback)
            except AttributeError:
                # Some SDK builds may use different names; ignore if missing
                pass

        # Pre-sync timestamps
        _send_timestamp(self._tap_conn)
        _send_timestamp(self._carpus_conn)

        self._open = True

    def start_logging(
        self, delete_existing: bool = False, make_unique_on_conflict: bool = True
    ) -> None:
        """Start the ximu3.DataLogger.

        If a previous session folder exists and `delete_existing` is False,
        we will automatically create a unique folder by appending a counter or
        timestamp so repeated runs don't crash with "Entity already exists".
        """
        if not self._open:
            self.open()
        # Determine target path
        base_path = os.path.join(self.log_destination, self.log_name)

        # Optionally remove old session
        if os.path.isdir(base_path) and delete_existing:
            shutil.rmtree(base_path)

        # Ensure unique folder if needed
        path = base_path
        if make_unique_on_conflict:
            if os.path.isdir(path):
                # Try a numeric suffix first, then fall back to timestamp
                idx = 1
                while os.path.isdir(f"{base_path} ({idx})") and idx < 1000:
                    idx += 1
                path = (
                    f"{base_path} ({idx})"
                    if not os.path.isdir(f"{base_path} ({idx})")
                    else base_path + time.strftime(" %Y-%m-%d_%H-%M-%S")
                )
                # Update log_name so files end up under the unique folder
                self.log_name = os.path.basename(path)

        # Start logger across tap + all IMUs
        conns = [self._tap_conn] + self._imu_conns
        assert all(conns)
        self._data_logger = ximu3.DataLogger(self.log_destination, self.log_name, conns)  # type: ignore[arg-type]
        result = self._data_logger.get_result()
        if result != ximu3.RESULT_OK:
            # Final safety: if the only issue is entity exists, retry once with timestamped name
            msg = ximu3.result_to_string(result)
            if make_unique_on_conflict and "Entity already exists" in msg:
                self.log_name = self.log_name + time.strftime(" %Y-%m-%d_%H-%M-%S")
                self._data_logger = ximu3.DataLogger(
                    self.log_destination, self.log_name, conns
                )  # retry
                result = self._data_logger.get_result()
            if result != ximu3.RESULT_OK:
                raise RuntimeError(
                    f"Data logger failed. {ximu3.result_to_string(result)}"
                )
            
    def get_msg_rate_hz(self, horizon_s: float = 2.0) -> float:
        """Return aggregate incoming message rate (Hz) over the last ~horizon_s seconds."""
        with self._latest_lock:
            if len(self._traffic_ts) < 2:
                return 0.0
            newest = self._traffic_ts[-1]
            cutoff = newest - int(horizon_s * 1_000_000)
            # Walk backwards until cutoff
            first = newest
            count = 1
            for t in reversed(self._traffic_ts):
                if t < cutoff:
                    break
                first = t
                count += 1
            dt = newest - first
            return (count - 1) / (dt / 1_000_000.0) if dt > 0 else 0.0


    def pause_logging(self) -> None:
        # Software pause; DataLogger continues, but we skip user callbacks
        self._paused = True

    def resume_logging(self) -> None:
        self._paused = False

    def stop_logging(self) -> None:
        # ximu3.DataLogger stops when the object is deleted; let GC collect it.
        self._data_logger = None

    def close(self) -> None:
        self.stop_logging()
        # Best effort; ximu3 connections are closed on GC too.
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
        for c in self._imu_conns:
            try:
                c.close()
            except Exception:
                pass
        self._imu_conns.clear()
        with self._latest_lock:
            self._latest_fsr = None
            self._traffic_ts.clear()
        self._open = False

        self._open = False

    # ----------------------------- Callbacks ----------------------------- #

    def add_fsr_callback(self, cb: Callable[[FSRPacket], None]) -> None:
        self._fsr_callbacks.append(cb)

    def clear_fsr_callbacks(self) -> None:
        self._fsr_callbacks.clear()

    def get_latest_fsr(self) -> Optional[FSRPacket]:
        with self._latest_lock:
            return self._latest_fsr

    # --------------------------- Internal wiring ------------------------- #

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
            self._traffic_ts.append(pkt.timestamp_us) 
        for cb in self._fsr_callbacks:
            try:
                cb(pkt)
            except Exception:
                # Never crash the callback chain
                pass

    def _imu_inertial_callback(self, message: "ximu3.InertialMessage"):
        if self._paused:
            return
        # We only need the timestamp to count traffic
        try:
            ts = int(message.timestamp)
        except Exception:
            return
        with self._latest_lock:
            self._traffic_ts.append(ts)

    @property
    def is_open(self) -> bool:
        return self._open

    @property
    def is_logging(self) -> bool:
        return self._data_logger is not None

    def remove_fsr_callback(self, cb: Callable[[FSRPacket], None]) -> None:
        try:
            self._fsr_callbacks.remove(cb)
        except ValueError:
            pass


# Convenience factory for quick scripts
_default_logger: Optional[TwintigLogger] = None


def get_logger() -> TwintigLogger:
    global _default_logger
    if _default_logger is None:
        _default_logger = TwintigLogger()
        _default_logger.open()
    return _default_logger
