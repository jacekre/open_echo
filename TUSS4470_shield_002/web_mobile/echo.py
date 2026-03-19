"""
Minimal echo reader — Serial and UDP support.
Parses the TUSS4470 binary frame format and calls a data_callback with JSON-ready dict.
"""
import asyncio
import logging
import struct

import numpy as np
import serial.tools.list_ports
import serial_asyncio_fast as aserial

log = logging.getLogger("uvicorn")

# ── Settings ──────────────────────────────────────────────────────────────────

class Settings:
    def __init__(
        self,
        connection: str = "serial",   # "serial" | "udp"
        serial_port: str = "",
        baud_rate: int = 250000,
        udp_host: str = "0.0.0.0",
        udp_port: int = 5005,
        num_samples: int = 750,
        speed_of_sound: float = 343.0,   # m/s
        sample_time: float = 13.2e-6,    # seconds per ADC sample
    ):
        self.connection = connection
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.udp_host = udp_host
        self.udp_port = udp_port
        self.num_samples = num_samples
        self.speed_of_sound = speed_of_sound
        self.sample_time = sample_time

    @property
    def resolution(self) -> float:
        """cm per sample index"""
        return (self.speed_of_sound * self.sample_time * 100) / 2

    @staticmethod
    def list_serial_ports():
        return [p.device for p in serial.tools.list_ports.comports()][::-1]


# ── Packet parser ──────────────────────────────────────────────────────────────

def parse_packet(header: bytes, payload: bytes, checksum: bytes, num_samples: int):
    """
    Returns (values: np.ndarray, depth_index: int, temperature: float, drive_voltage: float)
    or raises ValueError on bad packet.
    """
    if len(payload) != 6 + num_samples or len(checksum) != 1:
        raise ValueError("wrong length")

    calc = 0
    for b in payload:
        calc ^= b
    if calc != checksum[0]:
        raise ValueError("checksum mismatch")

    depth_idx, temp_scaled, vdrv_scaled = struct.unpack("<HhH", payload[:6])
    depth_idx = min(depth_idx, num_samples)
    values = np.frombuffer(payload[6:], dtype=np.uint8, count=num_samples)
    return values, depth_idx, temp_scaled / 100.0, vdrv_scaled / 100.0


# ── Serial reader ──────────────────────────────────────────────────────────────

class SerialReader:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._reader = None
        self._writer = None

    async def open(self):
        self._reader, self._writer = await aserial.open_serial_connection(
            url=self.settings.serial_port,
            baudrate=self.settings.baud_rate,
        )
        log.info(f"Serial opened: {self.settings.serial_port}")

    async def close(self):
        if self._writer:
            self._writer.close()
            await self._writer.wait_closed()

    async def read_one(self):
        while True:
            b = await self._reader.readexactly(1)
            if b != b"\xaa":
                continue
            payload = await self._reader.readexactly(6 + self.settings.num_samples)
            checksum = await self._reader.readexactly(1)
            try:
                return parse_packet(b, payload, checksum, self.settings.num_samples)
            except ValueError:
                continue


# ── UDP reader ─────────────────────────────────────────────────────────────────

class _UDPProtocol(asyncio.DatagramProtocol):
    def __init__(self, queue: asyncio.Queue, num_samples: int):
        self._queue = queue
        self._buf = bytearray()
        self._packet_size = 1 + 6 + num_samples + 1
        self._num_samples = num_samples

    def datagram_received(self, data: bytes, addr):
        for b in data:
            if not self._buf:
                if b == 0xAA:
                    self._buf.append(b)
            else:
                self._buf.append(b)

            if len(self._buf) == self._packet_size:
                payload = bytes(self._buf[1:1 + 6 + self._num_samples])
                checksum = bytes(self._buf[-1:])
                try:
                    result = parse_packet(b"\xaa", payload, checksum, self._num_samples)
                    self._queue.put_nowait(result)
                except ValueError:
                    pass
                finally:
                    self._buf.clear()


class UDPReader:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._transport = None
        self._queue: asyncio.Queue = asyncio.Queue()

    async def open(self):
        loop = asyncio.get_running_loop()
        self._transport, _ = await loop.create_datagram_endpoint(
            lambda: _UDPProtocol(self._queue, self.settings.num_samples),
            local_addr=(self.settings.udp_host, self.settings.udp_port),
        )
        log.info(f"UDP listening on {self.settings.udp_host}:{self.settings.udp_port}")

    async def close(self):
        if self._transport:
            self._transport.close()

    async def read_one(self):
        return await self._queue.get()


# ── EchoReader orchestrator ────────────────────────────────────────────────────

class EchoReader:
    def __init__(self, settings: Settings, data_callback):
        self.settings = settings
        self.data_callback = data_callback
        self._task: asyncio.Task | None = None

    def start(self):
        self._task = asyncio.create_task(self._run())

    def stop(self):
        if self._task:
            self._task.cancel()
            self._task = None

    async def _run(self):
        reader = SerialReader(self.settings) if self.settings.connection == "serial" else UDPReader(self.settings)
        try:
            await reader.open()
            while True:
                values, depth_idx, temperature, drive_voltage = await reader.read_one()
                resolution = self.settings.resolution
                depth_m = depth_idx * (resolution / 100)
                await self.data_callback({
                    "spectrogram": values.tolist(),
                    "measured_depth": round(depth_m, 3),
                    "temperature": temperature,
                    "drive_voltage": drive_voltage,
                    "resolution": resolution,
                })
        except asyncio.CancelledError:
            pass
        except Exception as e:
            log.error(f"EchoReader error: {e}", exc_info=True)
        finally:
            await reader.close()
