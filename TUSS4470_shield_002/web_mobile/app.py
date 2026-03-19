"""
Open Echo — minimal mobile-friendly web server.

Usage:
    uvicorn app:app --host 0.0.0.0 --port 8000

Configuration via environment variables or editing Settings() below:
    ECHO_CONNECTION   serial | udp          (default: serial)
    ECHO_PORT         e.g. COM3 or /dev/ttyUSB0
    ECHO_BAUD         baud rate             (default: 250000)
    ECHO_UDP_HOST     UDP bind address      (default: 0.0.0.0)
    ECHO_UDP_PORT     UDP port              (default: 5005)
    ECHO_SAMPLES      samples per frame     (default: 750)
    ECHO_SOS          speed of sound m/s   (default: 343)
    ECHO_SAMPLE_TIME  ADC sample time s     (default: 13.2e-6)
"""
import asyncio
import json
import logging
import os
import struct
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from echo import EchoReader, Settings

log = logging.getLogger("uvicorn")

# ── Settings from env ──────────────────────────────────────────────────────────

settings = Settings(
    connection=os.getenv("ECHO_CONNECTION", "serial"),
    serial_port=os.getenv("ECHO_PORT", ""),
    baud_rate=int(os.getenv("ECHO_BAUD", "250000")),
    udp_host=os.getenv("ECHO_UDP_HOST", "0.0.0.0"),
    udp_port=int(os.getenv("ECHO_UDP_PORT", "5005")),
    num_samples=int(os.getenv("ECHO_SAMPLES", "750")),
    speed_of_sound=float(os.getenv("ECHO_SOS", "343")),
    sample_time=float(os.getenv("ECHO_SAMPLE_TIME", "13.2e-6")),
)

RECORDINGS_DIR = Path(os.getenv("ECHO_RECORDINGS_DIR", "recordings"))
RECORDINGS_DIR.mkdir(exist_ok=True)

# ── Binary recording format ────────────────────────────────────────────────────
# Each frame: [timestamp:f64 8B][depth_index:u16 2B][temp_scaled:i16 2B]
#             [vdrv_scaled:u16 2B][samples:u8×N][checksum:u8 1B]
# Header:     [magic:4B "ECHO"][version:u8 1B][num_samples:u16 2B]
#             [speed_of_sound:f32 4B][sample_time:f64 8B]  = 19 bytes
MAGIC       = b"ECHO"
FILE_VERSION = 1
HEADER_FMT  = "<4sBHfd"   # magic, version, num_samples, sos, sample_time
HEADER_SIZE = struct.calcsize(HEADER_FMT)
FRAME_HDR   = "<dHhH"     # timestamp, depth_index, temp_scaled, vdrv_scaled
FRAME_HDR_S = struct.calcsize(FRAME_HDR)  # 14 bytes


def frame_size(num_samples: int) -> int:
    return FRAME_HDR_S + num_samples + 1  # +1 checksum


# ── Recorder ──────────────────────────────────────────────────────────────────

class Recorder:
    def __init__(self):
        self._file = None
        self.filename: str = ""
        self.frame_count: int = 0
        self._t0: float = 0.0

    @property
    def active(self) -> bool:
        return self._file is not None

    def start(self, s: Settings) -> str:
        if self._file:
            self.stop()
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.filename = f"echo_{ts}.echorec"
        path = RECORDINGS_DIR / self.filename
        self._file = open(path, "wb")
        self.frame_count = 0
        import time; self._t0 = time.time()
        # Write file header
        hdr = struct.pack(HEADER_FMT, MAGIC, FILE_VERSION,
                          s.num_samples, s.speed_of_sound, s.sample_time)
        self._file.write(hdr)
        self._file.flush()
        log.info(f"Recording started: {path}")
        return self.filename

    def write(self, data: dict, s: Settings):
        if not self._file:
            return
        import time
        ts     = time.time() - self._t0
        depth  = data.get("depth_index", 0)
        temp   = round(data.get("temperature", 0) * 100)
        vdrv   = round(data.get("drive_voltage", 0) * 100)
        samples = bytes(data.get("spectrogram", [0] * s.num_samples))

        # checksum = XOR of payload (depth+temp+vdrv bytes + samples)
        payload = struct.pack("<HhH", depth, temp, vdrv) + samples
        ck = 0
        for b in payload:
            ck ^= b

        frame = struct.pack(FRAME_HDR, ts, depth, temp, vdrv) + samples + bytes([ck])
        self._file.write(frame)
        self._file.flush()
        self.frame_count += 1

    def stop(self) -> str:
        name = self.filename
        if self._file:
            self._file.close()
            self._file = None
            log.info(f"Recording stopped: {name} ({self.frame_count} frames)")
        return name


recorder = Recorder()

# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

_clients: list[WebSocket] = []


async def broadcast(data: dict):
    recorder.write(data, settings)

    msg = json.dumps({
        **data,
        "recording": recorder.active,
        "rec_filename": recorder.filename,
        "rec_frames": recorder.frame_count,
    })
    dead = []
    for ws in _clients:
        try:
            await ws.send_text(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _clients.remove(ws)


echo_reader = EchoReader(settings, broadcast)


@app.on_event("startup")
async def startup():
    if settings.connection == "serial" and not settings.serial_port:
        ports = Settings.list_serial_ports()
        if ports:
            settings.serial_port = ports[0]
            log.info(f"Auto-selected serial port: {settings.serial_port}")
        else:
            log.warning("No serial port found. Set ECHO_PORT env variable.")
    echo_reader.start()


@app.on_event("shutdown")
async def shutdown():
    recorder.stop()
    echo_reader.stop()


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    with open("templates/index.html", encoding="utf-8") as f:
        return f.read()


@app.post("/record/start")
async def record_start():
    filename = recorder.start(settings)
    return JSONResponse({"status": "started", "filename": filename})


@app.post("/record/stop")
async def record_stop():
    filename = recorder.stop()
    return JSONResponse({"status": "stopped", "filename": filename})


@app.get("/record/status")
async def record_status():
    return JSONResponse({
        "recording": recorder.active,
        "filename": recorder.filename,
        "frames": recorder.frame_count,
    })


@app.get("/player", response_class=HTMLResponse)
async def player():
    with open("templates/player.html", encoding="utf-8") as f:
        return f.read()


@app.get("/recordings")
async def list_recordings():
    files = sorted(RECORDINGS_DIR.glob("*.echorec"), reverse=True)
    result = []
    for f in files:
        size  = f.stat().st_size
        fs    = frame_size(settings.num_samples)
        total = max(0, (size - HEADER_SIZE) // fs)
        result.append({"name": f.name, "size": size, "frames": total})
    return JSONResponse(result)


@app.websocket("/play/{filename}")
async def play_recording(ws: WebSocket, filename: str, speed: float = 1.0):
    """Stream a recording file to WebSocket client as JSON frames."""
    await ws.accept()
    path = RECORDINGS_DIR / filename
    if not path.exists() or not path.name.endswith(".echorec"):
        await ws.send_text(json.dumps({"error": "file not found"}))
        await ws.close()
        return

    try:
        with open(path, "rb") as f:
            # Read and validate header
            hdr_bytes = f.read(HEADER_SIZE)
            magic, version, num_samples, sos, sample_time = struct.unpack(HEADER_FMT, hdr_bytes)
            if magic != MAGIC:
                await ws.send_text(json.dumps({"error": "invalid file"}))
                return

            resolution = (sos * sample_time * 100) / 2
            fs   = frame_size(num_samples)
            size = path.stat().st_size
            total_frames = max(0, (size - HEADER_SIZE) // fs)

            await ws.send_text(json.dumps({
                "type": "header",
                "num_samples": num_samples,
                "resolution": resolution,
                "total_frames": total_frames,
            }))

            t_prev = None
            frame_idx = 0
            import time

            while True:
                raw = f.read(fs)
                if len(raw) < fs:
                    break

                ts, depth_idx, temp_scaled, vdrv_scaled = struct.unpack(
                    FRAME_HDR, raw[:FRAME_HDR_S])
                samples = list(raw[FRAME_HDR_S:FRAME_HDR_S + num_samples])

                depth_m = depth_idx * (resolution / 100)

                # Pace playback at original speed (adjusted by speed factor)
                if t_prev is not None and speed > 0:
                    delay = (ts - t_prev) / speed
                    if 0 < delay < 5:
                        await asyncio.sleep(delay)
                t_prev = ts

                await ws.send_text(json.dumps({
                    "type": "frame",
                    "spectrogram": samples,
                    "measured_depth": round(depth_m, 3),
                    "temperature": temp_scaled / 100.0,
                    "drive_voltage": vdrv_scaled / 100.0,
                    "resolution": resolution,
                    "frame": frame_idx,
                    "total_frames": total_frames,
                    "timestamp": round(ts, 2),
                }))
                frame_idx += 1

        await ws.send_text(json.dumps({"type": "done"}))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        log.error(f"Playback error: {e}")


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    _clients.append(ws)
    log.info(f"WebSocket connected. Total clients: {len(_clients)}")
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        if ws in _clients:
            _clients.remove(ws)
        log.info(f"WebSocket disconnected. Total clients: {len(_clients)}")
