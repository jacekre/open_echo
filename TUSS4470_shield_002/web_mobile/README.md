# Open Echo — mobile web interface

Minimal waterfall display accessible from any browser, including Android tablets.

## Quick start

```bash
cd TUSS4470_shield_002/web_mobile
pip install -r requirements.txt

# Serial (auto-detects first available port)
uvicorn app:app --host 0.0.0.0 --port 8000

# Serial — explicit port
ECHO_PORT=COM3 uvicorn app:app --host 0.0.0.0 --port 8000

# UDP (e.g. Arduino R4 WiFi or Pico W)
ECHO_CONNECTION=udp ECHO_UDP_PORT=5005 uvicorn app:app --host 0.0.0.0 --port 8000
```

Open on Android tablet: `http://<PC-IP>:8000`

## Environment variables

| Variable          | Default      | Description                        |
|-------------------|--------------|------------------------------------|
| `ECHO_CONNECTION` | `serial`     | `serial` or `udp`                  |
| `ECHO_PORT`       | auto-detect  | Serial port (COM3, /dev/ttyUSB0…)  |
| `ECHO_BAUD`       | `250000`     | Serial baud rate                   |
| `ECHO_UDP_HOST`   | `0.0.0.0`    | UDP bind address                   |
| `ECHO_UDP_PORT`   | `5005`       | UDP port                           |
| `ECHO_SAMPLES`    | `750`        | Samples per frame (match firmware) |
| `ECHO_SOS`        | `343`        | Speed of sound m/s (343=air, 1440=water) |
| `ECHO_SAMPLE_TIME`| `13.2e-6`    | ADC sample time in seconds         |

## Files

```
web_mobile/
├── app.py          — FastAPI server, WebSocket broadcast
├── echo.py         — Serial/UDP reader, packet parser
├── requirements.txt
├── templates/
│   └── index.html  — Single-page mobile UI (no external dependencies)
└── static/         — (empty, reserved for future assets)
```

## Features

- Mobile-first layout — works on Android tablet and phone browsers
- Pointer Events API — cursor line works with both touch and mouse
- Auto-reconnect WebSocket — survives server restarts
- Auto-gain (Welford running stats) — no manual brightness adjustment
- Auto-zoom — depth range adjusts to measured depth
- Manual zoom buttons (+/-) sized for touch (40×40 px)
- No external JS libraries — single HTML file, loads instantly on slow WiFi
- Auto-detect serial port on startup
