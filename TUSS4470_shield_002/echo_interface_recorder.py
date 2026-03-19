import sys
import os
import numpy as np
import serial
import serial.tools.list_ports
import struct
import time
import socket
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QWidget,
    QComboBox,
    QPushButton,
    QLabel,
    QLineEdit,
    QFileDialog,
    QMessageBox,
)
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
import pyqtgraph as pg
import qdarktheme
from PyQt5.QtWidgets import QHBoxLayout
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtWidgets import QVBoxLayout, QLabel, QCheckBox, QLineEdit
from PyQt5.QtWidgets import QApplication

# Serial Configuration
BAUD_RATE = 250000
NUM_SAMPLES = 750  # (X-axis)

MAX_ROWS = 300  # Number of time steps (Y-axis)
Y_LABEL_DISTANCE = 50  # distance between labels in cm

SPEED_OF_SOUND = 343  # default sound speed meters/second in air

SAMPLE_TIME = 13.2e-6     # 13.2 microseconds on Atmega328 max sample speed

DEFAULT_LEVELS = (0, 256)  # Expected data range

SAMPLE_RESOLUTION = (SPEED_OF_SOUND * SAMPLE_TIME * 100) / 2  # cm per row
PACKET_SIZE = 1 + 6 + NUM_SAMPLES + 1  # header + payload + checksum
MAX_DEPTH = NUM_SAMPLES * SAMPLE_RESOLUTION  # Total depth in cm
depth_labels = {int(i / SAMPLE_RESOLUTION): f"{i / 100}" for i in range(0, int(MAX_DEPTH), Y_LABEL_DISTANCE)}

# Recording file format:
# Each frame: [timestamp:float64 8 bytes][raw_packet: PACKET_SIZE bytes]
# This allows exact replay at original timing or fast-forward playback.
FRAME_HEADER_SIZE = 8  # float64 timestamp
FRAME_TOTAL_SIZE = FRAME_HEADER_SIZE + PACKET_SIZE


def read_packet(ser):
    while True:
        header = ser.read(1)
        if header != b"\xaa":
            continue

        payload = ser.read(6 + NUM_SAMPLES)
        checksum = ser.read(1)

        if len(payload) != 6 + NUM_SAMPLES or len(checksum) != 1:
            continue

        calc_checksum = 0
        for byte in payload:
            calc_checksum ^= byte
        if calc_checksum != checksum[0]:
            print("Checksum mismatch: {} != {}".format(calc_checksum, checksum[0]))
            continue

        depth, temp_scaled, vDrv_scaled = struct.unpack("<HhH", payload[:6])
        depth = min(depth, NUM_SAMPLES)

        sample_bytes = payload[6:6 + NUM_SAMPLES]
        values = np.frombuffer(sample_bytes, dtype=np.uint8, count=NUM_SAMPLES)

        temperature = temp_scaled / 100.0
        drive_voltage = vDrv_scaled / 100.0

        return values, depth, temperature, drive_voltage


def parse_raw_packet(raw_packet):
    """Parse a raw PACKET_SIZE byte packet. Returns (values, depth, temp, vdrv) or None."""
    if len(raw_packet) != PACKET_SIZE:
        return None
    if raw_packet[0] != 0xAA:
        return None

    payload = raw_packet[1:1 + 6 + NUM_SAMPLES]
    checksum = raw_packet[-1]
    calc = 0
    for b in payload:
        calc ^= b
    if calc != checksum:
        return None

    try:
        depth, temp_scaled, vDrv_scaled = struct.unpack("<HhH", payload[:6])
        depth = min(depth, NUM_SAMPLES)
        sample_bytes = payload[6:6 + NUM_SAMPLES]
        values = np.frombuffer(sample_bytes, dtype=np.uint8, count=NUM_SAMPLES)
        return values, depth, temp_scaled / 100.0, vDrv_scaled / 100.0
    except struct.error:
        return None


def get_serial_ports():
    return [port.device for port in serial.tools.list_ports.comports()][::-1]


def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


# ─────────────────────────────────────────────────
# SerialReader — reads from USB/COM and optionally
# writes raw frames to a recording file.
# ─────────────────────────────────────────────────
class SerialReader(QThread):
    data_received = pyqtSignal(np.ndarray, float, float, float)

    def __init__(self, port, baud_rate, record_file=None):
        super().__init__()
        self.port = port
        self.baud_rate = baud_rate
        self.running = True
        self.record_file = record_file  # open file object or None
        self._record_start = None

    def run(self):
        try:
            with serial.Serial(self.port, BAUD_RATE, timeout=1) as ser:
                print("Serial connected")
                self._record_start = time.time()
                while self.running:
                    # Read raw packet byte-by-byte (sync on 0xAA)
                    header = ser.read(1)
                    if header != b"\xaa":
                        continue
                    rest = ser.read(6 + NUM_SAMPLES + 1)
                    if len(rest) != 6 + NUM_SAMPLES + 1:
                        continue
                    raw_packet = header + rest

                    # Write to recording file if active
                    if self.record_file:
                        ts = time.time() - self._record_start
                        self.record_file.write(struct.pack("<d", ts))
                        self.record_file.write(raw_packet)
                        self.record_file.flush()

                    result = parse_raw_packet(raw_packet)
                    if result:
                        values, depth, temperature, drive_voltage = result
                        print(f"Depth: {depth}, Temp: {temperature}C, Vdrv: {drive_voltage}V")
                        self.data_received.emit(values, depth, temperature, drive_voltage)

        except serial.SerialException as e:
            print(f"Serial Error: {e}")

    def stop(self):
        self.running = False
        self.quit()
        self.wait()


# ─────────────────────────────────────────────────
# FilePlaybackReader — replays a recording file.
# Emits data_received at original timing (real-time)
# or as fast as possible (fast mode).
# ─────────────────────────────────────────────────
class FilePlaybackReader(QThread):
    data_received = pyqtSignal(np.ndarray, float, float, float)
    playback_finished = pyqtSignal()
    progress_updated = pyqtSignal(int, int)  # current_frame, total_frames

    def __init__(self, filepath, fast=False):
        super().__init__()
        self.filepath = filepath
        self.fast = fast  # if True: ignore timestamps, emit as fast as possible
        self.running = True

    def run(self):
        try:
            file_size = os.path.getsize(self.filepath)
            total_frames = file_size // FRAME_TOTAL_SIZE
            if total_frames == 0:
                print("Recording file is empty or incompatible.")
                self.playback_finished.emit()
                return

            print(f"Playback: {total_frames} frames from '{self.filepath}'")

            with open(self.filepath, "rb") as f:
                playback_start = time.time()
                first_ts = None
                frame_index = 0

                while self.running:
                    header_bytes = f.read(FRAME_HEADER_SIZE)
                    if len(header_bytes) < FRAME_HEADER_SIZE:
                        break  # End of file

                    (recorded_ts,) = struct.unpack("<d", header_bytes)
                    raw_packet = f.read(PACKET_SIZE)
                    if len(raw_packet) < PACKET_SIZE:
                        break

                    if first_ts is None:
                        first_ts = recorded_ts

                    # Real-time pacing: sleep until the frame's original timestamp
                    if not self.fast:
                        target_wall = playback_start + (recorded_ts - first_ts)
                        sleep_secs = target_wall - time.time()
                        if sleep_secs > 0:
                            time.sleep(sleep_secs)

                    result = parse_raw_packet(raw_packet)
                    if result:
                        values, depth, temperature, drive_voltage = result
                        self.data_received.emit(values, depth, temperature, drive_voltage)

                    frame_index += 1
                    self.progress_updated.emit(frame_index, total_frames)

            self.playback_finished.emit()
            print("Playback finished.")

        except Exception as e:
            print(f"Playback error: {e}")
            self.playback_finished.emit()

    def stop(self):
        self.running = False
        self.quit()
        self.wait()


class UDPReader(QThread):
    data_received = pyqtSignal(np.ndarray, float, float, float)

    def __init__(self, port: int, timeout: float = 1.0):
        super().__init__()
        self.host = ""
        self.port = port
        self.timeout = timeout
        self.running = True
        self._sock = None

    def run(self):
        try:
            import socket as _socket
            self._sock = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
            self._sock.settimeout(self.timeout)
            self._sock.bind((self.host, self.port))
            print(f"UDP listener bound to {self.host}:{self.port}")
            RECV_SIZE = PACKET_SIZE
            packet_buf = bytearray()
            packets_ok = 0
            checksum_errors = 0

            while self.running:
                try:
                    datagram, _addr = self._sock.recvfrom(RECV_SIZE)
                except _socket.timeout:
                    continue

                for byte in datagram:
                    if not packet_buf:
                        if byte == 0xAA:
                            packet_buf.append(byte)
                        else:
                            continue
                    else:
                        packet_buf.append(byte)

                    if len(packet_buf) == PACKET_SIZE:
                        payload = packet_buf[1:1 + 6 + NUM_SAMPLES]
                        checksum = packet_buf[-1]
                        calc = 0
                        for b in payload:
                            calc ^= b
                        if calc == checksum:
                            try:
                                depth, temp_scaled, vDrv_scaled = struct.unpack("<HhH", payload[:6])
                                depth = min(depth, NUM_SAMPLES)
                                sample_bytes = payload[6:6 + NUM_SAMPLES]
                                if len(sample_bytes) == NUM_SAMPLES:
                                    values = np.frombuffer(sample_bytes, dtype=np.uint8, count=NUM_SAMPLES)
                                    temperature = temp_scaled / 100.0
                                    drive_voltage = vDrv_scaled / 100.0
                                    self.data_received.emit(values, depth, temperature, drive_voltage)
                                    packets_ok += 1
                            except struct.error:
                                checksum_errors += 1
                        else:
                            checksum_errors += 1

                        last_byte = packet_buf[-1]
                        packet_buf.clear()
                        if last_byte == 0xAA:
                            packet_buf.append(last_byte)

        except Exception as e:
            print(f"UDP Reader error: {e}")
        finally:
            if self._sock:
                try:
                    self._sock.close()
                except Exception:
                    pass

    def stop(self):
        self.running = False
        if self._sock:
            try:
                import socket as _socket
                with _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM) as s:
                    s.sendto(b"\x00", (self.host or "127.0.0.1", self.port))
            except Exception:
                pass
        self.quit()
        self.wait()


class SettingsDialog(QWidget):
    def __init__(self, parent=None, current_gradient='cyclic', current_speed=343,
                 nmea_enabled=False, nmea_port=10110, nmea_address="127.0.0.1"):
        super().__init__(parent)
        self.setWindowTitle("Chart Settings")
        self.setFixedSize(320, 550)

        self.main_app = parent

        outer_layout = QVBoxLayout(self)
        outer_layout.setAlignment(Qt.AlignCenter)

        card = QWidget()
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(20, 20, 20, 20)
        card_layout.setSpacing(15)

        card_layout.addWidget(QLabel("Color Map:"))
        self.gradient_dropdown = QComboBox()
        self.gradient_dropdown.addItems([
            "viridis", "plasma", "inferno", "magma", "thermal",
            "flame", "yellowy", "bipolar", "spectrum", "cyclic", "greyclip", "grey",
        ])
        self.gradient_dropdown.setCurrentText(current_gradient)
        card_layout.addWidget(self.gradient_dropdown)

        card_layout.addWidget(QLabel("Speed of Sound:"))
        self.speed_dropdown = QComboBox()
        self.speed_dropdown.addItems(["343m/s (Air)", "1440m/s (Water)"])
        self.speed_dropdown.setCurrentIndex(1 if current_speed == 1440 else 0)
        card_layout.addWidget(self.speed_dropdown)

        nmea_section = QVBoxLayout()
        nmea_section.setSpacing(8)

        nmea_label = QLabel("NMEA TCP Output:")
        nmea_label.setStyleSheet("font-weight: bold;")
        nmea_section.addWidget(nmea_label)

        self.nmea_enable_checkbox = QCheckBox("Enable NMEA Output")
        nmea_section.addWidget(self.nmea_enable_checkbox)

        addr_row = QHBoxLayout()
        addr_label = QLabel("Address:")
        addr_label.setMinimumWidth(60)
        self.addr_display = QLabel(nmea_address)
        self.addr_display.setStyleSheet("color: #cccccc; padding: 2px;")
        self.addr_display.setTextInteractionFlags(Qt.TextSelectableByMouse)
        copy_button = QPushButton("Copy")
        copy_button.setFixedHeight(22)
        copy_button.setStyleSheet("font-size: 11px; padding: 2px 6px;")
        copy_button.clicked.connect(lambda: QApplication.clipboard().setText(nmea_address))
        addr_row.addWidget(addr_label)
        addr_row.addWidget(self.addr_display)
        addr_row.addWidget(copy_button)
        addr_row.addStretch()
        nmea_section.addLayout(addr_row)

        port_row = QHBoxLayout()

        self.large_depth_checkbox = QCheckBox("Show Depth Display")
        self.large_depth_checkbox.setChecked(getattr(parent, "large_depth_visible", True))
        card_layout.addWidget(self.large_depth_checkbox)

        port_label = QLabel("Port:")
        port_label.setMinimumWidth(40)
        self.port_input = QLineEdit()
        self.port_input.setPlaceholderText("TCP Port (default: 10110)")
        self.port_input.setText(str(nmea_port))
        self.port_input.setMaximumWidth(200)
        port_row.addWidget(port_label)
        port_row.addWidget(self.port_input)
        port_row.addStretch()
        nmea_section.addLayout(port_row)

        self.nmea_enable_checkbox.toggled.connect(self.port_input.setEnabled)
        self.nmea_enable_checkbox.setChecked(nmea_enabled)
        self.port_input.setEnabled(nmea_enabled)

        card_layout.addLayout(nmea_section)

        button_layout = QHBoxLayout()
        apply_button = QPushButton("Apply")
        apply_button.clicked.connect(self.apply_settings)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.close)
        button_layout.addWidget(apply_button)
        button_layout.addWidget(cancel_button)
        card_layout.addLayout(button_layout)

        outer_layout.addWidget(card)

        self.setStyleSheet("""
            QWidget#Card {
                background-color: #2b2b2b;
                border-radius: 12px;
                padding: 15px;
            }
            QLabel { color: #ffffff; font-size: 14px; }
            QComboBox { background-color: #3c3c3c; color: white; padding: 4px; border-radius: 4px; }
            QPushButton { background-color: #444444; border: 1px solid #666; padding: 5px 10px; border-radius: 6px; }
            QPushButton:hover { background-color: #555; }
        """)
        card.setObjectName("Card")
        self.setLayout(outer_layout)

    def apply_settings(self):
        selected_gradient = self.gradient_dropdown.currentText()
        selected_speed = 343 if self.speed_dropdown.currentIndex() == 0 else 1440
        nmea_enabled = self.nmea_enable_checkbox.isChecked()
        nmea_port = int(self.port_input.text()) if self.port_input.text().isdigit() else 10110

        if self.main_app:
            self.main_app.set_gradient(selected_gradient)
            self.main_app.set_sound_speed(selected_speed)
            self.main_app.configure_nmea_output(enabled=nmea_enabled, port=nmea_port)
            self.main_app.set_large_depth_display(self.large_depth_checkbox.isChecked())

        self.close()


class WaterfallApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.serial_thread = None
        self.udp_thread = None
        self.playback_thread = None

        self.nmea_enabled = False
        self.nmea_port = 10110
        self.nmea_socket = None
        self.nmea_output_enabled = False

        self.current_gradient = 'cyclic'
        self.current_speed = SPEED_OF_SOUND

        # Recording state
        self._record_file = None
        self._record_filepath = None

        self.setWindowTitle("Open Echo Interface (with Recording)")
        self.setGeometry(0, 0, 480, 900)

        self.data = np.zeros((MAX_ROWS, NUM_SAMPLES))

        self.setAttribute(Qt.WA_TranslucentBackground, False)
        self.setWindowFlags(self.windowFlags() & ~Qt.FramelessWindowHint)

        palette = self.palette()
        palette.setColor(QPalette.Window, QColor("#2b2b2b"))
        self.setPalette(palette)
        self.setAutoFillBackground(True)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)
        central_widget.setLayout(main_layout)

        # === Waterfall Plot ===
        self.waterfall = pg.PlotWidget()
        self.imageitem = pg.ImageItem(axisOrder="row-major")
        self.waterfall.addItem(self.imageitem)
        self.waterfall.setMouseEnabled(x=False, y=False)
        self.waterfall.setMinimumHeight(400)
        self.waterfall.invertY(True)
        main_layout.addWidget(self.waterfall)

        inverted_depth_labels = list(depth_labels.items())[::-1]
        self.waterfall.getAxis("left").setTicks([inverted_depth_labels])
        self.depth_line = pg.InfiniteLine(angle=0, pen=pg.mkPen("r", width=2))
        self.waterfall.addItem(self.depth_line)

        right_axis = self.waterfall.getAxis("right")
        right_axis.setTicks([inverted_depth_labels])
        right_axis.setStyle(showValues=True)

        for i in range(0, int(MAX_DEPTH), Y_LABEL_DISTANCE):
            row_index = int(i / SAMPLE_RESOLUTION)
            hline = pg.InfiniteLine(
                pos=row_index,
                angle=0,
                pen=pg.mkPen(color="w", style=pg.QtCore.Qt.DotLine),
            )
            self.waterfall.addItem(hline)

        self.colorbar = pg.HistogramLUTWidget()
        self.colorbar.setImageItem(self.imageitem)
        self.colorbar.item.gradient.loadPreset("cyclic")
        self.imageitem.setLevels(DEFAULT_LEVELS)

        controls_layout = QVBoxLayout()

        # === Serial row ===
        serial_row = QHBoxLayout()

        self.large_depth_label = QLabel("--- m")
        self.large_depth_label.setAlignment(Qt.AlignCenter)
        self.large_depth_label.setStyleSheet("""
            QLabel {
                color: #00ffcc;
                font-size: 64px;
                font-weight: bold;
            }
        """)
        self.large_depth_label.setVisible(True)
        serial_row.addWidget(self.large_depth_label)

        serial_row.addWidget(QLabel("Port:"))
        self.serial_dropdown = QComboBox()
        ports = get_serial_ports()
        self.serial_dropdown.addItems(ports)
        self.serial_dropdown.setMinimumWidth(150)
        serial_row.addWidget(self.serial_dropdown)

        self.connect_button = QPushButton("Connect")
        self.connect_button.clicked.connect(self.toggle_serial_connection)
        serial_row.addWidget(self.connect_button)

        controls_layout.addLayout(serial_row)

        # === UDP row ===
        udp_row = QHBoxLayout()
        udp_row.addWidget(QLabel("UDP Port:"))
        self.udp_port_input = QLineEdit()
        self.udp_port_input.setText("5005")
        self.udp_port_input.setMaximumWidth(100)
        udp_row.addWidget(self.udp_port_input)
        self.udp_connect_button = QPushButton("Connect UDP")
        self.udp_connect_button.clicked.connect(self.toggle_udp_connection)
        udp_row.addWidget(self.udp_connect_button)
        controls_layout.addLayout(udp_row)

        # === Recording row ===
        rec_row = QHBoxLayout()

        self.record_button = QPushButton("Start Recording")
        self.record_button.setStyleSheet("background-color: #1a5c1a; color: white;")
        self.record_button.clicked.connect(self.toggle_recording)
        rec_row.addWidget(self.record_button)

        self.record_status_label = QLabel("Not recording")
        self.record_status_label.setStyleSheet("color: #888888; font-size: 11px;")
        rec_row.addWidget(self.record_status_label)

        controls_layout.addLayout(rec_row)

        # === Playback row ===
        play_row = QHBoxLayout()

        self.open_file_button = QPushButton("Open Recording...")
        self.open_file_button.clicked.connect(self.open_recording_file)
        play_row.addWidget(self.open_file_button)

        self.playback_file_label = QLabel("No file selected")
        self.playback_file_label.setStyleSheet("color: #888888; font-size: 11px;")
        self.playback_file_label.setMaximumWidth(200)
        play_row.addWidget(self.playback_file_label)

        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(lambda: self.start_playback(fast=False))
        self.play_button.setEnabled(False)
        play_row.addWidget(self.play_button)

        self.play_fast_button = QPushButton("Play Fast")
        self.play_fast_button.clicked.connect(lambda: self.start_playback(fast=True))
        self.play_fast_button.setEnabled(False)
        play_row.addWidget(self.play_fast_button)

        self.stop_playback_button = QPushButton("Stop")
        self.stop_playback_button.clicked.connect(self.stop_playback)
        self.stop_playback_button.setEnabled(False)
        play_row.addWidget(self.stop_playback_button)

        controls_layout.addLayout(play_row)

        # === Playback progress label ===
        self.playback_progress_label = QLabel("")
        self.playback_progress_label.setAlignment(Qt.AlignCenter)
        self.playback_progress_label.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        controls_layout.addWidget(self.playback_progress_label)

        # === Info labels ===
        info_layout = QHBoxLayout()
        self.depth_label = QLabel("Depth: --- cm")
        self.temperature_label = QLabel("Temperature: --- C")
        self.drive_voltage_label = QLabel("vDRV: --- V")
        info_layout.addWidget(self.depth_label)
        info_layout.addWidget(self.temperature_label)
        info_layout.addWidget(self.drive_voltage_label)
        info_container = QWidget()
        info_container.setLayout(info_layout)
        controls_layout.addWidget(info_container)

        # === Hex input + Settings + Quit ===
        hex_row = QHBoxLayout()
        self.hex_input = QLineEdit()
        self.hex_input.setPlaceholderText("0x1F")
        hex_row.addWidget(self.hex_input)
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.send_hex_value)
        hex_row.addWidget(self.send_button)
        self.settings_button = QPushButton("Settings")
        self.settings_button.clicked.connect(self.open_settings)
        hex_row.addWidget(self.settings_button)
        self.quit_button = QPushButton("Quit")
        self.quit_button.clicked.connect(self.close)
        hex_row.addWidget(self.quit_button)
        controls_layout.addLayout(hex_row)

        controls_container = QWidget()
        controls_container.setLayout(controls_layout)
        main_layout.addWidget(controls_container)

    # ─── Recording ────────────────────────────────

    def toggle_recording(self):
        if self._record_file is not None:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self):
        if self.serial_thread is None or not self.serial_thread.isRunning():
            QMessageBox.warning(self, "Not Connected",
                                "Connect to a serial port before starting recording.")
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Recording As", "",
            "Echo Recording (*.echorec);;All Files (*)"
        )
        if not filepath:
            return

        if not filepath.endswith(".echorec"):
            filepath += ".echorec"

        try:
            self._record_file = open(filepath, "wb")
            self._record_filepath = filepath

            # Pass the open file to the running serial thread
            self.serial_thread.record_file = self._record_file
            self.serial_thread._record_start = time.time()

            self.record_button.setText("Stop Recording")
            self.record_button.setStyleSheet("background-color: #8b0000; color: white;")
            filename = os.path.basename(filepath)
            self.record_status_label.setText(f"Recording: {filename}")
            self.record_status_label.setStyleSheet("color: #ff4444; font-size: 11px;")
            print(f"Recording to: {filepath}")

        except Exception as e:
            QMessageBox.critical(self, "Recording Error", f"Could not open file:\n{e}")

    def _stop_recording(self):
        if self.serial_thread:
            self.serial_thread.record_file = None

        if self._record_file:
            self._record_file.close()
            self._record_file = None

        filepath = self._record_filepath
        self._record_filepath = None

        self.record_button.setText("Start Recording")
        self.record_button.setStyleSheet("background-color: #1a5c1a; color: white;")
        self.record_status_label.setText("Not recording")
        self.record_status_label.setStyleSheet("color: #888888; font-size: 11px;")
        print(f"Recording saved: {filepath}")

    # ─── Playback ─────────────────────────────────

    def open_recording_file(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open Recording", "",
            "Echo Recording (*.echorec);;All Files (*)"
        )
        if not filepath:
            return

        file_size = os.path.getsize(filepath)
        total_frames = file_size // FRAME_TOTAL_SIZE

        if total_frames == 0:
            QMessageBox.warning(self, "Invalid File",
                                "File is empty or not a valid .echorec recording.")
            return

        self._playback_filepath = filepath
        filename = os.path.basename(filepath)
        self.playback_file_label.setText(filename)
        self.play_button.setEnabled(True)
        self.play_fast_button.setEnabled(True)
        self.playback_progress_label.setText(f"Ready: {total_frames} frames")
        print(f"Loaded recording: {filepath} ({total_frames} frames)")

    def start_playback(self, fast=False):
        if not hasattr(self, '_playback_filepath') or not self._playback_filepath:
            return

        # Stop any active connections before playback
        self._disconnect_all()

        if self.playback_thread and self.playback_thread.isRunning():
            self.playback_thread.stop()
            self.playback_thread = None

        # Reset waterfall
        self.data = np.zeros((MAX_ROWS, NUM_SAMPLES))

        self.playback_thread = FilePlaybackReader(self._playback_filepath, fast=fast)
        self.playback_thread.data_received.connect(self.waterfall_plot_callback)
        self.playback_thread.playback_finished.connect(self._on_playback_finished)
        self.playback_thread.progress_updated.connect(self._on_playback_progress)
        self.playback_thread.start()

        mode = "fast" if fast else "real-time"
        self.playback_progress_label.setText(f"Playing ({mode})...")
        self.play_button.setEnabled(False)
        self.play_fast_button.setEnabled(False)
        self.stop_playback_button.setEnabled(True)
        print(f"Playback started ({mode})")

    def stop_playback(self):
        if self.playback_thread and self.playback_thread.isRunning():
            self.playback_thread.stop()
            self.playback_thread = None
        self._on_playback_finished()

    def _on_playback_finished(self):
        self.play_button.setEnabled(True)
        self.play_fast_button.setEnabled(True)
        self.stop_playback_button.setEnabled(False)
        self.playback_progress_label.setText("Playback finished.")

    def _on_playback_progress(self, current, total):
        self.playback_progress_label.setText(f"Frame {current} / {total}")

    # ─── Serial / UDP connections ──────────────────

    def _disconnect_all(self):
        if self.serial_thread and self.serial_thread.isRunning():
            self._stop_recording()
            self.serial_thread.stop()
            self.serial_thread = None
            self.connect_button.setText("Connect")
        if self.udp_thread and self.udp_thread.isRunning():
            self.udp_thread.stop()
            self.udp_thread = None
            self.udp_connect_button.setText("Connect UDP")

    def connect_serial(self):
        if self.serial_thread:
            self.serial_thread.stop()
            self.serial_thread = None

        selected_port = self.serial_dropdown.currentText()
        try:
            self.serial_thread = SerialReader(selected_port, BAUD_RATE, record_file=None)
            self.serial_thread.data_received.connect(self.waterfall_plot_callback)
            self.serial_thread.start()
            print(f"Connected to {selected_port}")
        except Exception as e:
            print(f"Connection failed: {e}")

    def toggle_serial_connection(self):
        if self.serial_thread and self.serial_thread.isRunning():
            self._stop_recording()
            self.serial_thread.stop()
            self.serial_thread = None
            self.connect_button.setText("Connect")
        else:
            self.connect_serial()
            if self.serial_thread and self.serial_thread.isRunning():
                self.connect_button.setText("Disconnect")

    def disconnect_serial(self):
        if self.serial_thread:
            try:
                self._stop_recording()
                self.serial_thread.stop()
                self.serial_thread.wait()
                self.serial_thread = None
                print("Disconnected from serial device")
            except Exception as e:
                print(f"Disconnection failed: {e}")

    def connect_udp(self):
        if self.udp_thread:
            self.udp_thread.stop()
            self.udp_thread = None
        try:
            udp_port = int(self.udp_port_input.text())
            self.udp_thread = UDPReader(port=udp_port)
            self.udp_thread.data_received.connect(self.waterfall_plot_callback)
            self.udp_thread.start()
            print(f"UDP listener started on port {udp_port}")
        except Exception as e:
            print(f"Failed to start UDP listener: {e}")

    def disconnect_udp(self):
        if self.udp_thread:
            self.udp_thread.stop()
            self.udp_thread = None
            print("UDP listener stopped")

    def toggle_udp_connection(self):
        if self.udp_thread and self.udp_thread.isRunning():
            self.disconnect_udp()
            self.udp_connect_button.setText("Connect UDP")
        else:
            self.connect_udp()
            if self.udp_thread and self.udp_thread.isRunning():
                self.udp_connect_button.setText("Disconnect UDP")

    # ─── Plot callback ─────────────────────────────

    def waterfall_plot_callback(self, spectrogram, depth_index, temperature, drive_voltage):
        self.data = np.roll(self.data, -1, axis=0)
        self.data[-1, :] = spectrogram
        self.imageitem.setImage(self.data.T, autoLevels=False)

        sigma = np.std(self.data)
        mean = np.mean(self.data)
        self.imageitem.setLevels((mean - 2 * sigma, mean + 2 * sigma))

        depth_cm = depth_index * SAMPLE_RESOLUTION
        self.depth_label.setText(f"Depth: {depth_cm:.1f} cm | Index: {depth_index:.0f}")
        self.temperature_label.setText(f"Temperature: {temperature:.1f} C")
        self.drive_voltage_label.setText(f"vDRV: {drive_voltage:.1f} V")
        self.depth_line.setPos(depth_index)

        if self.large_depth_label.isVisible():
            self.large_depth_label.setText(f"{depth_cm / 100:.1f} m")

        if hasattr(self, 'nmea_output_enabled') and self.nmea_output_enabled:
            now = time.time()
            if not hasattr(self, "_last_nmea_sent") or (now - self._last_nmea_sent) >= 1.0:
                try:
                    depth_m = depth_cm / 100
                    depth_ft = depth_m * 3.28084
                    depth_fathoms = depth_m * 0.546807

                    def calculate_checksum(sentence):
                        checksum = 0
                        for char in sentence:
                            checksum ^= ord(char)
                        return f"*{checksum:02X}"

                    nmea_sentence = f"DBT,{depth_ft:.1f},f,{depth_m:.1f},M,{depth_fathoms:.1f},F"
                    full_sentence = f"${nmea_sentence}{calculate_checksum(nmea_sentence)}\r\n"
                    self.nmea_client_socket.sendall(full_sentence.encode("ascii"))
                    self._last_nmea_sent = now
                except Exception as e:
                    print(f"NMEA send failed: {e}")

    # ─── Settings ─────────────────────────────────

    def set_large_depth_display(self, enabled: bool):
        self.large_depth_visible = enabled
        self.large_depth_label.setVisible(enabled)

    def configure_nmea_output(self, enabled: bool, port: int):
        self.nmea_output_enabled = enabled
        self.nmea_port = port

        if hasattr(self, "nmea_client_socket") and self.nmea_client_socket:
            try:
                self.nmea_client_socket.close()
            except Exception:
                pass
            self.nmea_client_socket = None

        if hasattr(self, "nmea_server_socket") and self.nmea_server_socket:
            try:
                self.nmea_server_socket.close()
            except Exception:
                pass
            self.nmea_server_socket = None

        if enabled:
            try:
                self.nmea_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.nmea_server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.nmea_server_socket.bind(("0.0.0.0", port))
                self.nmea_server_socket.listen(1)
                print(f"Waiting for TCP NMEA connection on port {port}...")
                self.nmea_client_socket, _ = self.nmea_server_socket.accept()
                print(f"NMEA client connected on port {port}")
            except Exception as e:
                print(f"Failed to set up NMEA output: {e}")
                self.nmea_output_enabled = False

    def set_gradient(self, gradient_name):
        self.current_gradient = gradient_name
        self.colorbar.item.gradient.loadPreset(gradient_name)

    def set_sound_speed(self, speed):
        global SPEED_OF_SOUND, SAMPLE_RESOLUTION, MAX_DEPTH, depth_labels

        SPEED_OF_SOUND = speed
        self.current_speed = speed
        SAMPLE_RESOLUTION = (SPEED_OF_SOUND * SAMPLE_TIME * 100) / 2
        MAX_DEPTH = NUM_SAMPLES * SAMPLE_RESOLUTION
        depth_labels = {
            int(i / SAMPLE_RESOLUTION): f"{i / 100}"
            for i in range(0, int(MAX_DEPTH), Y_LABEL_DISTANCE)
        }

        inverted_depth_labels = list(depth_labels.items())[::-1]
        self.waterfall.getAxis("left").setTicks([inverted_depth_labels])
        self.waterfall.getAxis("right").setTicks([inverted_depth_labels])

    def keyPressEvent(self, event):
        if event.key() == ord("Q"):
            self.close()
        elif event.key() == ord("C"):
            self.connect_button.click()
        else:
            super().keyPressEvent(event)

    def send_hex_value(self):
        hex_value = self.hex_input.text().strip()
        if hex_value.startswith("0x") and len(hex_value) > 2:
            try:
                if self.serial_thread and self.serial_thread.isRunning():
                    with serial.Serial(self.serial_dropdown.currentText(), BAUD_RATE) as ser:
                        ser.write(hex_value.encode())
                        print(f"Sent: {hex_value}")
            except ValueError:
                print("Invalid hex format.")
        else:
            print("Invalid hex value. Please enter a valid hex string (e.g., 0x1F)")

    def closeEvent(self, event):
        self._stop_recording()
        if self.serial_thread:
            self.serial_thread.stop()
        if self.udp_thread:
            self.udp_thread.stop()
        if self.playback_thread and self.playback_thread.isRunning():
            self.playback_thread.stop()
        event.accept()

    def open_settings(self):
        device_ip = get_local_ip()
        self.settings_dialog = SettingsDialog(
            parent=self,
            current_gradient=self.current_gradient,
            current_speed=self.current_speed,
            nmea_enabled=self.nmea_output_enabled,
            nmea_port=self.nmea_port,
            nmea_address=device_ip,
        )
        self.settings_dialog.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    try:
        setup_theme = getattr(qdarktheme, "setup_theme", None)
        if callable(setup_theme):
            setup_theme("dark")
        else:
            from qdarktheme.base import load_stylesheet
            app.setStyleSheet(load_stylesheet("dark"))
    except Exception as exc:
        print(f"Theme setup failed: {exc}")

    window = WaterfallApp()
    window.show()
    sys.exit(app.exec())
