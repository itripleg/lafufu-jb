import argparse
import json
import math
import random
import threading
import platform
import socket 
import subprocess
import sys
import time
import wave
from collections import deque
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import pyaudio
import pyttsx3
import whisper

# ------------------ Audio / VAD ------------------
CHUNK = 600
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
OUTPUT_FILENAME = "temp.wav"

SILENCE_THRESHOLD = 800
SILENCE_CHUNKS = int(1.5 * (RATE / CHUNK))
MAX_RECORD_SECONDS = 10
PRE_ROLL_CHUNKS = int(0.35 * (RATE / CHUNK))

# ------------------ TTS temp render ------------------
TTS_WAV_FILENAME = "tts_temp.wav"

# ------------------ Lafufu voice preset (offline) ------------------
LAFUFU_VOICE_PRESET = "lafufu"
LAFUFU_RATE = 210
LAFUFU_ESPEAK_PITCH = 75
LAFUFU_ESPEAK_VOICE = "en-us"
LAFUFU_PYTTSX3_KEYWORDS = ("child", "kid", "young", "girl", "boy", "zira", "hazel", "aria", "jenny")

# ------------------ Piper TTS (offline) ------------------
_IS_LINUX = platform.system().lower() == "linux"
_IS_ARM = platform.machine().lower() in ("aarch64", "arm64", "armv7l", "armv8l", "arm")
_PIPER_DEFAULT_QUALITY = "low" if _IS_LINUX and _IS_ARM else "high"
PIPER_MODEL_DEFAULT = f"models/en_US-libritts-{_PIPER_DEFAULT_QUALITY}.onnx"
PIPER_CONFIG_DEFAULT = f"models/en_US-libritts-{_PIPER_DEFAULT_QUALITY}.onnx.json"
PIPER_MODEL_CANDIDATES = (
    "models/en_US-libritts-low.onnx",
    "models/en_US-libritts-medium.onnx",
    "models/en_US-libritts-high.onnx",
)
PIPER_SPEAKER_DEFAULT = 0
PIPER_LENGTH_SCALE_DEFAULT = 0.9  # slightly faster speech by default
PIPER_NOISE_SCALE_DEFAULT = 0.667
PIPER_NOISE_W_DEFAULT = 0.8
PIPER_PITCH_CENTS_DEFAULT = 0.0  # disable pitch shifting by default for speed

# ------------------ Qwen model & system prompt ------------------
QWEN_MODEL = "qwen2.5:7b"
SYSTEM_PROMPT = (
    "You are Lafufu, a mischievous and playful humanoid creature. Reply in no more than 20 words. "
    "Always output an \"[emotion]\" where emotion is one of: happy, sad, angry, surprised, neutral, agree, disagree. "
    "This line must always appear at the very start, followed by the response. Never use emojis. "
    "Sample reply format:\n\n"
    "[happy]\nThat's great to hear!\n\n"
    "[neutral]\nI see. Thanks for sharing.\n\n"
    "[agree]\nI totally agree with you.\n\n"
    "[disagree]\nI don't think that's right."
)
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"

# ------------------ LIPSYNC Jaw (DXL tuning) ------------------
MOUTH_CLOSE_DXL = 1728  # same as DXL_JAW_CLOSE_POS
MOUTH_OPEN_DXL = 1534   # same as DXL_JAW_OPEN_POS
LIPSYNC_FPS = 20

LIPSYNC_GAMMA = 0.70
LIPSYNC_DEADZONE = 0.05
LIPSYNC_P_LOW = 0.10
LIPSYNC_P_HIGH = 0.95
LIPSYNC_ATTACK_S = 0.03
LIPSYNC_RELEASE_S = 0.08

# ------------------ Head pose / gesture (DXL tuning) ------------------
HEAD_IDLE_LR_DXL = 2063
HEAD_IDLE_UD_DXL = 3082

HEAD_LISTEN_LR_DXL = 2063
HEAD_LISTEN_UD_DXL = 2990

HEAD_TRANSITION_S = 0.25  # bigger = slower

# ------------------ Dynamixel calibration (POSITION <-> DEGREES) ------------------
# XM540
DXL_HEAD_LR_LEFT_POS = 2298
DXL_HEAD_LR_RIGHT_POS = 1828
HEAD_LR_RANGE_DEG = 41.31
HEAD_LR_HALF_DEG = HEAD_LR_RANGE_DEG * 0.5

DXL_HEAD_UD_UP_POS = 2885
DXL_HEAD_UD_DOWN_POS = 3278
HEAD_UD_RANGE_DEG = 34.54
HEAD_UD_HALF_DEG = HEAD_UD_RANGE_DEG * 0.5

# XC430
DXL_BROW_UP_POS = 2099
DXL_BROW_DOWN_POS = 2051
BROW_RANGE_DEG = 4.22
BROW_HALF_DEG = BROW_RANGE_DEG * 0.5

DXL_JAW_OPEN_POS = 1534
DXL_JAW_CLOSE_POS = 1728
JAW_RANGE_DEG = 17.05
SIM_JAW_CLOSE_DEG = 0.0
SIM_JAW_OPEN_DEG = JAW_RANGE_DEG

DXL_EYE_LEFT_POS = 2000
DXL_EYE_RIGHT_POS = 2077
EYE_RANGE_DEG = 6.75
EYE_HALF_DEG = EYE_RANGE_DEG * 0.5

# ------------------ Eyes (simulator degrees; realistic range) ------------------
EYE_MIN_DEG = -EYE_HALF_DEG
EYE_MAX_DEG = EYE_HALF_DEG
EYE_IDLE_DXL = 2039
EYE_TRANSITION_S = 0.12

# ------------------ Eyebrows (simulator degrees; realistic range) ------------------
BROW_MIN_DEG = -BROW_HALF_DEG
BROW_MAX_DEG = BROW_HALF_DEG
BROW_IDLE_DXL = 2075
BROW_TRANSITION_S = 0.10

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def _dxl_from_deg(deg: float, deg_min: float, deg_max: float, pos_min: float, pos_max: float) -> int:
    if deg_max == deg_min:
        return int(round(pos_min))
    t = (float(deg) - float(deg_min)) / (float(deg_max) - float(deg_min))
    t = max(0.0, min(1.0, t))
    pos = float(pos_min) + t * (float(pos_max) - float(pos_min))
    return int(round(pos))

def _deg_from_dxl(pos: float, pos_min: float, pos_max: float, deg_min: float, deg_max: float) -> float:
    if pos_max == pos_min:
        return float(deg_min)
    t = (float(pos) - float(pos_min)) / (float(pos_max) - float(pos_min))
    t = _clamp(t, 0.0, 1.0)
    return float(deg_min) + t * (float(deg_max) - float(deg_min))

def _pose_deg_to_dxl(jaw_deg: float, head_lr_deg: float, head_ud_deg: float, eye_deg: float, brow_deg: float):
    jaw_dxl = _dxl_from_deg(
        _clamp(jaw_deg, SIM_JAW_CLOSE_DEG, SIM_JAW_OPEN_DEG),
        SIM_JAW_CLOSE_DEG, SIM_JAW_OPEN_DEG,
        DXL_JAW_CLOSE_POS, DXL_JAW_OPEN_POS,
    )

    head_lr_dxl = _dxl_from_deg(
        _clamp(head_lr_deg, -HEAD_LR_HALF_DEG, HEAD_LR_HALF_DEG),
        -HEAD_LR_HALF_DEG, HEAD_LR_HALF_DEG,
        DXL_HEAD_LR_RIGHT_POS, DXL_HEAD_LR_LEFT_POS,
    )

    head_ud_dxl = _dxl_from_deg(
        _clamp(head_ud_deg, -HEAD_UD_HALF_DEG, HEAD_UD_HALF_DEG),
        -HEAD_UD_HALF_DEG, HEAD_UD_HALF_DEG,
        DXL_HEAD_UD_DOWN_POS, DXL_HEAD_UD_UP_POS,
    )

    eye_dxl = _dxl_from_deg(
        _clamp(eye_deg, EYE_MIN_DEG, EYE_MAX_DEG),
        EYE_MIN_DEG, EYE_MAX_DEG,
        DXL_EYE_LEFT_POS, DXL_EYE_RIGHT_POS,
    )

    brow_dxl = _dxl_from_deg(
        _clamp(brow_deg, BROW_MIN_DEG, BROW_MAX_DEG),
        BROW_MIN_DEG, BROW_MAX_DEG,
        DXL_BROW_DOWN_POS, DXL_BROW_UP_POS,
    )

    return jaw_dxl, head_lr_dxl, head_ud_dxl, eye_dxl, brow_dxl

def _pose_dxl_to_deg(jaw_dxl: int, head_lr_dxl: int, head_ud_dxl: int, eye_dxl: int, brow_dxl: int):
    jaw_deg = _deg_from_dxl(
        jaw_dxl, DXL_JAW_CLOSE_POS, DXL_JAW_OPEN_POS,
        0.0, JAW_RANGE_DEG
    )
    head_lr_deg = _deg_from_dxl(
        head_lr_dxl, DXL_HEAD_LR_RIGHT_POS, DXL_HEAD_LR_LEFT_POS,
        -HEAD_LR_HALF_DEG, HEAD_LR_HALF_DEG
    )
    head_ud_deg = _deg_from_dxl(
        head_ud_dxl, DXL_HEAD_UD_DOWN_POS, DXL_HEAD_UD_UP_POS,
        -HEAD_UD_HALF_DEG, HEAD_UD_HALF_DEG
    )
    eye_deg = _deg_from_dxl(
        eye_dxl, DXL_EYE_LEFT_POS, DXL_EYE_RIGHT_POS,
        EYE_MIN_DEG, EYE_MAX_DEG
    )
    brow_deg = _deg_from_dxl(
        brow_dxl, DXL_BROW_DOWN_POS, DXL_BROW_UP_POS,
        BROW_MIN_DEG, BROW_MAX_DEG
    )
    return jaw_deg, head_lr_deg, head_ud_deg, eye_deg, brow_deg

def _startup_pose_from_present_dict(present_dict: dict):
    jaw = present_dict.get("jaw", DXL_JAW_CLOSE_POS)
    head_lr = present_dict.get("head_lr", (DXL_HEAD_LR_LEFT_POS + DXL_HEAD_LR_RIGHT_POS) // 2)
    head_ud = present_dict.get("head_ud", (DXL_HEAD_UD_UP_POS + DXL_HEAD_UD_DOWN_POS) // 2)
    eye = present_dict.get("eye", (DXL_EYE_LEFT_POS + DXL_EYE_RIGHT_POS) // 2)
    brow = present_dict.get("brow", (DXL_BROW_DOWN_POS + DXL_BROW_UP_POS) // 2)
    return jaw, head_lr, head_ud, eye, brow

# ------------------ Expressions (neutral = none) ------------------
GESTURE_BASE_DURATION_S = 0.9

# agree/disagree
NOD_YES_AMP_UD_DXL = -68.26867400115808
NOD_NO_AMP_LR_DXL = 91.01912369886226
NOD_FREQ_HZ = 2.0
AGREE_BROW_RAISE_DXL = 10.23696682464455
DISAGREE_BROW_FURROW_DXL = -13.649289099526065

# happy
HAPPY_DURATION_S = 0.9
HAPPY_FREQ_HZ = 2.6
HAPPY_BOB_UD_AMP_DXL = -39.82339316734221
HAPPY_SWAY_LR_AMP_DXL = 34.13217138707335
HAPPY_EYE_BIAS_DXL = 15.92
HAPPY_EYE_JITTER_AMP_DXL = 9.09
HAPPY_EYE_JITTER_HZ = 4.0
HAPPY_BROW_RAISE_DXL = 20.4739336492891
HAPPY_BROW_BOUNCE_AMP_DXL = 6.824644549763033
HAPPY_BROW_BOUNCE_HZ = 3.0

# sad
SAD_DURATION_S = 1.2
SAD_FREQ_HZ = 1.2
SAD_DROOP_UD_DXL = -68.26867400115808
SAD_SWAY_LR_AMP_DXL = 13.652868554829338
SAD_EYE_BIAS_DXL = -20.48
SAD_EYE_SWAY_AMP_DXL = 4.55
SAD_EYE_SWAY_HZ = 1.2
SAD_BROW_RAISE_DXL = 10.23696682464455

# angry
ANGRY_DURATION_S = 0.75
ANGRY_FREQ_HZ = 3.6
ANGRY_SHAKE_LR_AMP_DXL = 113.77390462357782
ANGRY_CHIN_DOWN_UD_DXL = -34.13433700057904
ANGRY_BOB_UD_AMP_DXL = -13.653734800231614
ANGRY_EYE_BIAS_DXL = -9.09
ANGRY_EYE_SHAKE_AMP_DXL = 25.02
ANGRY_EYE_SHAKE_HZ = 5.0
ANGRY_BROW_FURROW_DXL = -24.0
ANGRY_BROW_SHAKE_AMP_DXL = 3.9810426540284354
ANGRY_BROW_SHAKE_HZ = 5.5

# surprised
SURPRISED_DURATION_S = 0.6
SURPRISED_POP_UD_DXL = 113.78112333526346
SURPRISED_TINY_LR_DXL = 17.066085693536674
SURPRISED_EYE_POP_DXL = 34.12
SURPRISED_BROW_POP_DXL = 24.0

# ------------------ Expression sustain (continuous) ------------------
EXPRESSION_SUSTAIN_FADE_S = 0.35
SUSTAIN_SCALE = 0.35
SUSTAIN_EYE_SCALE = 0.30
SUSTAIN_BROW_SCALE = 0.30
RETURN_TO_NEUTRAL_S = 0.35

# ------------------ Idle animation ------------------
HEAD_LR_RANGE_DXL = abs(DXL_HEAD_LR_LEFT_POS - DXL_HEAD_LR_RIGHT_POS)
HEAD_UD_RANGE_DXL = abs(DXL_HEAD_UD_UP_POS - DXL_HEAD_UD_DOWN_POS)
EYE_RANGE_DXL = abs(DXL_EYE_RIGHT_POS - DXL_EYE_LEFT_POS)
BROW_RANGE_DXL = abs(DXL_BROW_UP_POS - DXL_BROW_DOWN_POS)

IDLE_HZ = 20.0
IDLE_SEG_MIN_S = 2.0
IDLE_SEG_MAX_S = 5.0
IDLE_PAUSE_CHANCE = 0.30
IDLE_PAUSE_MIN_S = 1.0
IDLE_PAUSE_MAX_S = 3.5

IDLE_HEAD_LR_AMP_DXL = HEAD_LR_RANGE_DXL * 0.06
IDLE_HEAD_UD_AMP_DXL = HEAD_UD_RANGE_DXL * 0.05
IDLE_EYE_AMP_DXL = EYE_RANGE_DXL * 0.18
IDLE_BROW_AMP_DXL = BROW_RANGE_DXL * 0.16

IDLE_HEAD_FREQ_MIN = 0.08
IDLE_HEAD_FREQ_MAX = 0.22
IDLE_EYE_FREQ_MIN = 0.15
IDLE_EYE_FREQ_MAX = 0.45

# ------------------ Blender UDP (degrees values) ------------------
BLENDER_UDP_ENABLED = True
BLENDER_UDP_HOST = "127.0.0.1"
BLENDER_UDP_PORT = 5005
_udp_sock = None

def _udp_init():
    global _udp_sock
    if _udp_sock is None:
        _udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# ------------------ Live status line (persistent, IN-PLACE) ------------------
class StatusLine:
    def __init__(self):
        self._last_vals = None  # (jaw, hud, hlr, eye, brow) ints
        self._last_line = ""
        self._visible = False

    @staticmethod
    def _fmt(jaw: int, hud: int, hlr: int, eye: int, brow: int) -> str:
        return f"DXL J:{jaw:4d} HU:{hud:4d} HL:{hlr:4d} E:{eye:4d} B:{brow:4d}"

    def update_if_changed(self, jaw: int, head_ud: int, head_lr: int, eye: int, brow: int) -> None:
        vals = (int(jaw), int(head_ud), int(head_lr), int(eye), int(brow))
        if vals == self._last_vals:
            return

        line = self._fmt(*vals)
        if self._visible:
            sys.stdout.write("\r" + (" " * len(self._last_line)) + "\r")
        else:
            sys.stdout.write("\r")
        sys.stdout.write(line)
        sys.stdout.flush()

        self._last_vals = vals
        self._last_line = line
        self._visible = True

    def suspend_for_chat(self) -> None:
        if not self._visible:
            return
        sys.stdout.write("\r" + (" " * len(self._last_line)) + "\r")
        sys.stdout.flush()

    def resume_after_chat(self) -> None:
        if not self._visible:
            return
        sys.stdout.write("\r" + self._last_line)
        sys.stdout.flush()

_status = StatusLine()

def chat_print(s: str = "", end: str = "\n") -> None:
    _status.suspend_for_chat()
    sys.stdout.write(s + end)
    sys.stdout.flush()
    _status.resume_after_chat()

def chat_input(prompt: str) -> str:
    _status.suspend_for_chat()
    try:
        return input(prompt)
    finally:
        _status.resume_after_chat()

# ------------------ Dynamixel (REAL HW) ------------------
DXL_PROTOCOL = 2.0
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
LEN_GOAL_POSITION = 4
ADDR_PRESENT_POSITION = 132
TORQUE_ENABLE = 1
TORQUE_DISABLE = 0

# channel -> motor ID
DXL_IDS = {
    "head_lr": 1,  # XM540
    "head_ud": 2,  # XM540
    "brow": 3,     # XC430
    "jaw": 4,      # XC430
    "eye": 5,      # XC430
}

# safety clamps (DXL positions)
DXL_CLAMP = {
    "head_lr": (DXL_HEAD_LR_RIGHT_POS, DXL_HEAD_LR_LEFT_POS),  # 1828..2298
    "head_ud": (DXL_HEAD_UD_UP_POS, DXL_HEAD_UD_DOWN_POS),     # 2885..3278
    "brow":    (DXL_BROW_DOWN_POS, DXL_BROW_UP_POS),           # 2051..2099
    "jaw":     (DXL_JAW_OPEN_POS, DXL_JAW_CLOSE_POS),          # 1534..1728
    "eye":     (DXL_EYE_LEFT_POS, DXL_EYE_RIGHT_POS),          # 2000..2077
}

def _clamp_dxl(name: str, v: float) -> int:
    lo, hi = DXL_CLAMP[name]
    mn, mx = (lo, hi) if lo <= hi else (hi, lo)
    return max(mn, min(mx, int(round(v))))

def _clamp_dxl_f(name: str, v: float) -> float:
    lo, hi = DXL_CLAMP[name]
    mn, mx = (lo, hi) if lo <= hi else (hi, lo)
    return max(float(mn), min(float(mx), float(v)))

def _maybe_import_dxl():
    try:
        from dynamixel_sdk import (
            PortHandler,
            PacketHandler,
            GroupSyncWrite,
            DXL_LOBYTE, DXL_HIBYTE, DXL_LOWORD, DXL_HIWORD,
        )
        return PortHandler, PacketHandler, GroupSyncWrite, DXL_LOBYTE, DXL_HIBYTE, DXL_LOWORD, DXL_HIWORD
    except Exception:
        return None

class DynamixelBus:
    def __init__(self, port: str, baud: int, hz: float = 50.0, leave_torque: bool = False):
        imp = _maybe_import_dxl()
        if imp is None:
            raise RuntimeError("Missing dependency: dynamixel-sdk. Install with: pip install dynamixel-sdk")

        (self.PortHandler, self.PacketHandler, self.GroupSyncWrite,
         self.DXL_LOBYTE, self.DXL_HIBYTE, self.DXL_LOWORD, self.DXL_HIWORD) = imp

        self.port = port
        self.baud = int(baud)
        self.hz = float(hz)
        self.leave_torque = bool(leave_torque)

        self._ph = None
        self._pk = None
        self._sw = None
        self._last_send_t = 0.0
        self._last_vals = None

        self.connected_names = []
        self.present_start = {}

    def open(self, required_names=None):
        required_names = list(required_names or [])

        self._ph = self.PortHandler(self.port)
        self._pk = self.PacketHandler(DXL_PROTOCOL)

        if not self._ph.openPort():
            raise RuntimeError(f"Failed to open port: {self.port} (make sure Dynamixel Wizard is closed)")

        if not self._ph.setBaudRate(self.baud):
            raise RuntimeError(f"Failed to set baud rate {self.baud} on {self.port}")

        connected = []
        for name, dxl_id in DXL_IDS.items():
            try:
                model, comm, err = self._pk.ping(self._ph, dxl_id)
                if comm == 0 and err == 0:
                    connected.append(name)
            except Exception:
                pass

        self.connected_names = connected

        if not self.connected_names:
            raise RuntimeError("No Dynamixels responded on the bus (wrong COM/baud or wiring/power issue).")

        missing = [name for name in required_names if name not in self.connected_names]
        if missing:
            raise RuntimeError(f"Required motor(s) not found: {', '.join(missing)}")

        self._sw = self.GroupSyncWrite(self._ph, self._pk, ADDR_GOAL_POSITION, LEN_GOAL_POSITION)

        # Read current pose BEFORE torque enable
        present = self.read_present_pose(retries=4)
        self.present_start = dict(present)

        # Torque enable only on connected motors
        for name in self.connected_names:
            dxl_id = DXL_IDS[name]
            ok = False
            for _ in range(3):
                try:
                    _, comm, err = self._pk.write1ByteTxRx(self._ph, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
                    if comm == 0 and err == 0:
                        ok = True
                        break
                except Exception:
                    pass
                time.sleep(0.03)
            # if not ok:
            #     raise RuntimeError(f"Failed to enable torque on {name} (ID {dxl_id}).")

        # Immediately hold current pose to avoid snap
        self.send_pose_dict(self.present_start, force=True)
        time.sleep(0.1)
        self.send_pose_dict(self.present_start, force=True)

        return dict(self.present_start)

    def close(self):
        if self._pk and self._ph and not self.leave_torque:
            for name in self.connected_names:
                dxl_id = DXL_IDS[name]
                try:
                    self._pk.write1ByteTxRx(self._ph, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
                except Exception:
                    pass
        if self._ph:
            try:
                self._ph.closePort()
            except Exception:
                pass
        self._ph = None
        self._pk = None
        self._sw = None

    def _u32_le(self, v: int):
        v = int(v) & 0xFFFFFFFF
        return bytes([
            self.DXL_LOBYTE(self.DXL_LOWORD(v)),
            self.DXL_HIBYTE(self.DXL_LOWORD(v)),
            self.DXL_LOBYTE(self.DXL_HIWORD(v)),
            self.DXL_HIBYTE(self.DXL_HIWORD(v)),
        ])

    def read_present_position(self, name: str, retries: int = 3):
        if name not in self.connected_names:
            return None

        dxl_id = DXL_IDS[name]
        for _ in range(max(1, retries)):
            try:
                value, comm, err = self._pk.read4ByteTxRx(self._ph, dxl_id, ADDR_PRESENT_POSITION)
                if comm == 0 and err == 0:
                    return _clamp_dxl(name, int(value))
            except Exception:
                pass
            time.sleep(0.02)
        return None

    def read_present_pose(self, retries: int = 3):
        pose = {}
        names = self.connected_names if self.connected_names else list(DXL_IDS.keys())
        for name in names:
            value = self.read_present_position(name, retries=retries) if self.connected_names else None
            if value is None:
                lo, hi = DXL_CLAMP[name]
                mn, mx = (lo, hi) if lo <= hi else (hi, lo)
                value = int(round((mn + mx) * 0.5))
            pose[name] = _clamp_dxl(name, value)
        return pose

    def send_pose_dict(self, pose_by_name: dict, force: bool = False):
        if not self._sw:
            return

        now = time.time()
        dt_min = 1.0 / max(1e-6, self.hz)
        if not force and (now - self._last_send_t) < dt_min:
            return
        self._last_send_t = now

        vals = {}
        for name in self.connected_names:
            if name in pose_by_name:
                vals[name] = _clamp_dxl(name, int(pose_by_name[name]))

        if not vals:
            return

        tup = tuple((name, vals[name]) for name in sorted(vals.keys()))
        if (not force) and tup == self._last_vals:
            return
        self._last_vals = tup

        self._sw.clearParam()
        try:
            for name, v in vals.items():
                self._sw.addParam(DXL_IDS[name], self._u32_le(v))
            self._sw.txPacket()
        except Exception:
            # keep running even if a packet fails
            pass

    def send_pose(self, jaw: int, head_ud: int, head_lr: int, eye: int, brow: int, force: bool = False):
        pose_by_name = {
            "jaw": jaw,
            "head_ud": head_ud,
            "head_lr": head_lr,
            "eye": eye,
            "brow": brow,
        }
        self.send_pose_dict(pose_by_name, force=force)

def _default_port_candidates():
    if platform.system().lower() == "windows":
        return [f"COM{i}" for i in range(1, 51)]
    return ["/dev/ttyUSB0", "/dev/ttyUSB1", "/dev/ttyACM0", "/dev/ttyACM1"]

def _auto_find_u2d2_port_and_baud(baud_candidates):
    imp = _maybe_import_dxl()
    if imp is None:
        return None, None
    PortHandler, PacketHandler, *_ = imp
    pk = PacketHandler(DXL_PROTOCOL)

    for port in _default_port_candidates():
        ph = PortHandler(port)
        if not ph.openPort():
            continue
        try:
            for baud in baud_candidates:
                try:
                    if not ph.setBaudRate(int(baud)):
                        continue
                    # ping any expected ID
                    any_ok = False
                    for _, dxl_id in DXL_IDS.items():
                        model, comm, err = pk.ping(ph, dxl_id)
                        if comm == 0 and err == 0:
                            any_ok = True
                            break
                    if any_ok:
                        return port, int(baud)
                except Exception:
                    continue
        finally:
            try:
                ph.closePort()
            except Exception:
                pass
    return None, None

# ------------------ Output router (Blender + optional real motors) ------------------
_dxl_bus = None
_send_blender = True

def _send_pose_outputs(jaw_dxl: float, head_lr_dxl: float, head_ud_dxl: float, eye_dxl: float, brow_dxl: float):
    # clamp DXL positions
    jaw_dxl = _clamp_dxl("jaw", jaw_dxl)
    head_lr_dxl = _clamp_dxl("head_lr", head_lr_dxl)
    head_ud_dxl = _clamp_dxl("head_ud", head_ud_dxl)
    eye_dxl = _clamp_dxl("eye", eye_dxl)
    brow_dxl = _clamp_dxl("brow", brow_dxl)

    # Blender gets DEGREES (simulator)
    if _send_blender and BLENDER_UDP_ENABLED:
        try:
            _udp_init()
            jaw_deg, head_lr_deg, head_ud_deg, eye_deg, brow_deg = _pose_dxl_to_deg(
                jaw_dxl, head_lr_dxl, head_ud_dxl, eye_dxl, brow_dxl
            )
            payload = {
                "jaw": float(jaw_deg),
                "head_lr": float(head_lr_deg),
                "head_ud": float(head_ud_deg),
                "eye": float(eye_deg),
                "brow": float(brow_deg),
            }
            _udp_sock.sendto(
                json.dumps(payload, separators=(",", ":")).encode("utf-8"),
                (BLENDER_UDP_HOST, BLENDER_UDP_PORT),
            )
        except Exception:
            pass

    # Real motors get DXL values
    if _dxl_bus is not None:
        _dxl_bus.send_pose(
            jaw=int(jaw_dxl),
            head_ud=int(head_ud_dxl),
            head_lr=int(head_lr_dxl),
            eye=int(eye_dxl),
            brow=int(brow_dxl),
        )

    # status line expects: jaw, head_ud, head_lr, eye, brow
    return int(jaw_dxl), int(head_ud_dxl), int(head_lr_dxl), int(eye_dxl), int(brow_dxl)

# ------------------ Head state (smooth) ------------------
_head_lr_cur = HEAD_IDLE_LR_DXL
_head_ud_cur = HEAD_IDLE_UD_DXL
_head_lr_tgt = HEAD_IDLE_LR_DXL
_head_ud_tgt = HEAD_IDLE_UD_DXL

def _head_set_target(lr_dxl: float, ud_dxl: float):
    global _head_lr_tgt, _head_ud_tgt
    _head_lr_tgt = _clamp_dxl_f("head_lr", float(lr_dxl))
    _head_ud_tgt = _clamp_dxl_f("head_ud", float(ud_dxl))

def _head_step(dt: float):
    global _head_lr_cur, _head_ud_cur
    tau = max(1e-6, float(HEAD_TRANSITION_S))
    a = 1.0 - math.exp(-float(dt) / tau)
    _head_lr_cur += (_head_lr_tgt - _head_lr_cur) * a
    _head_ud_cur += (_head_ud_tgt - _head_ud_cur) * a
    _head_lr_cur = _clamp_dxl_f("head_lr", _head_lr_cur)
    _head_ud_cur = _clamp_dxl_f("head_ud", _head_ud_cur)

# ------------------ Eye state (smooth) ------------------
_eye_cur = EYE_IDLE_DXL
_eye_tgt = EYE_IDLE_DXL

def _eye_set_target(eye_dxl: float):
    global _eye_tgt
    _eye_tgt = _clamp_dxl_f("eye", float(eye_dxl))

def _eye_step(dt: float):
    global _eye_cur
    tau = max(1e-6, float(EYE_TRANSITION_S))
    a = 1.0 - math.exp(-float(dt) / tau)
    _eye_cur += (_eye_tgt - _eye_cur) * a
    _eye_cur = _clamp_dxl_f("eye", _eye_cur)

# ------------------ Brow state (smooth) ------------------
_brow_cur = BROW_IDLE_DXL
_brow_tgt = BROW_IDLE_DXL

def _brow_set_target(brow_dxl: float):
    global _brow_tgt
    _brow_tgt = _clamp_dxl_f("brow", float(brow_dxl))

def _brow_step(dt: float):
    global _brow_cur
    tau = max(1e-6, float(BROW_TRANSITION_S))
    a = 1.0 - math.exp(-float(dt) / tau)
    _brow_cur += (_brow_tgt - _brow_cur) * a
    _brow_cur = _clamp_dxl_f("brow", _brow_cur)

# ------------------ Idle animation (background) ------------------
_idle_thread = None
_idle_allowed = threading.Event()
_idle_allowed.set()
_idle_show_status = True

def _idle_suspend():
    _idle_allowed.clear()

def _idle_resume():
    _idle_allowed.set()

def _idle_loop():
    rng = random.Random()
    dt = 1.0 / max(1.0, float(IDLE_HZ))

    seg_start = time.time()
    seg_end = 0.0
    mode = "pause"
    lr_amp = ud_amp = eye_amp = brow_amp = 0.0
    lr_freq = ud_freq = eye_freq = brow_freq = 0.0
    lr_phase = ud_phase = eye_phase = brow_phase = 0.0

    while True:
        if not _idle_allowed.is_set():
            seg_end = 0.0
            time.sleep(0.05)
            continue

        now = time.time()
        if now >= seg_end:
            seg_start = now
            if rng.random() < IDLE_PAUSE_CHANCE:
                mode = "pause"
                seg_end = now + rng.uniform(IDLE_PAUSE_MIN_S, IDLE_PAUSE_MAX_S)
            else:
                mode = "move"
                seg_end = now + rng.uniform(IDLE_SEG_MIN_S, IDLE_SEG_MAX_S)
                lr_amp = IDLE_HEAD_LR_AMP_DXL * rng.uniform(0.5, 1.0)
                ud_amp = IDLE_HEAD_UD_AMP_DXL * rng.uniform(0.4, 0.9)
                eye_amp = IDLE_EYE_AMP_DXL * rng.uniform(0.4, 1.0)
                brow_amp = IDLE_BROW_AMP_DXL * rng.uniform(0.4, 1.0)
                lr_freq = rng.uniform(IDLE_HEAD_FREQ_MIN, IDLE_HEAD_FREQ_MAX)
                ud_freq = rng.uniform(IDLE_HEAD_FREQ_MIN, IDLE_HEAD_FREQ_MAX)
                eye_freq = rng.uniform(IDLE_EYE_FREQ_MIN, IDLE_EYE_FREQ_MAX)
                brow_freq = rng.uniform(IDLE_EYE_FREQ_MIN, IDLE_EYE_FREQ_MAX)
                lr_phase = rng.uniform(0.0, math.tau)
                ud_phase = rng.uniform(0.0, math.tau)
                eye_phase = rng.uniform(0.0, math.tau)
                brow_phase = rng.uniform(0.0, math.tau)

        if mode == "pause":
            _head_set_target(HEAD_IDLE_LR_DXL, HEAD_IDLE_UD_DXL)
            _eye_set_target(EYE_IDLE_DXL)
            _brow_set_target(BROW_IDLE_DXL)
        else:
            t = now - seg_start
            lr_off = math.sin((math.tau * lr_freq * t) + lr_phase) * lr_amp
            ud_off = math.sin((math.tau * ud_freq * t) + ud_phase) * ud_amp
            eye_off = math.sin((math.tau * eye_freq * t) + eye_phase) * eye_amp
            brow_off = math.sin((math.tau * brow_freq * t) + brow_phase) * brow_amp

            _head_set_target(HEAD_IDLE_LR_DXL + lr_off, HEAD_IDLE_UD_DXL + ud_off)
            _eye_set_target(EYE_IDLE_DXL + eye_off)
            _brow_set_target(BROW_IDLE_DXL + brow_off)

        _head_step(dt)
        _eye_step(dt)
        _brow_step(dt)

        vals = _send_pose_outputs(MOUTH_CLOSE_DXL, _head_lr_cur, _head_ud_cur, _eye_cur, _brow_cur)
        if _idle_show_status:
            _status.update_if_changed(*vals)

        time.sleep(dt)

def _start_idle_animation():
    global _idle_thread
    if _idle_thread is not None:
        return
    _idle_thread = threading.Thread(target=_idle_loop, daemon=True)
    _idle_thread.start()

# ------------------ Helpers ------------------
def audio_rms(data: bytes) -> float:
    count = len(data) // 2
    shorts = memoryview(data).cast("h")
    s = 0.0
    for sample in shorts:
        s += sample * sample
    return math.sqrt(s / max(count, 1))

def rms_from_bytes(data: bytes, sample_width: int) -> float:
    if not data:
        return 0.0
    if sample_width == 2:
        count = len(data) // 2
        samples = memoryview(data).cast("h")
        s = 0.0
        for v in samples:
            s += v * v
        return math.sqrt(s / max(count, 1))
    if sample_width == 1:
        s = 0.0
        for b in data:
            v = b - 128
            s += v * v
        return math.sqrt(s / max(len(data), 1))
    if sample_width == 4:
        count = len(data) // 4
        samples = memoryview(data).cast("i")
        s = 0.0
        for v in samples:
            s += v * v
        return math.sqrt(s / max(count, 1))
    s = 0.0
    for b in data:
        s += b * b
    return math.sqrt(s / max(len(data), 1))

def percentile_sorted(vals_sorted, p: float) -> float:
    if not vals_sorted:
        return 0.0
    p = max(0.0, min(1.0, p))
    idx = int(round(p * (len(vals_sorted) - 1)))
    return vals_sorted[idx]

def pa_format_from_sample_width(sample_width: int):
    if sample_width == 1:
        return pyaudio.paUInt8
    if sample_width == 2:
        return pyaudio.paInt16
    if sample_width == 4:
        return pyaudio.paInt32
    return None

# ------------------ Recording / Whisper ------------------
def record_until_silence() -> Path:
    _idle_suspend()
    chat_print("\n🎙️ Listening… (auto-stop on silence)")
    _head_set_target(HEAD_LISTEN_LR_DXL, HEAD_LISTEN_UD_DXL)
    _eye_set_target(EYE_IDLE_DXL)
    _brow_set_target(BROW_IDLE_DXL)

    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )

    frames, silent_chunks = [], 0
    pre_roll = deque(maxlen=max(PRE_ROLL_CHUNKS, 1))
    started = False
    total_chunks = 0
    max_chunks = int(RATE / CHUNK * MAX_RECORD_SECONDS)
    dt = CHUNK / RATE

    try:
        try:
            while True:
                data = stream.read(CHUNK, exception_on_overflow=False)
                rms = audio_rms(data)
                total_chunks += 1

                _head_step(dt)
                _eye_step(dt)
                _brow_step(dt)

                vals = _send_pose_outputs(MOUTH_CLOSE_DXL, _head_lr_cur, _head_ud_cur, _eye_cur, _brow_cur)
                _status.update_if_changed(*vals)

                if not started:
                    pre_roll.append(data)
                    if rms >= SILENCE_THRESHOLD:
                        started = True
                        frames.extend(pre_roll)
                    if total_chunks > max_chunks:
                        break
                    continue

                frames.append(data)
                silent_chunks = silent_chunks + 1 if rms < SILENCE_THRESHOLD else 0
                if silent_chunks > SILENCE_CHUNKS:
                    break
                if total_chunks > max_chunks:
                    break
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

        with wave.open(OUTPUT_FILENAME, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b"".join(frames))

        _head_set_target(HEAD_IDLE_LR_DXL, HEAD_IDLE_UD_DXL)
        _eye_set_target(EYE_IDLE_DXL)
        _brow_set_target(BROW_IDLE_DXL)
        return Path(OUTPUT_FILENAME)
    finally:
        _idle_resume()

def transcribe(audio_path: Path, model) -> str:
    chat_print("Transcribing...")
    result = model.transcribe(str(audio_path), fp16=False)
    return result["text"].strip()

# ------------------ Ollama ------------------
def _ollama_chat(payload: dict, timeout: int = 300):
    req = Request(
        OLLAMA_CHAT_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/x-ndjson"},
        method="POST",
    )
    return urlopen(req, timeout=timeout)

def warmup_qwen(model: str, system_prompt: str, keep_alive=None) -> None:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt or "You are a helpful assistant."},
            {"role": "user", "content": "warmup"},
        ],
        "stream": False,
        "options": {"num_predict": 1},
    }
    if keep_alive is not None:
        payload["keep_alive"] = keep_alive
    try:
        with _ollama_chat(payload, timeout=300) as resp:
            resp.read()
    except Exception:
        pass

def ask_qwen_stream_collect(prompt: str, model: str, system_prompt: str = None, keep_alive=None) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {"model": model, "messages": messages, "stream": True}
    if keep_alive is not None:
        payload["keep_alive"] = keep_alive

    full = []
    try:
        with _ollama_chat(payload, timeout=300) as resp:
            last_t = time.time()
            for raw_line in resp:
                now = time.time()
                dt = now - last_t
                last_t = now
                dt = max(0.0, min(dt, 0.1))

                _head_step(dt)
                _eye_set_target(EYE_IDLE_DXL)
                _eye_step(dt)
                _brow_set_target(BROW_IDLE_DXL)
                _brow_step(dt)

                vals = _send_pose_outputs(MOUTH_CLOSE_DXL, _head_lr_cur, _head_ud_cur, _eye_cur, _brow_cur)
                _status.update_if_changed(*vals)

                if not raw_line:
                    continue
                line = raw_line.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue

                obj = json.loads(line)
                if "error" in obj:
                    return ""

                chunk = (obj.get("message") or {}).get("content") or ""
                if chunk:
                    full.append(chunk)

                if obj.get("done", False):
                    break

        return "".join(full).strip()
    except URLError:
        return ""
    except Exception:
        return ""

def parse_emotion_and_spoken_text(reply: str):
    if not reply:
        return None, ""
    lines = reply.splitlines()
    first = lines[0].strip()
    if first.startswith("[") and "]" in first:
        close = first.find("]")
        tag = first[1:close].strip().lower()
        rest_same_line = first[close + 1:].strip()
        remaining = []
        if rest_same_line:
            remaining.append(rest_same_line)
        if len(lines) > 1:
            remaining.extend(lines[1:])
        spoken = "\n".join(remaining).strip()
        return tag, spoken
    return None, reply.strip()

def _print_text_cups(text: str) -> None:
    cmd = ["lp"]
    try:
        result = subprocess.run(cmd, input=text.encode("utf-8"), check=False)
        if result.returncode == 0:
            return
    except FileNotFoundError:
        pass
    cmd = ["lpr"]
    try:
        subprocess.run(cmd, input=text.encode("utf-8"), check=False)
    except FileNotFoundError:
        pass

def print_lafufu_text(text: str) -> None:
    clean = (text or "").strip()
    if not clean:
        return
    _print_text_cups(clean)

# ------------------ Expressions ------------------
def emotion_to_expression(emotion: str):
    e = (emotion or "").strip().lower()
    if not e or e == "neutral":
        return None
    if e in ("agree", "agreeing"):
        return "nod_yes"
    if e in ("disagree", "disagreeing"):
        return "nod_no"
    if e == "happy":
        return "happy"
    if e == "sad":
        return "sad"
    if e == "angry":
        return "angry"
    if e == "surprised":
        return "surprised"
    return None

def expression_duration(kind: str) -> float:
    if kind in ("nod_yes", "nod_no"):
        return GESTURE_BASE_DURATION_S
    if kind == "happy":
        return HAPPY_DURATION_S
    if kind == "sad":
        return SAD_DURATION_S
    if kind == "angry":
        return ANGRY_DURATION_S
    if kind == "surprised":
        return SURPRISED_DURATION_S
    return 0.0

def expression_offsets(kind: str, t: float):
    if not kind:
        return 0.0, 0.0, 0.0, 0.0

    dur = max(1e-6, expression_duration(kind))
    if t < 0.0 or t > dur:
        return 0.0, 0.0, 0.0, 0.0

    x = t / dur
    env = math.sin(math.pi * x)

    if kind == "nod_yes":
        osc = math.sin(2.0 * math.pi * NOD_FREQ_HZ * t) * env
        brow = env * AGREE_BROW_RAISE_DXL
        return 0.0, osc * NOD_YES_AMP_UD_DXL, 0.0, brow

    if kind == "nod_no":
        osc = math.sin(2.0 * math.pi * NOD_FREQ_HZ * t) * env
        brow = env * DISAGREE_BROW_FURROW_DXL
        return osc * NOD_NO_AMP_LR_DXL, 0.0, 0.0, brow

    if kind == "happy":
        ud = math.sin(2.0 * math.pi * HAPPY_FREQ_HZ * t) * env * HAPPY_BOB_UD_AMP_DXL
        lr = math.sin(2.0 * math.pi * (HAPPY_FREQ_HZ * 0.8) * t + (math.pi * 0.5)) * env * HAPPY_SWAY_LR_AMP_DXL
        eye = (env * HAPPY_EYE_BIAS_DXL) + (math.sin(2.0 * math.pi * HAPPY_EYE_JITTER_HZ * t) * env * HAPPY_EYE_JITTER_AMP_DXL)
        brow = (env * HAPPY_BROW_RAISE_DXL) + (math.sin(2.0 * math.pi * HAPPY_BROW_BOUNCE_HZ * t) * env * HAPPY_BROW_BOUNCE_AMP_DXL)
        return lr, ud, eye, brow

    if kind == "sad":
        ud = env * SAD_DROOP_UD_DXL
        lr = math.sin(2.0 * math.pi * SAD_FREQ_HZ * t) * env * SAD_SWAY_LR_AMP_DXL
        eye = (env * SAD_EYE_BIAS_DXL) + (math.sin(2.0 * math.pi * SAD_EYE_SWAY_HZ * t) * env * SAD_EYE_SWAY_AMP_DXL)
        brow = env * SAD_BROW_RAISE_DXL
        return lr, ud, eye, brow

    if kind == "angry":
        lr = math.sin(2.0 * math.pi * ANGRY_FREQ_HZ * t) * env * ANGRY_SHAKE_LR_AMP_DXL
        ud = (env * ANGRY_CHIN_DOWN_UD_DXL) + (
            math.sin(2.0 * math.pi * (ANGRY_FREQ_HZ * 0.5) * t) * env * ANGRY_BOB_UD_AMP_DXL
        )
        eye = (env * ANGRY_EYE_BIAS_DXL) + (math.sin(2.0 * math.pi * ANGRY_EYE_SHAKE_HZ * t) * env * ANGRY_EYE_SHAKE_AMP_DXL)
        brow = (env * ANGRY_BROW_FURROW_DXL) + (math.sin(2.0 * math.pi * ANGRY_BROW_SHAKE_HZ * t) * env * ANGRY_BROW_SHAKE_AMP_DXL)
        return lr, ud, eye, brow

    if kind == "surprised":
        rise = 0.18 * dur
        hold = 0.10 * dur
        fall = max(1e-6, dur - rise - hold)
        if t < rise:
            a = t / max(1e-6, rise)
        elif t < rise + hold:
            a = 1.0
        else:
            a = 1.0 - ((t - (rise + hold)) / fall)
        a = max(0.0, min(1.0, a))
        ud = a * SURPRISED_POP_UD_DXL
        lr = env * SURPRISED_TINY_LR_DXL
        eye = a * SURPRISED_EYE_POP_DXL
        brow = a * SURPRISED_BROW_POP_DXL
        return lr, ud, eye, brow

    return 0.0, 0.0, 0.0, 0.0

def expression_sustain_offsets(kind: str, t: float):
    if not kind:
        return 0.0, 0.0, 0.0, 0.0

    fade = 1.0 - math.exp(-t / max(1e-6, EXPRESSION_SUSTAIN_FADE_S))

    if kind == "nod_yes":
        freq = NOD_FREQ_HZ * 0.6
        osc = math.sin(2.0 * math.pi * freq * t)
        brow = math.sin(2.0 * math.pi * (freq * 0.7) * t) * AGREE_BROW_RAISE_DXL * SUSTAIN_BROW_SCALE
        return 0.0, osc * NOD_YES_AMP_UD_DXL * SUSTAIN_SCALE * fade, 0.0, brow * fade

    if kind == "nod_no":
        freq = NOD_FREQ_HZ * 0.6
        osc = math.sin(2.0 * math.pi * freq * t)
        brow = math.sin(2.0 * math.pi * (freq * 0.7) * t) * DISAGREE_BROW_FURROW_DXL * SUSTAIN_BROW_SCALE
        return osc * NOD_NO_AMP_LR_DXL * SUSTAIN_SCALE * fade, 0.0, 0.0, brow * fade

    if kind == "happy":
        freq = HAPPY_FREQ_HZ * 0.7
        ud = math.sin(2.0 * math.pi * freq * t) * HAPPY_BOB_UD_AMP_DXL * SUSTAIN_SCALE
        lr = math.sin(2.0 * math.pi * (freq * 0.8) * t + (math.pi * 0.5)) * HAPPY_SWAY_LR_AMP_DXL * SUSTAIN_SCALE
        eye = (HAPPY_EYE_BIAS_DXL * SUSTAIN_EYE_SCALE) + (
            math.sin(2.0 * math.pi * (HAPPY_EYE_JITTER_HZ * 0.7) * t) * HAPPY_EYE_JITTER_AMP_DXL * SUSTAIN_EYE_SCALE
        )
        brow = (HAPPY_BROW_RAISE_DXL * SUSTAIN_BROW_SCALE) + (
            math.sin(2.0 * math.pi * (HAPPY_BROW_BOUNCE_HZ * 0.7) * t) * HAPPY_BROW_BOUNCE_AMP_DXL * SUSTAIN_BROW_SCALE
        )
        return lr * fade, ud * fade, eye * fade, brow * fade

    if kind == "sad":
        freq = SAD_FREQ_HZ * 0.6
        ud = (SAD_DROOP_UD_DXL * SUSTAIN_SCALE) + (
            math.sin(2.0 * math.pi * freq * t) * abs(SAD_DROOP_UD_DXL) * 0.12
        )
        lr = math.sin(2.0 * math.pi * freq * t) * SAD_SWAY_LR_AMP_DXL * SUSTAIN_SCALE
        eye = (SAD_EYE_BIAS_DXL * SUSTAIN_EYE_SCALE) + (
            math.sin(2.0 * math.pi * (SAD_EYE_SWAY_HZ * 0.7) * t) * SAD_EYE_SWAY_AMP_DXL * SUSTAIN_EYE_SCALE
        )
        brow = (SAD_BROW_RAISE_DXL * SUSTAIN_BROW_SCALE) + (
            math.sin(2.0 * math.pi * (freq * 0.7) * t) * SAD_BROW_RAISE_DXL * 0.12
        )
        return lr * fade, ud * fade, eye * fade, brow * fade

    if kind == "angry":
        freq = ANGRY_FREQ_HZ * 0.7
        lr = math.sin(2.0 * math.pi * freq * t) * ANGRY_SHAKE_LR_AMP_DXL * SUSTAIN_SCALE
        ud = (ANGRY_CHIN_DOWN_UD_DXL * 0.25) + (
            math.sin(2.0 * math.pi * (freq * 0.5) * t) * ANGRY_BOB_UD_AMP_DXL * 0.45
        )
        eye = (ANGRY_EYE_BIAS_DXL * SUSTAIN_EYE_SCALE) + (
            math.sin(2.0 * math.pi * (ANGRY_EYE_SHAKE_HZ * 0.7) * t) * ANGRY_EYE_SHAKE_AMP_DXL * SUSTAIN_EYE_SCALE
        )
        brow = (ANGRY_BROW_FURROW_DXL * 0.4) + (
            math.sin(2.0 * math.pi * (ANGRY_BROW_SHAKE_HZ * 0.7) * t) * ANGRY_BROW_SHAKE_AMP_DXL * SUSTAIN_BROW_SCALE
        )
        return lr * fade, ud * fade, eye * fade, brow * fade

    if kind == "surprised":
        freq = 1.3
        ud = math.sin(2.0 * math.pi * freq * t) * SURPRISED_POP_UD_DXL * 0.12
        lr = math.sin(2.0 * math.pi * (freq * 0.8) * t) * SURPRISED_TINY_LR_DXL * SUSTAIN_SCALE
        eye = (SURPRISED_EYE_POP_DXL * 0.25) + (
            math.sin(2.0 * math.pi * (freq * 1.6) * t) * SURPRISED_EYE_POP_DXL * 0.08
        )
        brow = (SURPRISED_BROW_POP_DXL * 0.35) + (
            math.sin(2.0 * math.pi * (freq * 1.4) * t) * SURPRISED_BROW_POP_DXL * 0.12
        )
        return lr * fade, ud * fade, eye * fade, brow * fade

    return 0.0, 0.0, 0.0, 0.0

def expression_offsets_continuous(kind: str, t: float):
    if not kind:
        return 0.0, 0.0, 0.0, 0.0

    dur = float(expression_duration(kind))
    if dur > 0.0 and t <= dur:
        return expression_offsets(kind, t)
    return expression_sustain_offsets(kind, max(0.0, t - dur))

def _smooth_return_to_neutral(start_jaw_dxl: float, duration_s: float = RETURN_TO_NEUTRAL_S, fps: float = 60.0):
    duration_s = max(0.0, float(duration_s))
    if duration_s <= 0.0:
        vals = _send_pose_outputs(MOUTH_CLOSE_DXL, _head_lr_cur, _head_ud_cur, _eye_cur, _brow_cur)
        _status.update_if_changed(*vals)
        return

    _head_set_target(HEAD_IDLE_LR_DXL, HEAD_IDLE_UD_DXL)
    _eye_set_target(EYE_IDLE_DXL)
    _brow_set_target(BROW_IDLE_DXL)

    steps = max(1, int(duration_s * fps))
    dt = duration_s / steps
    start = float(start_jaw_dxl)
    end = float(MOUTH_CLOSE_DXL)

    for i in range(steps):
        t = float(i + 1) / float(steps)
        jaw_dxl = start + (end - start) * t

        _head_step(dt)
        _eye_step(dt)
        _brow_step(dt)

        vals = _send_pose_outputs(jaw_dxl, _head_lr_cur, _head_ud_cur, _eye_cur, _brow_cur)
        _status.update_if_changed(*vals)

        time.sleep(dt)

def perform_expression(kind: str):
    if not kind:
        return
    dur = expression_duration(kind)
    if dur <= 0.0:
        return

    _idle_suspend()

    _head_set_target(HEAD_IDLE_LR_DXL, HEAD_IDLE_UD_DXL)
    _eye_set_target(EYE_IDLE_DXL)
    _brow_set_target(BROW_IDLE_DXL)

    fps = 60.0
    dt = 1.0 / fps
    t0 = time.time()
    try:
        while True:
            t = time.time() - t0
            if t > dur:
                break

            _head_step(dt)
            lr_off, ud_off, eye_off, brow_off = expression_offsets(kind, t)

            head_lr = _head_lr_cur + lr_off
            head_ud = _head_ud_cur + ud_off

            _eye_set_target(EYE_IDLE_DXL + eye_off)
            _eye_step(dt)

            _brow_set_target(BROW_IDLE_DXL + brow_off)
            _brow_step(dt)

            vals = _send_pose_outputs(MOUTH_CLOSE_DXL, head_lr, head_ud, _eye_cur, _brow_cur)
            _status.update_if_changed(*vals)

            time.sleep(dt)

        _eye_set_target(EYE_IDLE_DXL)
        _brow_set_target(BROW_IDLE_DXL)
        vals = _send_pose_outputs(MOUTH_CLOSE_DXL, _head_lr_cur, _head_ud_cur, _eye_cur, _brow_cur)
        _status.update_if_changed(*vals)
    finally:
        _idle_resume()

# ------------------ TTS + lipsync playback ------------------
def init_tts(rate: int, volume: float, voice: str = None, engine: str = "auto"):
    if engine == "auto":
        engine = "espeak" if platform.system().lower() == "linux" else "pyttsx3"
    style = None
    if voice and voice.strip().lower() == LAFUFU_VOICE_PRESET:
        style = LAFUFU_VOICE_PRESET
        voice = None
    return {"rate": rate, "volume": volume, "voice": voice, "engine": engine, "style": style}

def _pick_pyttsx3_voice(engine, keywords) -> str | None:
    try:
        voices = engine.getProperty("voices") or []
    except Exception:
        return None
    for v in voices:
        name = (getattr(v, "name", "") or "").lower()
        vid = (getattr(v, "id", "") or "").lower()
        if any(k in name or k in vid for k in keywords):
            return getattr(v, "id", None)
    return getattr(voices[0], "id", None) if voices else None

def _resolve_tts_params(engine_name: str, tts_config: dict, pyttsx3_engine=None):
    rate = int(tts_config.get("rate", 170))
    volume = float(tts_config.get("volume", 1.0))
    voice = tts_config.get("voice")
    pitch = None

    if tts_config.get("style") == LAFUFU_VOICE_PRESET:
        rate = max(rate, LAFUFU_RATE)
        if engine_name == "espeak":
            pitch = LAFUFU_ESPEAK_PITCH
            if not voice:
                voice = LAFUFU_ESPEAK_VOICE
        elif pyttsx3_engine is not None and not voice:
            voice = _pick_pyttsx3_voice(pyttsx3_engine, LAFUFU_PYTTSX3_KEYWORDS)

    return rate, volume, voice, pitch

def _resolve_piper_paths(tts_config: dict):
    base = Path(__file__).parent
    explicit_model = (tts_config.get("piper_model") or "").strip()
    explicit_config = (tts_config.get("piper_config") or "").strip()

    if explicit_model:
        model = Path(explicit_model)
        if not model.is_absolute():
            model = base / model
    else:
        model = None
        for rel in PIPER_MODEL_CANDIDATES:
            cand = base / rel
            if cand.exists():
                model = cand
                break
        if model is None:
            model = Path(PIPER_MODEL_DEFAULT)
            if not model.is_absolute():
                model = base / model

    if explicit_config:
        config = Path(explicit_config)
        if not config.is_absolute():
            config = base / config
    else:
        config = model.with_suffix(model.suffix + ".json")
        if not config.exists():
            config = Path(PIPER_CONFIG_DEFAULT)
            if not config.is_absolute():
                config = base / config

    return model, config

def _apply_pitch_shift(wav_path: Path, cents: float) -> None:
    if not cents:
        return
    try:
        import numpy as np
        import librosa
        import soundfile as sf
    except Exception:
        return

    try:
        data, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
        if data.size == 0:
            return
        n_steps = float(cents) / 100.0
        shifted = []
        for ch in range(data.shape[1]):
            shifted_ch = librosa.effects.pitch_shift(data[:, ch], sr=sr, n_steps=n_steps)
            shifted.append(shifted_ch)
        max_len = max(len(ch) for ch in shifted)
        out = np.zeros((max_len, len(shifted)), dtype=np.float32)
        for i, ch in enumerate(shifted):
            out[: len(ch), i] = ch
        # preserve original length for lipsync timing
        out = out[: data.shape[0], :]
        sf.write(str(wav_path), out, sr, subtype="PCM_16")
    except Exception:
        return

def _piper_tts_to_wav(tts_config: dict, text: str, out_wav: Path) -> bool:
    model, config = _resolve_piper_paths(tts_config)
    cmd = ["piper", "--model", str(model), "--output_file", str(out_wav)]
    if config.exists():
        cmd.extend(["--config", str(config)])
    speaker = tts_config.get("piper_speaker")
    if speaker is not None:
        cmd.extend(["--speaker", str(int(speaker))])
    length_scale = tts_config.get("piper_length_scale")
    if length_scale is not None:
        cmd.extend(["--length_scale", str(float(length_scale))])
    noise_scale = tts_config.get("piper_noise_scale")
    if noise_scale is not None:
        cmd.extend(["--noise_scale", str(float(noise_scale))])
    noise_w = tts_config.get("piper_noise_w")
    if noise_w is not None:
        cmd.extend(["--noise_w", str(float(noise_w))])

    subprocess.run(cmd, input=text.encode("utf-8"), check=False)
    _apply_pitch_shift(out_wav, float(tts_config.get("piper_pitch_cents") or 0.0))
    return out_wav.exists() and out_wav.stat().st_size > 44

def speak(tts_config: dict, text: str) -> None:
    engine = tts_config.get("engine", "pyttsx3")
    if engine == "piper":
        out_wav = Path(TTS_WAV_FILENAME)
        ok = render_tts_to_wav(tts_config, text, out_wav)
        if ok:
            play_wav_plain(out_wav)
        return
    if engine == "espeak":
        rate, volume, voice, pitch = _resolve_tts_params(engine, tts_config)
        rate = str(int(rate))
        volume = str(int(max(0.0, min(1.0, volume)) * 200))
        cmd = ["espeak-ng", "-s", rate, "-a", volume]
        if pitch is not None:
            cmd.extend(["-p", str(int(pitch))])
        if voice:
            cmd.extend(["-v", voice])
        cmd.append(text)
        subprocess.run(cmd, check=False)
        return

    e = pyttsx3.init()
    rate, volume, voice, _ = _resolve_tts_params(engine, tts_config, pyttsx3_engine=e)
    e.setProperty("rate", int(rate))
    e.setProperty("volume", float(volume))
    if voice:
        e.setProperty("voice", voice)
    e.say(text)
    e.runAndWait()

def render_tts_to_wav(tts_config: dict, text: str, out_wav: Path) -> bool:
    engine = tts_config.get("engine", "pyttsx3")
    try:
        if out_wav.exists():
            out_wav.unlink()
    except Exception:
        pass

    if engine == "piper":
        return _piper_tts_to_wav(tts_config, text, out_wav)

    if engine == "espeak":
        rate, volume, voice, pitch = _resolve_tts_params(engine, tts_config)
        rate = str(int(rate))
        volume = str(int(max(0.0, min(1.0, volume)) * 200))
        cmd = ["espeak-ng", "-s", rate, "-a", volume, "-w", str(out_wav)]
        if pitch is not None:
            cmd.extend(["-p", str(int(pitch))])
        if voice:
            cmd.extend(["-v", voice])
        cmd.append(text)
        subprocess.run(cmd, check=False)
        return out_wav.exists() and out_wav.stat().st_size > 44

    try:
        tts = pyttsx3.init()
        rate, volume, voice, _ = _resolve_tts_params(engine, tts_config, pyttsx3_engine=tts)
        tts.setProperty("rate", int(rate))
        tts.setProperty("volume", float(volume))
        if voice:
            tts.setProperty("voice", voice)
        tts.save_to_file(text, str(out_wav))
        tts.runAndWait()
        return out_wav.exists() and out_wav.stat().st_size > 44
    except Exception:
        return False

def play_wav_plain(wav_path: Path) -> None:
    if not wav_path.exists():
        return

    wf = wave.open(str(wav_path), "rb")
    channels = wf.getnchannels()
    sample_width = wf.getsampwidth()
    framerate = wf.getframerate()

    pa_fmt = pa_format_from_sample_width(sample_width)
    if pa_fmt is None:
        wf.close()
        if platform.system().lower() == "linux":
            subprocess.run(["aplay", str(wav_path)], check=False)
        return

    p = pyaudio.PyAudio()
    stream = p.open(
        format=pa_fmt,
        channels=channels,
        rate=framerate,
        output=True,
    )
    try:
        while True:
            data = wf.readframes(1024)
            if not data:
                break
            stream.write(data)
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        wf.close()

def play_wav_with_lipsync(wav_path: Path, head_expression: str = None) -> None:
    if not wav_path.exists():
        return

    _idle_suspend()
    try:
        wf = wave.open(str(wav_path), "rb")
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        framerate = wf.getframerate()

        pa_fmt = pa_format_from_sample_width(sample_width)
        if pa_fmt is None:
            wf.close()
            if platform.system().lower() == "linux":
                subprocess.run(["aplay", str(wav_path)], check=False)
            return

        fps = max(5, int(LIPSYNC_FPS))
        chunk_frames = max(1, int(framerate / fps))

        chunks, rms_vals = [], []
        wf.rewind()
        while True:
            data = wf.readframes(chunk_frames)
            if not data:
                break
            chunks.append(data)
            rms_vals.append(rms_from_bytes(data, sample_width))

        if not chunks:
            wf.close()
            return

        vals_sorted = sorted(rms_vals)
        floor = percentile_sorted(vals_sorted, LIPSYNC_P_LOW)
        ceil = percentile_sorted(vals_sorted, LIPSYNC_P_HIGH)
        denom = max(1e-6, (ceil - floor))

        dt = 1.0 / fps
        attack_coeff = 1.0 - math.exp(-dt / max(1e-6, LIPSYNC_ATTACK_S))
        release_coeff = 1.0 - math.exp(-dt / max(1e-6, LIPSYNC_RELEASE_S))

        p = pyaudio.PyAudio()
        stream = p.open(
            format=pa_fmt,
            channels=channels,
            rate=framerate,
            output=True,
            frames_per_buffer=chunk_frames,
        )

        env = 0.0
        jaw_range = float(MOUTH_OPEN_DXL - MOUTH_CLOSE_DXL)
        last_jaw_dxl = float(MOUTH_CLOSE_DXL)

        _head_set_target(HEAD_IDLE_LR_DXL, HEAD_IDLE_UD_DXL)

        t0 = time.time()
        try:
            for i, data in enumerate(chunks):
                stream.write(data)

                _head_step(dt)

                x = (rms_vals[i] - floor) / denom
                x = max(0.0, min(1.0, x))
                if x <= LIPSYNC_DEADZONE:
                    target = 0.0
                else:
                    target = (x - LIPSYNC_DEADZONE) / max(1e-6, (1.0 - LIPSYNC_DEADZONE))
                target = max(0.0, min(1.0, target))
                target = target ** max(1e-6, LIPSYNC_GAMMA)

                coeff = attack_coeff if target > env else release_coeff
                env = env + (target - env) * coeff

                jaw_dxl = float(MOUTH_CLOSE_DXL) + env * jaw_range
                last_jaw_dxl = jaw_dxl

                t = time.time() - t0
                lr_off, ud_off, eye_off, brow_off = expression_offsets_continuous(head_expression, t)

                head_lr = _head_lr_cur + lr_off
                head_ud = _head_ud_cur + ud_off

                _eye_set_target(EYE_IDLE_DXL + eye_off)
                _eye_step(dt)

                _brow_set_target(BROW_IDLE_DXL + brow_off)
                _brow_step(dt)

                vals = _send_pose_outputs(jaw_dxl, head_lr, head_ud, _eye_cur, _brow_cur)
                _status.update_if_changed(*vals)

        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()
            wf.close()

            _smooth_return_to_neutral(last_jaw_dxl)
    finally:
        _idle_resume()

def speak_with_lipsync(tts_config: dict, text: str, head_expression: str = None) -> None:
    out_wav = Path(TTS_WAV_FILENAME)
    ok = render_tts_to_wav(tts_config, text, out_wav)
    if not ok:
        speak(tts_config, text)
        return
    play_wav_with_lipsync(out_wav, head_expression=head_expression)

# ------------------ Main ------------------
def main() -> int:
    global _dxl_bus, _send_blender
    global _head_lr_cur, _head_ud_cur, _head_lr_tgt, _head_ud_tgt
    global _eye_cur, _eye_tgt, _brow_cur, _brow_tgt
    global _idle_show_status

    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", "-a", type=Path)
    parser.add_argument("--whisper-model", "-w", default="tiny", choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--qwen-model", "-q", default=QWEN_MODEL)
    parser.add_argument("--keep-alive", default="10m")
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--no-tts", action="store_true")
    parser.add_argument("--tts-rate", type=int, default=170)
    parser.add_argument("--tts-volume", type=float, default=1.0)
    parser.add_argument("--tts-voice", default=LAFUFU_VOICE_PRESET)
    parser.add_argument("--tts-engine", choices=["piper", "auto", "pyttsx3", "espeak"], default="piper")
    parser.add_argument("--piper-model", default="", help="Piper model path. Empty = auto-select fastest available.")
    parser.add_argument("--piper-config", default="", help="Piper config path. Empty = auto from model.")
    parser.add_argument("--piper-speaker", type=int, default=PIPER_SPEAKER_DEFAULT)
    parser.add_argument("--piper-length-scale", type=float, default=PIPER_LENGTH_SCALE_DEFAULT)
    parser.add_argument("--piper-noise-scale", type=float, default=PIPER_NOISE_SCALE_DEFAULT)
    parser.add_argument("--piper-noise-w", type=float, default=PIPER_NOISE_W_DEFAULT)
    parser.add_argument("--piper-pitch", type=float, default=PIPER_PITCH_CENTS_DEFAULT, help="Pitch shift cents via librosa (0 disables).")
    parser.add_argument("--text-input", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--no-lipsync", action="store_true")
    parser.add_argument("--no-printer", action="store_true", help="Disable printing Lafufu replies.")

    # sim vs real
    parser.add_argument("--sim", action="store_true", help="Simulator only (no Dynamixel). Still sends UDP to Blender.")
    parser.add_argument("--no-blender", action="store_true", help="Disable UDP streaming (Blender).")
    parser.add_argument("--blender-host", default=BLENDER_UDP_HOST)
    parser.add_argument("--blender-port", type=int, default=BLENDER_UDP_PORT)

    # real Dynamixel params (default mode)
    parser.add_argument("--dxl-port", default="", help="e.g. COM3. If empty, auto-scan COM1..COM50.")
    parser.add_argument("--dxl-baud", type=int, default=0, help="If 0, tries common baud rates.")
    parser.add_argument("--dxl-hz", type=float, default=50.0, help="Max send rate to motors.")
    parser.add_argument("--leave-torque", action="store_true", help="Do not torque-disable on exit.")
    args = parser.parse_args()
    _idle_show_status = not args.text_input

    # Blender settings
    if args.no_blender:
        _send_blender = False
    else:
        _send_blender = True
        globals()["BLENDER_UDP_HOST"] = args.blender_host
        globals()["BLENDER_UDP_PORT"] = int(args.blender_port)

    enable_lipsync = not args.no_lipsync
    keep_alive = None if args.keep_alive == "" else args.keep_alive
    seed_jaw_dxl = MOUTH_CLOSE_DXL
    enable_printer = not args.no_printer

    # REAL Dynamixel default (unless --sim)
    if not args.sim:
        baud_candidates = [args.dxl_baud] if args.dxl_baud else [57600, 115200, 1000000, 2000000, 3000000, 4000000]
        port = args.dxl_port.strip()
        required_names = list(DXL_IDS.keys())

        if not port:
            found_port, found_baud = _auto_find_u2d2_port_and_baud(baud_candidates)
            if not found_port:
                chat_print("❌ Could not auto-find U2D2 bus. Provide --dxl-port COMx and --dxl-baud N.")
                return 2
            port, baud = found_port, found_baud
        else:
            baud = int(args.dxl_baud) if args.dxl_baud else 1000000

        chat_print(f"DXL: opening {port} @ {baud} ... (close Dynamixel Wizard first)")
        try:
            _dxl_bus = DynamixelBus(port=port, baud=baud, hz=args.dxl_hz, leave_torque=args.leave_torque)
            present_start = _dxl_bus.open(required_names=required_names)

            start_jaw_dxl, start_head_lr_dxl, start_head_ud_dxl, start_eye_dxl, start_brow_dxl = _startup_pose_from_present_dict(present_start)
            seed_jaw_dxl = start_jaw_dxl

            _head_lr_cur = _head_lr_tgt = start_head_lr_dxl
            _head_ud_cur = _head_ud_tgt = start_head_ud_dxl
            _eye_cur = _eye_tgt = start_eye_dxl
            _brow_cur = _brow_tgt = start_brow_dxl

            chat_print("✅ DXL connected.")
        except Exception as e:
            chat_print(f"❌ DXL init failed: {e}")
            return 2
    else:
        _dxl_bus = None
        chat_print("SIM mode: no Dynamixel output (UDP only).")

    # seed pose
    vals = _send_pose_outputs(seed_jaw_dxl, _head_lr_cur, _head_ud_cur, _eye_cur, _brow_cur)
    _status.update_if_changed(*vals)
    _start_idle_animation()

    whisper_model = None
    if not args.text_input:
        chat_print(f"Loading Whisper model '{args.whisper_model}'...")
        whisper_model = whisper.load_model(args.whisper_model)

    if not args.no_warmup:
        warmup_qwen(args.qwen_model, SYSTEM_PROMPT, keep_alive=keep_alive)

    tts_config = None
    if not args.no_tts:
        tts_config = init_tts(args.tts_rate, args.tts_volume, args.tts_voice or None, engine=args.tts_engine)
        if tts_config.get("engine") == "piper":
            tts_config.update({
                "piper_model": args.piper_model,
                "piper_config": args.piper_config,
                "piper_speaker": args.piper_speaker,
                "piper_length_scale": args.piper_length_scale,
                "piper_noise_scale": args.piper_noise_scale,
                "piper_noise_w": args.piper_noise_w,
                "piper_pitch_cents": args.piper_pitch,
            })

    try:
        while True:
            if args.text_input:
                text = chat_input("\nYou: ").strip()
                if not text:
                    if args.once:
                        return 0
                    continue
            else:
                if args.audio:
                    audio_path = args.audio
                    if not audio_path.exists():
                        chat_print(f"\n[Transcription]\n(missing file: {audio_path})\n")
                        return 1
                else:
                    audio_path = record_until_silence()

                text = transcribe(audio_path, model=whisper_model).strip()
                if not text:
                    if args.once:
                        return 0
                    continue

            chat_print(f'\n[Transcription]\n"{text}"\n')

            raw_reply = ask_qwen_stream_collect(
                text, model=args.qwen_model, system_prompt=SYSTEM_PROMPT, keep_alive=keep_alive
            )

            emotion, spoken = parse_emotion_and_spoken_text(raw_reply)
            expr = emotion_to_expression(emotion)

            chat_print(f'[Reply]\n"{raw_reply}"\n')

            if enable_printer and spoken:
                print_lafufu_text(spoken)

            if tts_config is None or not enable_lipsync:
                if expr:
                    perform_expression(expr)

            if tts_config is not None and spoken:
                if enable_lipsync:
                    speak_with_lipsync(tts_config, spoken, head_expression=expr)
                else:
                    speak(tts_config, spoken)

            if args.audio or args.once:
                return 0
    finally:
        if _dxl_bus is not None:
            _dxl_bus.close()

if __name__ == "__main__":
    raise SystemExit(main())