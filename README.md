# lafufu-jb

Lafufu robot controller — voice-interactive animatronic with Dynamixel servos, Whisper STT, Qwen LLM, and Piper TTS. Runs on Raspberry Pi 5.

## What's Different in This Fork

Compared to the upstream `tinker-vr/lafufu`:

- **Mic support for Shure MV88** — records at 44100 Hz (native rate) and resamples to 16 kHz for Whisper
- **Whisper forced to English** — no more random language detection
- **Jabra SPEAK 510 as default speaker** — audio goes to USB speaker instead of HDMI
- **Wider eye servo range** — mapped actual physical limits (1960–2130) instead of the old 2000–2077
- **Signal cleanup** — SIGTERM/SIGINT handler disables servo torque on shutdown
- **Silence threshold flag** — `--silence-threshold` to tune voice activity detection
- **System prompt override** — `--system-prompt` to change Labubu's personality
- **Debug mode** — `--debug` loops eye/jaw/nod sweeps for servo testing
- **Numpad servo controller** — `servo_controller.py` for manual servo positioning
- **Chromium keyring fix** — set to basic password store so no unlock popup on boot

## Pi Setup

### Hardware

- Raspberry Pi 5
- Dynamixel servos via U2D2 on `/dev/ttyUSB0` @ 57600 baud
- Shure MV88-USBC microphone
- Jabra SPEAK 510 USB speaker
- Servo IDs: head_lr=1, head_ud=2, brow=3, jaw=4, eye=5

### Software

The repo lives at `/lafufu` on the Pi. The Python venv is at `~/lafufu-env`.

```bash
# Activate the venv (always do this first)
source ~/lafufu-env/bin/activate

# Run the robot
cd /lafufu
python dynamixel.py

# Or with options
python dynamixel.py --text-input              # type instead of talk
python dynamixel.py --silence-threshold 1200  # noisy room
python dynamixel.py --system-prompt "You are a pirate. Say arr. Use [happy] emotion tags."
python dynamixel.py --debug                   # test servo movement
python dynamixel.py --sim                     # no hardware, UDP only
```

### Servo Controller

For manual servo testing with the numpad:

```bash
source ~/lafufu-env/bin/activate
cd /lafufu
python servo_controller.py
```

```
  7 = jaw open        8 = head up       9 = jaw close
  4 = eyes left       5 = center all    6 = eyes right
  1 = brow down       2 = head down     3 = brow up
  + = head left       - = head right
  0 = print positions    q = quit
```

### Pulling Updates

```bash
cd /lafufu
git pull
```

If git asks for credentials, the remote is set to `https://github.com/itripleg/lafufu-jb.git`.

## Remote Access

The Pi uses [bore](https://github.com/ekzhang/bore) for remote SSH tunneling. Once bore is running on the Pi, connect with:

```bash
ssh -p <PORT> lafufu@bore.pub
```

To check if a port is open before connecting: https://www.yougetsignal.com/tools/open-ports/ — enter `bore.pub` and the port number.

## Useful Commands

```bash
# Check if dynamixel is running
ps aux | grep dynamixel

# Kill it
pkill -f dynamixel.py

# Syntax check after editing
python -c "import py_compile; py_compile.compile('dynamixel.py', doraise=True)"

# Test audio output (uses ALSA default; list devices with `aplay -L`,
# or set LAFUFU_APLAY_DEVICE to override the device the app uses)
aplay -D default /tmp/test.wav

# Generate TTS manually
source ~/lafufu-env/bin/activate
echo "Hello world" | piper --model /lafufu/models/*.onnx --output_file /tmp/test.wav
aplay -D default /tmp/test.wav
```
