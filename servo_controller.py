#!/usr/bin/env python3
"""Numpad controller for Lafufu servos.

Numpad layout:
  7 = jaw open        8 = head up       9 = jaw close
  4 = eyes left       5 = center all    6 = eyes right
  1 = brow down       2 = head down     3 = brow up
  + = head left       - = head right
  0 = print positions

  q = quit
"""

import sys
import time
import tty
import termios

from dynamixel_sdk import PortHandler, PacketHandler

PORT = "/dev/ttyUSB0"
BAUD = 57600

DXL_IDS = {"jaw": 4, "head_ud": 2, "head_lr": 1, "eye": 5, "brow": 3}

ADDR_TORQUE = 64
ADDR_GOAL = 116
ADDR_PRESENT = 132

STEP = {
    "jaw": 15,
    "head_ud": 20,
    "head_lr": 20,
    "eye": 10,
    "brow": 5,
}

CLAMP = {
    "jaw":     (1534, 1728),
    "head_ud": (2885, 3278),
    "head_lr": (1828, 2298),
    "eye":     (1960, 2130),
    "brow":    (2051, 2099),
}


def clamp(name, val):
    lo, hi = CLAMP[name]
    mn, mx = min(lo, hi), max(lo, hi)
    return max(mn, min(mx, val))


def read_pos(pk, port, dxl_id):
    val, _, _ = pk.read4ByteTxRx(port, dxl_id, ADDR_PRESENT)
    return val


def write_pos(pk, port, dxl_id, pos):
    pk.write4ByteTxRx(port, dxl_id, ADDR_GOAL, pos)


def getch():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def main():
    port = PortHandler(PORT)
    port.openPort()
    port.setBaudRate(BAUD)
    pk = PacketHandler(2.0)

    # Enable torque and read current positions
    pos = {}
    for name, dxl_id in DXL_IDS.items():
        pk.write1ByteTxRx(port, dxl_id, ADDR_TORQUE, 1)
        pos[name] = read_pos(pk, port, dxl_id)

    print(__doc__)
    print("Current positions:")
    for name, v in pos.items():
        lo, hi = CLAMP[name]
        print(f"  {name:10s}: {v}  (range {min(lo,hi)}-{max(lo,hi)})")
    print()

    # Key mappings: key -> (servo_name, direction)
    keymap = {
        "7": ("jaw", -1),       # jaw open (lower value)
        "9": ("jaw", +1),       # jaw close (higher value)
        "8": ("head_ud", -1),   # head up (lower value)
        "2": ("head_ud", +1),   # head down (higher value)
        "4": ("eye", -1),       # eyes left
        "6": ("eye", +1),       # eyes right
        "+": ("head_lr", +1),   # head left
        "-": ("head_lr", -1),   # head right
        "1": ("brow", -1),      # brow down
        "3": ("brow", +1),      # brow up
    }

    try:
        while True:
            ch = getch()
            if ch in ("q", "Q", "\x03"):  # q or Ctrl+C
                break

            if ch == "0":
                print("\rPositions:                              ")
                for name in DXL_IDS:
                    actual = read_pos(pk, port, DXL_IDS[name])
                    print(f"  {name:10s}: {actual}")
                continue

            if ch == "5":
                # Center all
                for name in DXL_IDS:
                    lo, hi = CLAMP[name]
                    mid = (min(lo, hi) + max(lo, hi)) // 2
                    pos[name] = mid
                    write_pos(pk, port, DXL_IDS[name], mid)
                print(f"\r  ALL -> CENTER                         ", end="", flush=True)
                continue

            if ch in keymap:
                name, direction = keymap[ch]
                pos[name] = clamp(name, pos[name] + direction * STEP[name])
                write_pos(pk, port, DXL_IDS[name], pos[name])
                print(f"\r  {name}: {pos[name]}      ", end="", flush=True)

    except KeyboardInterrupt:
        pass
    finally:
        print("\nDisabling torque...")
        for name, dxl_id in DXL_IDS.items():
            pk.write1ByteTxRx(port, dxl_id, ADDR_TORQUE, 0)
        port.closePort()
        print("Done.")


if __name__ == "__main__":
    main()
