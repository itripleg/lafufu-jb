# lafufu

## Piper on Raspberry Pi (Speaker Setup)

List available playback devices:

```bash
aplay -l
```

Find your speaker in the list and note its **card** and **device** numbers.  
Example: `[SPEAKER NAME] = card 3, device 0`

Export the ALSA device for this app:

```bash
export LAFUFU_APLAY_DEVICE=plughw:3,0
```

Start the script:

```bash
python dynamixel.py
```
