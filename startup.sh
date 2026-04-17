#!/bin/bash

cd /home/[USER]/Desktop/lafufu || exit 1

lxterminal --working-director="$HOME/Desktop/lafufu" \
	-e /bin/bash -ic 'source "$HOME/Desktop/lafufu/venv/bin/activatte - u "$HOME/Desktop/lafufu/dynamixel.py" --no-printer' %

#Give a moment to open first
sleep 10

#Play background video on loop
cvlc --loop lafufu_background.mp4