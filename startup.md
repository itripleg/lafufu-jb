1) Create startup script:
sudo nano /startup.sh

startup.sh:
    #!/bin/bash

    cd /home/[USER]/Desktop/lafufu || exit 1

    lxterminal --working-director="$HOME/Desktop/lafufu" \
        -e /bin/bash -ic 'source "$HOME/Desktop/lafufu/venv/bin/activatte - u "$HOME/Desktop/lafufu/dynamixel.py" --no-printer' %

    #Give a moment to open first
    sleep 10

    #Play background video on loop
    cvlc --loop lafufu_background.mp4


2) sudo chmod +x /startup.sh

3) Enable autostart:
    mkdir -p ~/.config/labwc
    nano ~/.config/labwc/autostart

    Add:
    /startup.sh &

4) sudo nano /etc/systemd/system/kiosk.service

kiosk.service:
    [Unit]
    Description=Lafufu Startup

    [Service]
    ExecStart=/startup.sh
    Restart=always

    [Install]
    WantedBy=multi-user.target

5) Then:
sudo systemctl daemon-reload
sudo systemctl enable kiosk.service