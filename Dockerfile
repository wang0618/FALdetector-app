FROM python:3.6

RUN apt-get -qq update && apt-get -qq install -y --fix-missing --no-install-recommends cmake unzip \
    && rm -rf /var/lib/apt/lists/*

RUN wget --no-check-certificate -O /FALdetector.zip "https://github.com/PeterWang512/FALdetector/archive/master.zip" \
    && unzip -d / /FALdetector.zip \
    && mv FALdetector-master /app \
    && rm -f /FALdetector.zip \
    && cd /app \
    && mkdir -p /opt/tmp \
    # fix "Could not install packages due to an EnvironmentError: [Errno 28] No space left on device"
    && export TMPDIR=/opt/tmp \
    && pip install --build /opt/tmp --no-cache-dir torch torchvision \
    && pip install --build /opt/tmp --no-cache-dir -r requirements.txt \
    && bash weights/download_weights.sh

WORKDIR /app