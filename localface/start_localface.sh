#!/bin/sh
cd "$(dirname "$0")"
python3 -c "import cv2, numpy" 2>/dev/null || python3 -m pip install -r requirements.txt
python3 -m localface
