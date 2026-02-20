import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # .../fairino/windows
SDK_DIR = os.path.join(BASE_DIR, "fairino")             # .../fairino/windows/fairino

if SDK_DIR not in sys.path:
    sys.path.insert(0, SDK_DIR)

import Robot
RPC = Robot.RPC