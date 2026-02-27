from camera import MechEyeCamera
import os
import time
from datetime import datetime

import cv2
from ultralytics import YOLO

# ===============================
# 설정
# ===============================

SAVE_DIR = "./dataset/rgb"
RESULT_DIR = "./dataset/results"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

CAPTURE_INTERVAL = 3.0  # 초

MODEL_PATH = "./best.pt"
IMG_SIZE = 1024
CONF_THRES = 0.25
IOU_THRES = 0.7

# ===============================
# YOLO 모델 로드
# ===============================

model = YOLO(MODEL_PATH)

# ===============================
# 카메라 연결
# ===============================

cam = MechEyeCamera()
cam.connect(None)

print("✅ Camera connected")

try:
    idx = 0

    while True:

        start_t = time.time()

        # ---------------------------
        # 📷 캡처
        # ---------------------------
        rgb, _, _ = cam.capture_textured_point_cloud()

        if rgb is None:
            print("[WARN] RGB capture failed")
            time.sleep(1)
            continue

        if rgb.shape[-1] == 3:
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        else:
            frame = rgb.copy()

        # ---------------------------
        # 💾 원본 저장
        # ---------------------------
        rgb_path = os.path.join(SAVE_DIR, f"rgb_{idx:06d}.png")
        cv2.imwrite(rgb_path, frame)

        # ---------------------------
        # 🔍 YOLO 추론
        # ---------------------------
        results = model.predict(
            source=frame,
            imgsz=IMG_SIZE,
            conf=CONF_THRES,
            iou=IOU_THRES,
            device=0,
            verbose=False,
        )

        r = results[0]

        # ---------------------------
        # 🖼 결과 overlay
        # ---------------------------
        overlay = r.plot()

        result_path = os.path.join(RESULT_DIR, f"result_{idx:06d}.png")
        cv2.imwrite(result_path, overlay)

        print(f"💾 Saved RGB: {rgb_path}")
        print(f"🔍 Saved Result: {result_path}")

        idx += 1

        # ---------------------------
        # ⏱ 3초 주기 유지
        # ---------------------------
        elapsed = time.time() - start_t
        sleep_t = max(0.0, CAPTURE_INTERVAL - elapsed)
        time.sleep(sleep_t)

except KeyboardInterrupt:
    print("🛑 Stopped by user")

finally:
    try:
        cam.disconnect()
    except Exception:
        pass