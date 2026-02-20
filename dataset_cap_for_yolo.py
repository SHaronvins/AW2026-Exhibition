# from camera import MechEyeCamera
# import os
# import time
# from datetime import datetime
# import cv2
# import numpy as np

# SAVE_DIR = './dataset'
# os.makedirs(SAVE_DIR, exist_ok=True)

# cam = MechEyeCamera()
# cam.connect(None)

# try:
# 	idx = 0
# 	while True:
# 		rgb, depth, _ = cam.capture_textured_point_cloud()
# 		rgb_path = os.path.join(SAVE_DIR, f'rgb_{idx:06d}.png')
# 		depth_path = os.path.join(SAVE_DIR, f'depth_{idx:06d}.png')
# 		# RGB 저장 (cv2는 BGR이므로 변환)
# 		if rgb is not None:
# 			if rgb.shape[-1] == 3:
# 				cv2.imwrite(rgb_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
# 			else:
# 				cv2.imwrite(rgb_path, rgb)

# 		print(f'Saved: {rgb_path}, {depth_path}')
# 		idx += 1
# 		time.sleep(1)
# except KeyboardInterrupt:
# 	print('Stopped by user.')

from camera import MechEyeCamera
import os
import time
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO

# ===============================
# 설정
# ===============================

MODEL_PATH = "./best.pt"   # 학습된 모델
RESULT_DIR = "./result_live"
os.makedirs(RESULT_DIR, exist_ok=True)

CAPTURE_INTERVAL = 3.0  # 초

IMG_SIZE = 1024
CONF_THRES = 0.1
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
        rgb, depth, _ = cam.capture_textured_point_cloud()

        if rgb is None:
            print("[WARN] RGB capture failed")
            time.sleep(1)
            continue

        if rgb.shape[-1] == 3:
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        else:
            frame = rgb.copy()

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
        # 🖼 Overlay 결과
        # ---------------------------
        overlay = r.plot()

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(RESULT_DIR, f"result_{idx:06d}.png")
        cv2.imwrite(out_path, overlay)

        print(f"💾 Saved result: {out_path}")

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
