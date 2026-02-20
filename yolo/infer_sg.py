from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np

# ===============================
# 설정
# ===============================

MODEL_PATH = "./best.pt"

INPUT_DIR = Path("./dataset/val/images")

OUT_ROOT = Path("./result")
OVERLAY_DIR = OUT_ROOT / "overlay"
MASK_DIR = OUT_ROOT / "masks"

OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
MASK_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 1024
CONF_THRES = 0.1
IOU_THRES = 0.7

# ===============================
# 모델 로드
# ===============================

model = YOLO(MODEL_PATH)

# ===============================
# 전체 이미지 추론
# ===============================

image_paths = sorted(INPUT_DIR.glob("*.jpg"))

print(f"[INFO] Found {len(image_paths)} images")

for img_path in image_paths:

    print(f"[RUN] {img_path.name}")

    results = model.predict(
        source=str(img_path),
        imgsz=IMG_SIZE,
        conf=CONF_THRES,
        iou=IOU_THRES,
        device=0,
        verbose=False,
    )

    r = results[0]

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[WARN] Failed to load {img_path}")
        continue

    h, w = img.shape[:2]

    # -------------------------------
    # ✅ 1. Overlay 이미지 저장
    # -------------------------------
    overlay = r.plot()
    cv2.imwrite(str(OVERLAY_DIR / img_path.name), overlay)

    # -------------------------------
    # ✅ 2. 마스크만 따로 저장
    # -------------------------------
    if r.masks is not None:

        masks = r.masks.data.cpu().numpy()  # (N, H, W)

        for i, mask in enumerate(masks):

            mask_bin = (mask > 0.5).astype(np.uint8) * 255

            mask_path = MASK_DIR / f"{img_path.stem}_inst{i}.png"
            cv2.imwrite(str(mask_path), mask_bin)

print("✅ Inference finished!")
print(f"📁 Overlay saved to: {OVERLAY_DIR.resolve()}")
print(f"📁 Masks saved to: {MASK_DIR.resolve()}")
