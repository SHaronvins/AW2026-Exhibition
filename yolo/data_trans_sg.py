import json
import os
from pathlib import Path


def convert_coco_to_yolo_seg(coco_json, label_dir):
    with open(coco_json, "r") as f:
        coco = json.load(f)

    os.makedirs(label_dir, exist_ok=True)

    # category_id → class index
    cat_id_to_cls = {
        cat["id"]: i for i, cat in enumerate(coco["categories"])
    }

    images = {img["id"]: img for img in coco["images"]}

    # 기존 label 파일 초기화
    for f in Path(label_dir).glob("*.txt"):
        f.unlink()

    for ann in coco["annotations"]:

        # segmentation 없는 annotation 스킵
        if "segmentation" not in ann:
            continue

        seg = ann["segmentation"]

        # RLE 마스크 형식 방어
        if isinstance(seg, dict):
            continue

        img = images[ann["image_id"]]
        w, h = img["width"], img["height"]

        cls = cat_id_to_cls[ann["category_id"]]

        label_path = Path(label_dir) / (Path(img["file_name"]).stem + ".txt")

        with open(label_path, "a") as f:

            for poly in seg:

                if len(poly) < 6:
                    continue  # 점 3개 미만 스킵

                coords = []

                for i in range(0, len(poly), 2):
                    x = poly[i] / w
                    y = poly[i + 1] / h
                    coords.append(f"{x:.6f}")
                    coords.append(f"{y:.6f}")

                line = f"{cls} " + " ".join(coords)
                f.write(line + "\n")


# === 실행 ===
convert_coco_to_yolo_seg(
    "./dataset/train/annotations.json",
    "./dataset/train/labels"
)

convert_coco_to_yolo_seg(
    "./dataset/val/annotations.json",
    "./dataset/val/labels"
)
