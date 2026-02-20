#!/usr/bin/env python3
"""
LabelMe JSON에서 마스크 이미지 생성
obj_images/rgb/*.json -> obj_images/masks/*.png
"""

import cv2
import numpy as np
import json
import os
from glob import glob


def create_mask_from_labelme(json_path, output_path):
    """LabelMe JSON에서 마스크 이미지 생성"""
    with open(json_path, 'r') as f:
        data = json.load(f)

    h = data['imageHeight']
    w = data['imageWidth']

    # 빈 마스크 생성
    mask = np.zeros((h, w), dtype=np.uint8)

    # shapes가 비어있으면 스킵
    if not data['shapes']:
        print(f"  [SKIP] {json_path} - 라벨 없음")
        return False

    # 각 shape에 대해 폴리곤 그리기
    for shape in data['shapes']:
        if shape['shape_type'] == 'polygon':
            points = np.array(shape['points'], dtype=np.int32)
            cv2.fillPoly(mask, [points], 255)

    # 마스크 저장
    cv2.imwrite(output_path, mask)
    return True


def main():
    base_dir = os.path.dirname(os.path.realpath(__file__))
    rgb_dir = os.path.join(base_dir, "obj_images", "rgb")
    mask_dir = os.path.join(base_dir, "obj_images", "masks")

    os.makedirs(mask_dir, exist_ok=True)

    # JSON 파일 찾기
    json_files = sorted(glob(os.path.join(rgb_dir, "*.json")))

    print("=" * 50)
    print("LabelMe JSON -> Mask 변환")
    print("=" * 50)
    print(f"JSON 파일 수: {len(json_files)}")
    print(f"저장 위치: {mask_dir}")
    print("=" * 50)

    success_count = 0
    for json_path in json_files:
        filename = os.path.basename(json_path).replace('.json', '.png')
        output_path = os.path.join(mask_dir, filename)

        if create_mask_from_labelme(json_path, output_path):
            print(f"  [OK] {filename}")
            success_count += 1

    print("=" * 50)
    print(f"완료! {success_count}/{len(json_files)} 마스크 생성됨")
    print("=" * 50)


if __name__ == "__main__":
    main()
