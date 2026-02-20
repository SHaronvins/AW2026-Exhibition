#!/usr/bin/env python3
"""
Intel RealSense D435 물체 참조 이미지 촬영 스크립트
스페이스바: 촬영 / q: 종료

저장:
- obj_images/rgb/000000.png (BGR 8-bit)
- obj_images/depth/000000.png (uint16, mm)
- obj_images/cam_K.txt (3x3)
"""

import os
import cv2
import numpy as np
import pyrealsense2 as rs

W, H = 1280, 720
FPS = 30

def depth_to_vis(depth_u16: np.ndarray) -> np.ndarray:
    """uint16 depth(mm)를 보기 좋은 컬러맵으로 변환"""
    if depth_u16 is None:
        return None
    d = depth_u16.astype(np.float32)
    valid = d[d > 0]
    if valid.size < 10:
        norm = np.zeros_like(d, dtype=np.uint8)
    else:
        vmin = np.percentile(valid, 5)
        vmax = np.percentile(valid, 95)
        if vmax <= vmin:
            vmax = vmin + 1.0
        norm = np.clip((d - vmin) * 255.0 / (vmax - vmin), 0, 255).astype(np.uint8)
    return cv2.applyColorMap(norm, cv2.COLORMAP_JET)

def try_enable_ir_laser(dev: rs.device):
    """
    D435 계열: Depth Sensor(스테레오 IR)에서 emitter(프로젝터) / laser power 설정 시도.
    환경/펌웨어/모델에 따라 지원 범위 다를 수 있음.
    """
    try:
        depth_sensor = dev.first_depth_sensor()
    except Exception:
        print("[IR] depth sensor not found")
        return

    # Emitter on/off
    try:
        if depth_sensor.supports(rs.option.emitter_enabled):
            depth_sensor.set_option(rs.option.emitter_enabled, 1.0)
            print("[IR] emitter_enabled = 1")
    except Exception as e:
        print("[IR] emitter_enabled not supported:", e)

    # Laser power (0 ~ 360 보통)
    try:
        if depth_sensor.supports(rs.option.laser_power):
            rng = depth_sensor.get_option_range(rs.option.laser_power)
            # 적당한 기본값 (최대의 ~70% 정도)
            val = min(rng.max, max(rng.min, rng.max * 0.7))
            depth_sensor.set_option(rs.option.laser_power, val)
            print(f"[IR] laser_power = {val} (range {rng.min}..{rng.max})")
    except Exception as e:
        print("[IR] laser_power not supported:", e)

def main():
    base_dir = os.path.dirname(os.path.realpath(__file__))
    save_dir = os.path.join(base_dir, "obj_images")
    rgb_dir = os.path.join(save_dir, "rgb")
    depth_dir = os.path.join(save_dir, "depth")
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    existing_files = [f for f in os.listdir(rgb_dir) if f.endswith(".png")]
    capture_count = len(existing_files)

    print("=" * 50)
    print("Intel RealSense D435 촬영 스크립트")
    print("=" * 50)
    print("SPACE: Capture")
    print("q: Quit")
    print(f"저장 위치: {save_dir}")
    print(f"기존 이미지 수: {capture_count}")
    print("=" * 50)

    # -------- RealSense pipeline --------
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)

    profile = pipeline.start(config)
    dev = profile.get_device()

    # IR projector(Emitter) 켜기 시도
    try_enable_ir_laser(dev)

    # Align depth to color (DepthAI의 setDepthAlign(CAM_A)와 유사)
    align = rs.align(rs.stream.color)

    # Intrinsics 저장 (Color 기준)
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_stream.get_intrinsics()
    K = np.array([
        [intr.fx, 0.0, intr.ppx],
        [0.0, intr.fy, intr.ppy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    k_path = os.path.join(save_dir, "cam_K.txt")
    np.savetxt(k_path, K, fmt="%.6f")
    print(f"카메라 내부 파라미터 저장: {k_path}")
    print("K=\n", K)

    latest_rgb = None
    latest_depth_mm = None

    try:
        while True:
            # 프레임 수신
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # numpy 변환
            latest_rgb = np.asanyarray(color_frame.get_data())  # BGR8
            depth_raw = np.asanyarray(depth_frame.get_data())   # uint16, depth units(보통 mm가 아님!)

            # RealSense depth scale 적용해서 mm로 변환
            depth_sensor = profile.get_device().first_depth_sensor()
            depth_scale = depth_sensor.get_depth_scale()  # meters per unit
            depth_m = depth_raw.astype(np.float32) * depth_scale
            latest_depth_mm = (depth_m * 1000.0).astype(np.uint16)

            # 표시
            display_rgb = latest_rgb.copy()
            cv2.putText(display_rgb, f"Captured: {capture_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_rgb, "SPACE: Capture / Q: Quit", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("RGB", display_rgb)

            depth_vis = depth_to_vis(latest_depth_mm)
            if depth_vis is not None:
                cv2.imshow("Depth(mm)", depth_vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            if key == ord(' ') and latest_rgb is not None and latest_depth_mm is not None:
                filename = f"{capture_count:06d}.png"

                rgb_path = os.path.join(rgb_dir, filename)
                depth_path = os.path.join(depth_dir, filename)

                ok1 = cv2.imwrite(rgb_path, latest_rgb)
                ok2 = cv2.imwrite(depth_path, latest_depth_mm)

                if ok1 and ok2:
                    capture_count += 1
                    print(f"[{capture_count}] 저장 완료: {filename}")
                else:
                    print(f"[WARN] 저장 실패: rgb={ok1}, depth={ok2}, file={filename}")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    print("=" * 50)
    print(f"촬영 완료! 총 {capture_count}장")
    print(f"저장 위치: {save_dir}")
    print("=" * 50)

if __name__ == "__main__":
    main()
