#!/usr/bin/env python3
"""
Intel RealSense D435 실시간 6D Pose Estimation (FoundationPose)
마스크 파일 기반 초기화

키:
- s: 시작 (마스크 기반 초기화)
- r: 재초기화
- q: 종료

입력:
- Color: 1280x720 (BGR8)
- Depth: 1280x720 (Z16) -> m 로 변환하여 FoundationPose에 전달
- K: Color intrinsics(1280x720)에서 추출

주의:
- mesh 단위(mm vs m) 반드시 확인 (depth는 m)
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# 로깅 끄기
import logging
logging.disable(logging.CRITICAL)

# estimater import 전에 로깅 비활성화
from estimater import *
import argparse
import cv2
import numpy as np
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as R


W, H = 1280, 720
FPS = 30


def pose_to_Rt(pose):
    """4x4 pose matrix -> R(euler xyz deg), t(xyz m)"""
    rot_mat = pose[:3, :3]
    t = pose[:3, 3]
    # 회전행렬 -> euler angles (xyz, degrees)
    r = R.from_matrix(rot_mat)
    euler = r.as_euler('xyz', degrees=True)
    return euler, t


def try_enable_ir_laser(dev: rs.device):
    """D435 계열: emitter / laser power 설정 시도 (지원되는 경우만)"""
    try:
        depth_sensor = dev.first_depth_sensor()
    except Exception:
        return

    try:
        if depth_sensor.supports(rs.option.emitter_enabled):
            depth_sensor.set_option(rs.option.emitter_enabled, 1.0)
    except Exception:
        pass

    try:
        if depth_sensor.supports(rs.option.laser_power):
            rng = depth_sensor.get_option_range(rs.option.laser_power)
            val = min(rng.max, max(rng.min, rng.max * 0.7))
            depth_sensor.set_option(rs.option.laser_power, val)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/obj_images/mesh/model.obj')
    parser.add_argument('--mask_file', type=str, default=f'{code_dir}/obj_images/masks/000000.png')
    parser.add_argument('--est_refine_iter', type=int, default=5)
    parser.add_argument('--track_refine_iter', type=int, default=2)
    args = parser.parse_args()

    set_seed(0)

    # ---------- Mesh ----------
    mesh = trimesh.load(args.mesh_file)
    # dtype 통일 (torch / nvdiffrast에서 dtype mismatch 방지)
    mesh.vertices = mesh.vertices.astype(np.float32)
    mesh.vertex_normals = mesh.vertex_normals.astype(np.float32)

    # ⚠️ OBJ가 mm 단위면 아래 주석 해제 (depth는 m)
    # mesh.apply_scale(0.001)

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2, 3)

    # ---------- FoundationPose ----------
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    debug_dir = f'{code_dir}/debug_live'
    os.makedirs(debug_dir, exist_ok=True)

    est = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug_dir=debug_dir,
        debug=0,
        glctx=glctx
    )
    # ---------- Mask ----------
    mask_img = cv2.imread(args.mask_file, cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        print(f"Error: mask file not found: {args.mask_file}")
        raise SystemExit(1)

    if mask_img.shape[:2] != (H, W):
        mask_img = cv2.resize(mask_img, (W, H), interpolation=cv2.INTER_NEAREST)

    mask_bool = (mask_img > 0)

    pose = None
    initialized = False

    print("s: start, r: reset, q: quit")

    # ---------- RealSense pipeline ----------
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)

    profile = pipeline.start(config)
    dev = profile.get_device()

    # IR / emitter on (가능한 경우)
    try_enable_ir_laser(dev)

    # Depth -> Color 정렬
    align = rs.align(rs.stream.color)

    # K: color intrinsics (W,H)
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_stream.get_intrinsics()
    K = np.array([[intr.fx, 0.0, intr.ppx],
                  [0.0, intr.fy, intr.ppy],
                  [0.0, 0.0, 1.0]], dtype=np.float32)

    # depth scale (meters per unit)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = float(depth_sensor.get_depth_scale())

    try:
        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            rgb_frame = np.asanyarray(color_frame.get_data())    # (H,W,3) BGR uint8
            depth_raw = np.asanyarray(depth_frame.get_data())    # (H,W) uint16 (unit)

            # FoundationPose 입력: RGB는 RGB 순서, depth는 meters(float32)
            color = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)

            depth_m = depth_raw.astype(np.float32) * depth_scale
            # (optional) invalid 0 처리
            # depth_m[depth_raw == 0] = 0.0

            vis = rgb_frame.copy()

            if not initialized:
                mask_overlay = np.zeros_like(vis)
                mask_overlay[:, :, 1] = mask_img
                vis = cv2.addWeighted(vis, 0.7, mask_overlay, 0.3, 0)
                cv2.putText(vis, "Press 's' to start (mask-based)", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                if pose is not None:
                    pose = est.track_one(rgb=color, depth=depth_m, K=K, iteration=args.track_refine_iter)

                    center_pose = pose @ np.linalg.inv(to_origin)

                    # R, t 추출
                    euler, t = pose_to_Rt(center_pose)

                    vis = draw_posed_3d_box(K, img=color, ob_in_cam=center_pose, bbox=bbox)
                    vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=K,
                                        thickness=3, transparency=0, is_input_rgb=True)
                    vis = vis[..., ::-1]  # RGB -> BGR

                    vis = np.ascontiguousarray(vis)
                    if vis.dtype != np.uint8:
                        vis = (np.clip(vis, 0, 255)).astype(np.uint8)

                    # 화면에 R, t 표시
                    cv2.putText(vis, f"Rx:{euler[0]:6.1f} Ry:{euler[1]:6.1f} Rz:{euler[2]:6.1f} deg",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(vis, f"tx:{t[0]:5.3f} ty:{t[1]:5.3f} tz:{t[2]:5.3f} m",
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("FoundationPose Live (D435)", vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print()  # 줄바꿈
                break
            elif key == ord('r'):
                initialized = False
                pose = None
                print("\nReset")
            elif key == ord('s'):
                pose = est.register(K=K, rgb=color, depth=depth_m, ob_mask=mask_bool,
                                    iteration=args.est_refine_iter)
                initialized = True
                print("\nInitialized!")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
