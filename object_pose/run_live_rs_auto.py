#!/usr/bin/env python3
"""
Intel RealSense D435 실시간 6D Pose Estimation (FoundationPose)
마스크 없이 depth 기반 자동 검출

키:
- s: 시작 (depth 범위 내 물체 자동 검출)
- r: 재초기화
- q: 종료
- +/-: 검출 거리 조절
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import logging
logging.disable(logging.CRITICAL)

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
    r = R.from_matrix(rot_mat)
    euler = r.as_euler('xyz', degrees=True)
    return euler, t


def create_mask_from_depth(depth_m, min_dist=0.2, max_dist=0.8):
    """
    Depth 기반 자동 마스크 생성
    min_dist ~ max_dist (m) 범위의 물체를 마스크로
    """
    mask = (depth_m > min_dist) & (depth_m < max_dist)
    mask = (mask * 255).astype(np.uint8)

    # morphology로 노이즈 제거
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 가장 큰 연결 영역만 남기기
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(mask)
        cv2.drawContours(mask, [largest], -1, 255, -1)

    return mask


def try_enable_ir_laser(dev: rs.device):
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
    parser.add_argument('--min_dist', type=float, default=0.2, help='최소 거리 (m)')
    parser.add_argument('--max_dist', type=float, default=0.8, help='최대 거리 (m)')
    parser.add_argument('--est_refine_iter', type=int, default=5)
    parser.add_argument('--track_refine_iter', type=int, default=2)
    args = parser.parse_args()

    set_seed(0)

    min_dist = args.min_dist
    max_dist = args.max_dist

    # ---------- Mesh ----------
    mesh = trimesh.load(args.mesh_file)
    mesh.vertices = mesh.vertices.astype(np.float32)
    mesh.vertex_normals = mesh.vertex_normals.astype(np.float32)

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

    pose = None
    initialized = False

    print("=" * 50)
    print("FoundationPose Live (Auto Mask from Depth)")
    print("=" * 50)
    print(f"거리 범위: {min_dist:.2f}m ~ {max_dist:.2f}m")
    print("s: start, r: reset, q: quit")
    print("+/-: 거리 조절")
    print("=" * 50)

    # ---------- RealSense pipeline ----------
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)

    profile = pipeline.start(config)
    dev = profile.get_device()
    try_enable_ir_laser(dev)

    align = rs.align(rs.stream.color)

    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_stream.get_intrinsics()
    K = np.array([[intr.fx, 0.0, intr.ppx],
                  [0.0, intr.fy, intr.ppy],
                  [0.0, 0.0, 1.0]], dtype=np.float32)

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

            rgb_frame = np.asanyarray(color_frame.get_data())
            depth_raw = np.asanyarray(depth_frame.get_data())

            color = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
            depth_m = depth_raw.astype(np.float32) * depth_scale

            vis = rgb_frame.copy()

            if not initialized:
                # 자동 마스크 미리보기
                auto_mask = create_mask_from_depth(depth_m, min_dist, max_dist)

                # 마스크 오버레이 (초록색)
                mask_overlay = np.zeros_like(vis)
                mask_overlay[:, :, 1] = auto_mask
                vis = cv2.addWeighted(vis, 0.7, mask_overlay, 0.3, 0)

                cv2.putText(vis, f"Range: {min_dist:.2f}m ~ {max_dist:.2f}m | Press 's' to start",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(vis, "+/-: adjust range",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            else:
                if pose is not None:
                    pose = est.track_one(rgb=color, depth=depth_m, K=K, iteration=args.track_refine_iter)

                    center_pose = pose @ np.linalg.inv(to_origin)
                    euler, t = pose_to_Rt(center_pose)

                    vis = draw_posed_3d_box(K, img=color, ob_in_cam=center_pose, bbox=bbox)
                    vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=K,
                                        thickness=3, transparency=0, is_input_rgb=True)
                    vis = vis[..., ::-1]

                    vis = np.ascontiguousarray(vis)
                    if vis.dtype != np.uint8:
                        vis = (np.clip(vis, 0, 255)).astype(np.uint8)

                    # R, t 표시
                    cv2.putText(vis, f"Rx:{euler[0]:6.1f} Ry:{euler[1]:6.1f} Rz:{euler[2]:6.1f} deg",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(vis, f"tx:{t[0]:5.3f} ty:{t[1]:5.3f} tz:{t[2]:5.3f} m",
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("FoundationPose Live (Auto)", vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                initialized = False
                pose = None
                print("Reset")
            elif key == ord('s'):
                # depth 기반 자동 마스크 생성 후 초기화
                auto_mask = create_mask_from_depth(depth_m, min_dist, max_dist)
                mask_pixels = (auto_mask > 0).sum()

                if mask_pixels > 1000:  # 최소 픽셀 수 체크
                    mask_bool = (auto_mask > 0)
                    pose = est.register(K=K, rgb=color, depth=depth_m, ob_mask=mask_bool,
                                        iteration=args.est_refine_iter)
                    initialized = True
                    print(f"Initialized! (mask pixels: {mask_pixels})")
                else:
                    print(f"Not enough pixels in range ({mask_pixels}). Adjust distance.")
            elif key == ord('+') or key == ord('='):
                max_dist = min(max_dist + 0.1, 3.0)
                print(f"Range: {min_dist:.2f}m ~ {max_dist:.2f}m")
            elif key == ord('-') or key == ord('_'):
                max_dist = max(max_dist - 0.1, min_dist + 0.1)
                print(f"Range: {min_dist:.2f}m ~ {max_dist:.2f}m")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
