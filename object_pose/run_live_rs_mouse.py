#!/usr/bin/env python3
"""
Intel RealSense D435 실시간 6D Pose Estimation (FoundationPose)
마우스 드래그로 ROI 선택하여 마스크 생성

키:
- 마우스 드래그: ROI 선택
- s: 시작 (선택한 ROI로 초기화)
- r: 재초기화
- q: 종료
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

# 마우스 드래그 상태
drawing = False
roi_start = None
roi_end = None
roi_selected = False


def mouse_callback(event, x, y, flags, param):
    global drawing, roi_start, roi_end, roi_selected

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        roi_start = (x, y)
        roi_end = (x, y)
        roi_selected = False

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            roi_end = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        roi_end = (x, y)
        # 최소 크기 체크
        if abs(roi_end[0] - roi_start[0]) > 20 and abs(roi_end[1] - roi_start[1]) > 20:
            roi_selected = True


def pose_to_Rt(pose):
    """4x4 pose matrix -> R(euler xyz deg), t(xyz m)"""
    rot_mat = pose[:3, :3]
    t = pose[:3, 3]
    r = R.from_matrix(rot_mat)
    euler = r.as_euler('xyz', degrees=True)
    return euler, t


def create_mask_from_roi(shape, roi_start, roi_end):
    """ROI 사각형에서 마스크 생성"""
    h, w = int(shape[0]), int(shape[1])
    mask = np.zeros((h, w), dtype=np.uint8)
    x1, y1 = min(roi_start[0], roi_end[0]), min(roi_start[1], roi_end[1])
    x2, y2 = max(roi_start[0], roi_end[0]), max(roi_start[1], roi_end[1])
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))
    mask[y1:y2, x1:x2] = 255
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


def start_realsense_pipeline(
    serial: str | None,
    color_width: int,
    color_height: int,
    color_fps: int,
    depth_width: int | None = None,
    depth_height: int | None = None,
    depth_fps: int | None = None,
    verbose: bool = False,
    no_fallback: bool = False,
):
    """Start RealSense with a reasonable fallback list if the exact request can't be resolved.

    Notes:
    - On D435-class devices, high color resolutions may work while matching depth resolution may not.
      Using (color 1280x720) + (depth 848x480) is often the best compromise.
    """
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        raise RuntimeError(
            "No RealSense device found. Check USB connection and that no other app is using the camera."
        )

    pipeline = rs.pipeline()

    if depth_width is None:
        depth_width = color_width
    if depth_height is None:
        depth_height = color_height
    if depth_fps is None:
        depth_fps = color_fps

    # Try a few common D435-compatible profiles.
    # Each candidate is (cw, ch, cfps, dw, dh, dfps)
    candidates: list[tuple[int, int, int, int, int, int]] = [
        (color_width, color_height, color_fps, depth_width, depth_height, depth_fps),

        # Common combos
        (1280, 720, 30, 1280, 720, 30),
        (1280, 720, 30, 848, 480, 30),
        (1280, 720, 15, 848, 480, 15),
        (848, 480, 30, 848, 480, 30),
        (640, 480, 30, 640, 480, 30),
        (640, 360, 30, 640, 360, 30),
        (424, 240, 30, 424, 240, 30),
    ]

    if no_fallback:
        candidates = candidates[:1]

    last_err: Exception | None = None
    for cw, ch, cf, dw, dh, df in candidates:
        try:
            cfg = rs.config()
            if serial:
                cfg.enable_device(serial)
            cfg.enable_stream(rs.stream.color, cw, ch, rs.format.bgr8, cf)
            cfg.enable_stream(rs.stream.depth, dw, dh, rs.format.z16, df)

            profile = pipeline.start(cfg)
            # Print actual negotiated profiles.
            cprof = profile.get_stream(rs.stream.color).as_video_stream_profile()
            dprof = profile.get_stream(rs.stream.depth).as_video_stream_profile()
            c = cprof.width(), cprof.height(), cprof.fps()
            d = dprof.width(), dprof.height(), dprof.fps()
            print(f"[RealSense] Started streams: color {c[0]}x{c[1]}@{c[2]}, depth {d[0]}x{d[1]}@{d[2]}")
            return pipeline, profile
        except Exception as e:
            last_err = e
            if verbose:
                print(f"[RealSense] Failed: color {cw}x{ch}@{cf}, depth {dw}x{dh}@{df} -> {e}")
            try:
                pipeline.stop()
            except Exception:
                pass

    raise RuntimeError(
        "Couldn't resolve RealSense stream requests. "
        "This usually means the requested (width/height/fps) combo isn't supported, "
        "or the device is already in use. "
        f"Last error: {last_err}"
    )
    try:
        if depth_sensor.supports(rs.option.laser_power):
            rng = depth_sensor.get_option_range(rs.option.laser_power)
            val = min(rng.max, max(rng.min, rng.max * 0.7))
            depth_sensor.set_option(rs.option.laser_power, val)
    except Exception:
        pass


def main():
    global drawing, roi_start, roi_end, roi_selected

    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/obj_images/mesh/model.obj')
    parser.add_argument('--est_refine_iter', type=int, default=5)
    parser.add_argument('--track_refine_iter', type=int, default=2)
    parser.add_argument('--rs_width', type=int, default=W)
    parser.add_argument('--rs_height', type=int, default=H)
    parser.add_argument('--rs_fps', type=int, default=FPS)
    parser.add_argument('--rs_color_width', type=int, default=None)
    parser.add_argument('--rs_color_height', type=int, default=None)
    parser.add_argument('--rs_color_fps', type=int, default=None)
    parser.add_argument('--rs_depth_width', type=int, default=None)
    parser.add_argument('--rs_depth_height', type=int, default=None)
    parser.add_argument('--rs_depth_fps', type=int, default=None)
    parser.add_argument('--rs_verbose', action='store_true', help='Print RealSense fallback attempts')
    parser.add_argument('--rs_no_fallback', action='store_true', help='Disable fallback; fail if requested mode not available')
    parser.add_argument('--rs_serial', type=str, default=None, help='Optional RealSense device serial')
    args = parser.parse_args()

    set_seed(0)

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
    print("FoundationPose Live (Mouse ROI Selection)")
    print("=" * 50)
    print("마우스로 물체 영역을 드래그하세요")
    print("s: start, r: reset, q: quit")
    print("=" * 50)

    # ---------- RealSense pipeline ----------
    cw = args.rs_color_width if args.rs_color_width is not None else args.rs_width
    ch = args.rs_color_height if args.rs_color_height is not None else args.rs_height
    cf = args.rs_color_fps if args.rs_color_fps is not None else args.rs_fps
    pipeline, profile = start_realsense_pipeline(
        serial=args.rs_serial,
        color_width=cw,
        color_height=ch,
        color_fps=cf,
        depth_width=args.rs_depth_width,
        depth_height=args.rs_depth_height,
        depth_fps=args.rs_depth_fps,
        verbose=args.rs_verbose,
        no_fallback=args.rs_no_fallback,
    )
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

    # 마우스 콜백 설정
    window_name = "FoundationPose Live (Mouse)"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

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
                # ROI 드래그 중이거나 선택됨
                if roi_start is not None and roi_end is not None:
                    x1, y1 = min(roi_start[0], roi_end[0]), min(roi_start[1], roi_end[1])
                    x2, y2 = max(roi_start[0], roi_end[0]), max(roi_start[1], roi_end[1])

                    # 사각형 표시
                    color_rect = (0, 255, 0) if roi_selected else (0, 255, 255)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), color_rect, 2)

                    # 선택된 영역 반투명 오버레이
                    if roi_selected:
                        overlay = vis.copy()
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)
                        vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)

                if roi_selected:
                    cv2.putText(vis, "ROI selected! Press 's' to start", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(vis, "Drag mouse to select object region", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
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

            cv2.imshow(window_name, vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                initialized = False
                pose = None
                roi_start = None
                roi_end = None
                roi_selected = False
                print("Reset")
            elif key == ord('s') and roi_selected:
                # ROI에서 마스크 생성
                # Use current frame size (fallback resolution may differ from global H/W).
                mask = create_mask_from_roi(depth_m.shape[:2], roi_start, roi_end)
                mask_bool = (mask > 0)

                pose = est.register(K=K, rgb=color, depth=depth_m, ob_mask=mask_bool,
                                    iteration=args.est_refine_iter)
                initialized = True
                print("Initialized!")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
