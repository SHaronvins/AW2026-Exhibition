import os
import sys
import time
import warnings
from types import SimpleNamespace
import math

import cv2
import numpy as np
from base_coordi import convert_cam_to_base, cam_pose6_to_T
from circle_move import circle_move_0, move_offset_0, circle_move_1, move_offset_1, compute_rotvec_lookat_vertical
from ultralytics import YOLO
import threading
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

UR5E_IP = "192.168.2.100"
ROBOT_POSES_TXT = "robot_poses.txt"

class RTDEKeepAliveThread(threading.Thread):
    def __init__(self, ip, interval=1.0):
        super().__init__(daemon=True)
        self.interval = interval
        self.running = True
        self.rtde_recv = RTDEReceiveInterface(ip)

    def run(self):
        print("[RTDE KeepAlive] started")
        while self.running:
            try:
                tcp = self.rtde_recv.getActualTCPPose()
            except Exception as e:
                print("[RTDE KeepAlive ERROR]", e)

            time.sleep(self.interval)

    def stop(self):
        self.running = False
        try:
            self.rtde_recv.disconnect()
        except:
            pass

def _load_robot_poses_txt(path: str) -> list[dict[str, float]]:

    poses: list[dict[str, float]] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:

            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = [p.strip() for p in line.split(",")]

            if len(parts) != 6:
                print("⚠ invalid pose line:", line)
                continue

            vals = list(map(float, parts))

            pose = {
                "x": vals[0],
                "y": vals[1],
                "z": vals[2],
                "rx": vals[3],
                "ry": vals[4],
                "rz": vals[5],
            }

            poses.append(pose)

    return poses

def _normalize_depth_for_view(depth_m: np.ndarray, d_min_m: float = None, d_max_m: float = None) -> np.ndarray:

    if depth_m is None or depth_m.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    d = depth_m.copy()
    invalid = ~np.isfinite(d) | (d <= 0)
    if np.all(invalid):
        return np.zeros(d.shape, dtype=np.uint8)
    valid_vals = d[~invalid]
    if d_min_m is None:
        d_min_m = float(valid_vals.min())
    if d_max_m is None:
        d_max_m = float(valid_vals.max())
    if d_max_m - d_min_m < 1e-6:
        d_max_m = d_min_m + 0.1
    d = np.clip(d, d_min_m, d_max_m)
    d = (255.0 * (d_max_m - d) / (d_max_m - d_min_m)).astype(np.uint8)
    d[invalid] = 0
    return d

def _load_K(args: SimpleNamespace, width: int, height: int) -> np.ndarray:
    if args.fx > 0 and args.fy > 0:
        cx = args.cx if args.cx >= 0 else (width - 1) / 2.0
        cy = args.cy if args.cy >= 0 else (height - 1) / 2.0
        return np.array([[args.fx, 0.0, cx], [0.0, args.fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    fx = max(width, height) * 1.2
    fy = fx
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


def _scale_K(K: np.ndarray, sx: float, sy: float) -> np.ndarray:
    K2 = np.asarray(K, dtype=np.float32).copy()
    K2[0, 0] *= float(sx)
    K2[1, 1] *= float(sy)
    K2[0, 2] *= float(sx)
    K2[1, 2] *= float(sy)
    return K2

def _rot_to_rpy(R: np.ndarray) -> tuple[float, float, float]:
    sy = float(np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0]))
    if sy < 1e-6:
        roll = float(np.arctan2(-R[1, 2], R[1, 1]))
        pitch = float(np.arctan2(-R[2, 0], sy))
        yaw = 0.0
    else:
        roll = float(np.arctan2(R[2, 1], R[2, 2]))
        pitch = float(np.arctan2(-R[2, 0], sy))
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    return roll, pitch, yaw

def _setup_foundationpose(args: SimpleNamespace):
    code_dir = os.path.dirname(os.path.realpath(__file__))
    object_pose_dir = os.path.join(code_dir, "object_pose")
    if object_pose_dir not in sys.path:
        sys.path.insert(0, object_pose_dir)

    warnings.filterwarnings("ignore")
    import logging

    logging.disable(logging.CRITICAL)

    from estimater import (
        FoundationPose,
        PoseRefinePredictor,
        ScorePredictor,
        dr,
        draw_posed_3d_box,
        draw_xyz_axis,
        set_seed,
        trimesh,
    )

    set_seed(0)

    mesh = trimesh.load(args.mesh_file)
    if getattr(args, "mesh_scale", 1.0) != 1.0:
        mesh.apply_scale(float(args.mesh_scale))
    mesh.vertices = mesh.vertices.astype(np.float32)
    try:
        normals = mesh.vertex_normals
        if normals is None or len(normals) != len(mesh.vertices):
            normals = mesh.vertex_normals
        mesh.vertex_normals = np.asarray(normals, dtype=np.float32)
    except Exception:
        mesh.vertex_normals = np.zeros_like(mesh.vertices, dtype=np.float32)

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    extents = np.asarray(extents, dtype=np.float32)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()

    debug_dir = os.path.join(code_dir, "debug_pose_est")
    os.makedirs(debug_dir, exist_ok=True)

    est = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug_dir=debug_dir,
        debug=0,
        glctx=glctx,
    )

    return est, to_origin, bbox, draw_posed_3d_box, draw_xyz_axis

def _make_grid_2x2(top_left, top_right, bottom_left, bottom_right, gap: int = 10):
    """2x2 grid img"""
    h, w = top_left.shape[:2]
    gap_v = np.full((h, gap, 3), 255, dtype=top_left.dtype)
    gap_h = np.full((gap, w * 2 + gap, 3), 255, dtype=top_left.dtype)
    top = np.hstack([top_left, gap_v, top_right])
    bottom = np.hstack([bottom_left, gap_v, bottom_right])
    return np.vstack([top, gap_h, bottom])

def _label_panel(img: np.ndarray, text: str) -> np.ndarray:
    """img title"""
    labeled = img.copy()
    cv2.putText(
        labeled,
        text,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return labeled

def _draw_pose_values(img: np.ndarray, pose_vals: dict) -> np.ndarray:
    """img bottom-right pose values"""
    labeled = img.copy()

    lines = [
        # "Center point",
        "(meter)",
        f"x={pose_vals['x']:.2f}",
        f"y={pose_vals['y']:.2f}",
        f"z={pose_vals['z']:.2f}",
        # "Rotation vector",
        "(radian)",
        f"rx={pose_vals['roll']:.4f}",
        f"ry={pose_vals['pitch']:.4f}",
        f"rz={pose_vals['yaw']:.4f}",
    ]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9
    thickness = 2
    color = (255, 0, 0)

    sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in lines]
    max_h = max(h for _, h in sizes)

    h, w = labeled.shape[:2]

    margin = 10
    line_gap = 6

    total_height = len(lines) * max_h + (len(lines) - 1) * line_gap

    y = h - total_height - margin + max_h  

    for line, (line_w, _line_h) in zip(lines, sizes):
        x = w - line_w - margin
        cv2.putText(labeled, line, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
        y += max_h + line_gap

    return labeled


def _overlay_rgba(dst: np.ndarray, src: np.ndarray, x: int, y: int) -> None:
    """img overlay"""
    if src is None or src.size == 0:
        return
    h, w = src.shape[:2]
    H, W = dst.shape[:2]
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(W, x + w)
    y1 = min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    src_roi = src[(y0 - y):(y1 - y), (x0 - x):(x1 - x)]
    dst_roi = dst[y0:y1, x0:x1]
    if src_roi.shape[2] == 4:
        alpha = src_roi[:, :, 3:4].astype(np.float32) / 255.0
        dst_roi[:] = (alpha * src_roi[:, :, :3] + (1.0 - alpha) * dst_roi).astype(np.uint8)
    else:
        dst_roi[:] = src_roi[:, :, :3]


def _add_ui_border(combo: np.ndarray, pad_y: int = 100, pad_x: int = 100, scale: float = 1.0) -> np.ndarray:
    """pixel padding, text."""
    canvas = cv2.copyMakeBorder(combo, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    new_w = int(canvas.shape[1] * scale)
    new_h = int(canvas.shape[0] * scale)
    canvas = cv2.resize(canvas, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    splash = cv2.imread("./splash_cropped.png", cv2.IMREAD_UNCHANGED)
    x_offset = int(pad_x * scale)
    y_offset = int(pad_y * scale * 0.5)
    if splash is not None:
        target_h = max(1, int(pad_y * scale * 0.5))  # 0.5 = splash ratio
        scale_factor = target_h / float(splash.shape[0])
        target_w = max(1, int(splash.shape[1] * scale_factor))
        splash_rs = cv2.resize(splash, (target_w, target_h), interpolation=cv2.INTER_AREA)
        _overlay_rgba(canvas, splash_rs, x_offset, max(0, y_offset - target_h // 2))
    else:
        mr_text = "MakinaRocks"
        mr_font = cv2.FONT_HERSHEY_SIMPLEX
        mr_scale = 1.9
        mr_thickness = 4
        mr_color = (215, 119, 107)  # BGR (RGB 107,119,215)
        (mr_w, mr_h), _ = cv2.getTextSize(mr_text, mr_font, mr_scale, mr_thickness)
        mr_y = int(pad_y * scale * 0.5) + mr_h // 2  # 
        cv2.putText(canvas, mr_text, (x_offset, mr_y),
                    mr_font, mr_scale, mr_color, mr_thickness, cv2.LINE_AA)

    title_text = "[Smart Welding Automation System]"
    title_font = cv2.FONT_HERSHEY_SIMPLEX
    title_scale = 0.9
    title_thickness = 2
    title_color = (215, 119, 107)  # BGR (RGB 107,119,215)
    (ttw, tth), _ = cv2.getTextSize(title_text, title_font, title_scale, title_thickness)
    title_x = (canvas.shape[1] - ttw - pad_x)
    title_y = int(pad_y * scale * 0.5) + tth // 2
    cv2.putText(canvas, title_text, (title_x, title_y),
                title_font, title_scale, title_color, title_thickness, cv2.LINE_AA)

    return canvas


def run_pose_estimation(
    cam,
    *,
    mesh_scale: float = 1.0,
    fx: float = 0.0,
    fy: float = 0.0,
    cx: float = -1.0,
    cy: float = -1.0,
    out_width: int = 640,
    out_height: int = 512,
    depth_scale: float = 0.001,
    window_name: str = "MakinaRocks",
    est_refine_iter: int = 5,
    # on_pose=None,
    return_on_estimate: bool = True,
) -> dict | None:
    args = SimpleNamespace(
        mesh_file=None,
        mesh_scale=mesh_scale,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        est_refine_iter=est_refine_iter,
    )

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    pad_x = 426
    pad_y = 154
    _ui_w = int((out_width * 2 + pad_x * 2) * 1.00)  #
    _ui_h = int((out_height * 2 + pad_y * 2) * 1.00)  # ui size setting
    cv2.resizeWindow(window_name, _ui_w, _ui_h)
    
    K = None
    base_rgb = None
    base_depth_m = None
    pose = None
    last_pose_6d = None
    robot_ctrl = RTDEControlInterface(UR5E_IP)
    robot_recv = RTDEReceiveInterface(UR5E_IP)
    keepalive = RTDEKeepAliveThread(UR5E_IP, interval=1.0)
    keepalive.start()


    yolo_model = YOLO("./best.pt")

    CLASS_TO_MESH = {
        "a": "./model_ex1.obj",
        "b": "./model_ex2.obj",
    }

    # capture pose move
    robot_ctrl.moveL(
        [0.407480,-0.049565,0.234639,1.782498,-1.762627,-0.709733],
        speed=0.05, acceleration=0.05,)

    try:
        while True:

            # 1. 캡처
            print("Capturing frame...")
            time.sleep(3.0)  
            rgb, depth, _ = cam.capture_textured_point_cloud()
            if rgb is None:
                print("Capture failed, retrying...")
                time.sleep(0.5)
                continue

            if K is None:
                cap_h, cap_w = rgb.shape[:2]
                K_native = _load_K(args, cap_w, cap_h)
                sx = float(out_width) / float(cap_w)
                sy = float(out_height) / float(cap_h)
                K = _scale_K(K_native, sx=sx, sy=sy)
                print("Camera K initialized:\n", K)

            rgb_rs = cv2.resize(rgb, (out_width, out_height))

            if depth is not None:
                depth_rs = cv2.resize(depth, (out_width, out_height), interpolation=cv2.INTER_NEAREST)
                depth_m = depth_rs.astype(np.float32) * depth_scale
                depth_u8 = _normalize_depth_for_view(depth_m)
                depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)
                invalid = ~np.isfinite(depth_m) | (depth_m <= 0)
                depth_color[invalid] = [0, 0, 0]
            else:
                depth_m = None
                depth_color = np.zeros((out_height, out_width, 3), dtype=np.uint8)

            base_rgb = rgb_rs.copy()
            base_depth_m = depth_m.copy() if depth_m is not None else None

            vis_rgb_bgr = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2BGR)
            panel_rgb = _label_panel(vis_rgb_bgr, "RGB")
            panel_depth = _label_panel(depth_color, "Depth")
            panel_detect = _label_panel(np.zeros_like(vis_rgb_bgr), "Detection")
            panel_pose = _label_panel(np.zeros_like(vis_rgb_bgr), "Pose Estimation")
            combo = _make_grid_2x2(panel_rgb, panel_depth, panel_detect, panel_pose)
            cv2.imshow(window_name, _add_ui_border(combo, pad_y=pad_y, pad_x=pad_x))
            cv2.waitKey(1)

            # 2. YOLO segmentation
            print("YOLO segmentation")
            results = yolo_model.predict(
                base_rgb,
                conf=0.5,
                imgsz=1024,
                verbose=False,
            )
            r = results[0]

            if r.boxes is None or r.masks is None:
                print("detection failed - retrying...")
                time.sleep(3.0)
                continue

            num_objects = len(r.boxes)
            if num_objects != 1:
                print(f"detection failed - {num_objects} objects detected - retrying...")
                time.sleep(3.0)
                continue

            box = r.boxes[0]
            cls_id = int(box.cls.item())
            detected_class = yolo_model.names[cls_id]
            conf = float(box.conf.item())

            mask = r.masks.data[0].cpu().numpy()
            h, w = base_rgb.shape[:2]
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            detected_mask_bool = mask > 0.5

            print(f"YOLO class={detected_class} conf={conf:.3f}")

            if detected_class not in CLASS_TO_MESH:
                print(f"mesh mapping not found: {detected_class} - retrying...")
                time.sleep(3.0)
                continue

            args.mesh_file = CLASS_TO_MESH[detected_class]

            ys, xs = np.where(detected_mask_bool)
            if len(xs) == 0:
                print("mask area not found - retrying...")
                time.sleep(3.0)
                continue

            pad = 10
            x0 = max(0, xs.min() - pad)
            y0 = max(0, ys.min() - pad)
            x1 = min(w - 1, xs.max() + pad)
            y1 = min(h - 1, ys.max() + pad)

            roi_mask = np.zeros_like(detected_mask_bool, dtype=bool)
            roi_mask[y0:y1 + 1, x0:x1 + 1] = True

            vis_rgb = base_rgb.copy()
            color_overlay = np.zeros_like(vis_rgb)
            color_overlay[detected_mask_bool] = (255, 0, 0)
            vis_rgb = cv2.addWeighted(vis_rgb, 1.0, color_overlay, 0.5, 0)
            cv2.rectangle(vis_rgb, (x0, y0), (x1, y1), (0, 255, 0), 2)

            CLASS_TO_LABEL = {"a": "Cylinder", "b": "U-channel"}
            label_text = CLASS_TO_LABEL.get(detected_class, detected_class)
            cx_box = (x0 + x1) // 2
            cy_box = (y0 + y1) // 2
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 1
            (tw, th), _ = cv2.getTextSize(label_text, font, font_scale, thickness)
            cv2.putText(vis_rgb, label_text, (cx_box - tw // 2, cy_box + th // 2),
                        font, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)

            vis_bgr = cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR)
            panel_detect = _label_panel(vis_bgr, "Detection")
            combo = _make_grid_2x2(panel_rgb, panel_depth, panel_detect, panel_pose)
            cv2.imshow(window_name, _add_ui_border(combo, pad_y=pad_y, pad_x=pad_x))
            cv2.waitKey(1)

            # 3. Pose estimation
            print("Initializing FoundationPose...")
            est, to_origin, bbox, draw_posed_3d_box, draw_xyz_axis = _setup_foundationpose(args)

            print("Pose estimation running...")
            pose = est.register(
                K=K,
                rgb=base_rgb,
                depth=base_depth_m,
                ob_mask=roi_mask,
                iteration=int(est_refine_iter),
            )

            if pose is None:
                print("Pose estimation failed - retrying...")
                time.sleep(3.0)
                continue

            center_pose = pose @ np.linalg.inv(to_origin)

            t = center_pose[:3, 3]
            roll, pitch, yaw = _rot_to_rpy(center_pose[:3, :3])
            last_pose_6d = {
                "x": float(t[0]),
                "y": float(t[1]),
                "z": float(t[2]),
                "roll": roll,
                "pitch": pitch,
                "yaw": yaw,
            }

            vis_rgb_pose = draw_posed_3d_box(K, img=base_rgb, ob_in_cam=center_pose, bbox=bbox)
            vis_rgb_pose = draw_xyz_axis(
                vis_rgb_pose,
                ob_in_cam=center_pose,
                scale=0.1,
                K=K,
                thickness=3,
                transparency=0,
                is_input_rgb=True,
            )
            vis_bgr_pose = cv2.cvtColor(vis_rgb_pose, cv2.COLOR_RGB2BGR)
            panel_pose = _label_panel(vis_bgr_pose, "Pose Estimation")
            panel_pose = _draw_pose_values(panel_pose, last_pose_6d)
            combo = _make_grid_2x2(panel_rgb, panel_depth, panel_detect, panel_pose)
            cv2.imshow(window_name, _add_ui_border(combo, pad_y=pad_y, pad_x=pad_x))
            cv2.waitKey(1)
            tcp_pose = robot_recv.getActualTCPPose()
            cam_pose6 = [last_pose_6d[k] for k in ("x", "y", "z", "roll", "pitch", "yaw")]
            p_base_cur = convert_cam_to_base(cam_pose6[0:3], tcp_pose)
            poses = _load_robot_poses_txt(ROBOT_POSES_TXT)
            cap_pose = poses[0]
            cap_pose = [cap_pose["x"], cap_pose["y"], cap_pose["z"],
                        cap_pose["rx"], cap_pose["ry"], cap_pose["rz"]]
            ready_pose = poses[1]

            # 4. Ready pose로 이동 
            print("Moving robot to ready pose")
            robot_ctrl.moveL(
                [ready_pose["x"], ready_pose["y"], ready_pose["z"],
                 ready_pose["rx"], ready_pose["ry"], ready_pose["rz"]],
                speed=0.05, acceleration=0.05,)
            time.sleep(0.5)

            # ======== 5. 'm' 키 대기 ========
            # print("'m' 키를 누르면 로봇이 작업을 시작합니다 (ESC/q: 종료)")
            # cv2.imshow(window_name, _add_ui_border(combo, pad_y=pad_y, pad_x=pad_x))
            # quit_flag = False
            # while True:
            #     k2 = cv2.waitKey(50) & 0xFF
            #     if k2 == ord("m"):
            #         break
            #     elif k2 in (27, ord("q")):
            #         quit_flag = True
            #         break
            # if quit_flag:
            #     break

            # 5. 로봇 작업 수행 
            if detected_class == "a":
                RADIUS = -0.047
                start_pose = [p_base_cur[0] + RADIUS, p_base_cur[1], 0.0395,
                              ready_pose["rx"], ready_pose["ry"], ready_pose["rz"]] # Z 0.0395
                c_x, c_y, c_z = p_base_cur
                dt = 0.005
                T = 3.0

                # 반시계 방향
                robot_ctrl.moveL(start_pose, speed=0.15, acceleration=0.15)
                time.sleep(0.5)
                final_pose = circle_move_0(c_x, c_y, dt, T, RADIUS, robot_ctrl, start_pose)
                move_offset_0(final_pose, dt, robot_ctrl, robot_recv)

                # 시계 방향
                robot_ctrl.moveL(start_pose, speed=0.15, acceleration=0.15)
                time.sleep(0.5)
                final_pose = circle_move_1(c_x, c_y, dt, T, RADIUS, robot_ctrl, start_pose)
                move_offset_1(final_pose, dt, robot_ctrl, robot_recv)

                # 캡처 포즈로 복귀
                robot_ctrl.moveL(cap_pose, 0.10, 0.10)

            elif detected_class == "b":
                cam_pose6_T = cam_pose6_to_T(cam_pose6)
                R = cam_pose6_T[:3, :3]
                t = cam_pose6_T[:3, 3]

                half_w = 0.075 / 2.0
                half_h = 0.200 / 2.0

                corners_cam = [
                    t + np.array([-half_w, -half_h, 0.0], dtype=np.float32),
                    t + np.array([ half_w, -half_h, 0.0], dtype=np.float32),
                    t + np.array([ half_w,  half_h, 0.0], dtype=np.float32),
                    t + np.array([-half_w,  half_h, 0.0], dtype=np.float32),
                ]

                # point rotation
                axis_y = R[:, 1]
                scale = 0.1
                p0_3d = t
                p1_3d = t + scale * axis_y

                p0 = K @ p0_3d
                p1 = K @ p1_3d
                u0, v0 = p0[0] / p0[2], p0[1] / p0[2]
                u1, v1 = p1[0] / p1[2], p1[1] / p1[2]
                dx = u1 - u0
                dy = v1 - v0

                angle_rad = math.atan2(dy, dx)
                angle_deg_img = math.degrees(angle_rad)

                rot_rad = math.radians(angle_deg_img)
                cos_a = math.cos(rot_rad)
                sin_a = math.sin(rot_rad)

                rotated_corners_cam = []
                for corner in corners_cam:
                    dx = corner[0] - t[0]
                    dy = corner[1] - t[1]
                    rx = cos_a * dx - sin_a * dy + t[0]
                    ry = sin_a * dx + cos_a * dy + t[1]
                    rotated_corners_cam.append(np.array([rx, ry, corner[2]], dtype=np.float32))

                rotated_corners_base = []
                for rc in rotated_corners_cam:
                    p_base = convert_cam_to_base(rc, tcp_pose)
                    rotated_corners_base.append(p_base)

                rotated_corners_base.sort(key=lambda p: p[0])

                center_base = p_base_cur
                edge_pairs = [(0, 1), (1, 3), (0, 2), (2, 3)]
                edge_rvs = []
                for idx, (a, b) in enumerate(edge_pairs):
                    p0 = rotated_corners_base[a]
                    p1 = rotated_corners_base[b]
                    mid = [(p0[0] + p1[0]) / 2.0,
                           (p0[1] + p1[1]) / 2.0,
                           0.080]
                    dx_m = mid[0] - center_base[0]
                    dy_m = mid[1] - center_base[1]
                    virtual_cur = [center_base[0] + dy_m,
                                   center_base[1] - dx_m,
                                   0.080]
                    rv = compute_rotvec_lookat_vertical(virtual_cur, [center_base[0], center_base[1], 0.080])
                    edge_rvs.append(rv)

                OFFSET_M = 0.005
                center_xy = np.array([center_base[0], center_base[1]])
                offset_paths = []
                for idx, (a, b) in enumerate(edge_pairs):
                    pa = np.array(rotated_corners_base[a][:2])
                    pb = np.array(rotated_corners_base[b][:2])
                    edge_dir = pb - pa
                    normal = np.array([-edge_dir[1], edge_dir[0]])
                    normal = normal / (np.linalg.norm(normal) + 1e-12)
                    mid_xy = (pa + pb) / 2.0
                    if np.dot(normal, mid_xy - center_xy) < 0:
                        normal = -normal
                    pa_off = pa + OFFSET_M * normal
                    pb_off = pb + OFFSET_M * normal
                    offset_paths.append((pa_off, pb_off))
                    # print(f"path {a}-{b}: start=({pa_off[0]:.5f}, {pa_off[1]:.5f}) end=({pb_off[0]:.5f}, {pb_off[1]:.5f})")

                Z_HEIGHT = 0.0395 # Z 0.0395
                full_paths = []
                for idx, ((pa_off, pb_off), rv) in enumerate(zip(offset_paths, edge_rvs)):
                    start_6d = [pa_off[0], pa_off[1], Z_HEIGHT, rv[0], rv[1], rv[2]]
                    end_6d   = [pb_off[0], pb_off[1], Z_HEIGHT, rv[0], rv[1], rv[2]]
                    full_paths.append((start_6d, end_6d))
                    # print(f"path {idx}: start={[f'{v:.5f}' for v in start_6d]} end={[f'{v:.5f}' for v in end_6d]}")

                # path 0 이동
                robot_ctrl.moveL(full_paths[0][0], speed=0.10, acceleration=0.10)
                time.sleep(0.5)
                robot_ctrl.moveL(full_paths[0][1], speed=0.10, acceleration=0.10)
                time.sleep(0.5)

                # path 1 이동
                robot_ctrl.moveL(full_paths[1][0], speed=0.10, acceleration=0.10)
                time.sleep(0.5)
                robot_ctrl.moveL(full_paths[1][1], speed=0.10, acceleration=0.10)
                time.sleep(0.5)

                # offset 이동
                cur_pose = robot_recv.getActualTCPPose()
                robot_ctrl.moveL([cur_pose[0], cur_pose[1], cur_pose[2]+0.05, cur_pose[3], cur_pose[4], cur_pose[5]], speed=0.25, acceleration=0.25)
                time.sleep(0.5)
                cur_pose = robot_recv.getActualTCPPose()
                robot_ctrl.moveL([cur_pose[0]-0.1, cur_pose[1], cur_pose[2], full_paths[0][1][3], full_paths[0][1][4], full_paths[0][1][5]], speed=0.25, acceleration=0.25)
                time.sleep(0.5)
                robot_ctrl.moveL(
                    [ready_pose["x"], ready_pose["y"], ready_pose["z"],
                     ready_pose["rx"], ready_pose["ry"], ready_pose["rz"]],
                    speed=0.25, acceleration=0.25,)
                time.sleep(0.5)

                # path 2 이동
                robot_ctrl.moveL(full_paths[2][0], speed=0.10, acceleration=0.10)
                time.sleep(0.5)
                robot_ctrl.moveL(full_paths[2][1], speed=0.10, acceleration=0.10)
                time.sleep(0.5)

                # path 3 이동
                robot_ctrl.moveL(full_paths[3][0], speed=0.10, acceleration=0.10)
                time.sleep(0.5)
                robot_ctrl.moveL(full_paths[3][1], speed=0.10, acceleration=0.10)
                time.sleep(0.5)

                # offset 이동
                cur_pose = robot_recv.getActualTCPPose()
                robot_ctrl.moveL([cur_pose[0], cur_pose[1], cur_pose[2]+0.05, cur_pose[3], cur_pose[4], cur_pose[5]], speed=0.25, acceleration=0.25)
                time.sleep(0.5)
                cur_pose = robot_recv.getActualTCPPose()
                robot_ctrl.moveL([cur_pose[0]-0.1, cur_pose[1], cur_pose[2], full_paths[2][1][3], full_paths[2][1][4], full_paths[2][1][5]], speed=0.25, acceleration=0.25)
                time.sleep(0.5)
                robot_ctrl.moveL(
                    [ready_pose["x"], ready_pose["y"], ready_pose["z"],
                     ready_pose["rx"], ready_pose["ry"], ready_pose["rz"]],
                    speed=0.25, acceleration=0.25,)
                time.sleep(0.5)

                robot_ctrl.moveL(cap_pose, 0.10, 0.10)

            print("Task completed - retrying capture")
            time.sleep(1.0)

    finally:
        cv2.destroyAllWindows()
        keepalive.stop()

    return last_pose_6d
