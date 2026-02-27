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
# REFERENCE_POSE_TXT = "reference_pose.txt"
ROBOT_POSES_TXT = "robot_poses.txt"

class RTDEKeepAliveThread(threading.Thread):
    # RTDE 연결 유지용 쓰레드. 주기적으로 TCP 포즈를 받아와서 연결이 끊어지는 것을 방지
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

def _add_ui_border(combo: np.ndarray, pad: int = 100, scale: float = 1.0) -> np.ndarray:
    """combo 이미지에 상하좌우 pad 픽셀 흰색 테두리를 추가하고, scale배 확대한 뒤 텍스트를 삽입한다."""
    canvas = cv2.copyMakeBorder(combo, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    new_w = int(canvas.shape[1] * scale)
    new_h = int(canvas.shape[0] * scale)
    canvas = cv2.resize(canvas, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # 좌상단에 MakinaRocks 텍스트 (두껍게)
    mr_text = "MakinaRocks"
    mr_font = cv2.FONT_HERSHEY_SIMPLEX
    mr_scale = 1.5
    mr_thickness = 3
    mr_color = (215, 119, 107)  # BGR (RGB 107,119,215)
    (mr_w, mr_h), _ = cv2.getTextSize(mr_text, mr_font, mr_scale, mr_thickness)
    x_offset = int(pad * scale)
    mr_y = int(pad * scale * 0.5) + mr_h // 2  # 상단 패딩 세로 중앙
    cv2.putText(canvas, mr_text, (x_offset, mr_y),
                mr_font, mr_scale, mr_color, mr_thickness, cv2.LINE_AA)

    # 상측 중앙에 타이틀 텍스트 (MakinaRocks보다 약간 작게, 밑면 맞춤)
    title_text = "Smart Welding Automation System"
    title_font = cv2.FONT_HERSHEY_SIMPLEX
    title_scale = 0.9
    title_thickness = 2
    title_color = (215, 119, 107)  # BGR (RGB 107,119,215)
    (ttw, tth), _ = cv2.getTextSize(title_text, title_font, title_scale, title_thickness)
    title_x = (canvas.shape[1] - ttw) // 2
    title_y = mr_y  # MakinaRocks와 밑면(baseline) 일치
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
    out_width: int = 1280,
    out_height: int = 1024,
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
    # 창 크기를 (이미지 너비 + 패딩) * 2배로 설정
    _ui_w = int((out_width * 2 + 200) * 1.5)  # (RGB + Depth + 좌우패딩) * scale
    _ui_h = int((out_height + 200) * 1.5)      # (높이 + 상하패딩) * scale
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


    yolo_model = YOLO("./yolo/best.pt")

    print("Press 'c' to capture")

    captured = False
    detected_class = None
    detected_mask_bool = None
    roi_mask = None
    depth_color = None
    show_mode = "idle"  
    combo_yolo = None
    combo_pose = None
    est = to_origin = bbox = draw_posed_3d_box = draw_xyz_axis = None

    CLASS_TO_MESH = {
        "a": "./model_ex1.obj",
        "b": "./model_ex2.obj",
    }
    # 로봇 capture pose move
    robot_ctrl.moveL(
        [0.400241,-0.089987,0.236343,1.781257,-1.780893,-0.703689],
        speed=0.05,acceleration=0.05,)
    
    try:
        while True:

            if not captured:

                key = cv2.waitKey(50) & 0xFF

                if key == ord("c"):
                    print("Capturing frame...")

                    rgb, depth, _ = cam.capture_textured_point_cloud()
                    if rgb is None:
                        print("Capture failed")
                        continue

                    # K 세팅
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

                    vis_rgb_bgr = cv2.cvtColor(rgb_rs, cv2.COLOR_RGB2BGR)
                    combo_idle = np.hstack([vis_rgb_bgr, depth_color])

                    base_rgb = rgb_rs.copy()
                    base_depth_m = depth_m.copy() if depth_m is not None else None

                    detected_class = None
                    detected_mask_bool = None
                    roi_mask = None
                    pose = None
                    last_pose_6d = None

                    show_mode = "idle"
                    combo_yolo = None
                    combo_pose = None

                    captured = True
                    continue

                if key in (27, ord("q")):
                    break

                continue

            if show_mode == "idle":
                vis_rgb_bgr = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2BGR)
                combo_idle = np.hstack([vis_rgb_bgr, depth_color])

                cv2.putText(
                    combo_idle,
                    "c: recapture | y: Detection | s: Pose | r: ready robot | m: move robot | q: quit",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )

                cv2.imshow(window_name, _add_ui_border(combo_idle))

            elif show_mode == "yolo" and combo_yolo is not None:
                cv2.imshow(window_name, _add_ui_border(combo_yolo))

            elif show_mode == "pose" and combo_pose is not None:
                cv2.imshow(window_name, _add_ui_border(combo_pose))

            key = cv2.waitKey(50) & 0xFF

            if key in (27, ord("q")):
                break

            if key == ord("c"):
                captured = False
                show_mode = "idle"
                combo_yolo = None
                combo_pose = None
                print("Recapture mode. Press 'c' again to capture.")
                continue

            # 2) y YOLO start
            if key == ord("y"):

                print("YOLO segmentation 실행")

                results = yolo_model.predict(
                    base_rgb,
                    conf=0.5,
                    imgsz=out_width,
                    verbose=False,
                )

                r = results[0]

                if r.boxes is None or r.masks is None:
                    print("YOLO 결과 없음 - segmentation을 다시해야 됩니다")
                    continue

                num_objects = len(r.boxes)

                if num_objects != 1:
                    print(f"YOLO 객체 수가 {num_objects}개 입니다 - segmentation을 다시해야 됩니다")
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
                    print(f"mesh 매핑 없음: {detected_class}")
                    roi_mask = None
                    continue

                args.mesh_file = CLASS_TO_MESH[detected_class]

                ys, xs = np.where(detected_mask_bool)
                if len(xs) == 0:
                    roi_mask = None
                    continue

                pad = 10
                x0 = max(0, xs.min() - pad)
                y0 = max(0, ys.min() - pad)
                x1 = min(w - 1, xs.max() + pad)
                y1 = min(h - 1, ys.max() + pad)

                roi_mask = np.zeros_like(detected_mask_bool, dtype=bool)
                roi_mask[y0:y1 + 1, x0:x1 + 1] = True

                vis_rgb = base_rgb.copy()
                color = np.zeros_like(vis_rgb)
                color[detected_mask_bool] = (255, 0, 0)
                vis_rgb = cv2.addWeighted(vis_rgb, 1.0, color, 0.5, 0)

                cv2.rectangle(vis_rgb, (x0, y0), (x1, y1), (0, 255, 255), 2)

                # 바운딩 박스 중앙에 시료 라벨 표시
                CLASS_TO_LABEL = {"a": "Target 1", "b": "Target 2"}
                label_text = CLASS_TO_LABEL.get(detected_class, detected_class)
                cx_box = (x0 + x1) // 2
                cy_box = (y0 + y1) // 2
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 2
                (tw, th), _ = cv2.getTextSize(label_text, font, font_scale, thickness)
                cv2.putText(vis_rgb, label_text, (cx_box - tw // 2, cy_box + th // 2),
                            font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

                vis_bgr = cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR)
                combo_yolo = np.hstack([vis_bgr, depth_color])

                show_mode = "yolo"
                continue

            # s Pose estimation
            if key == ord("s"):

                if roi_mask is None or args.mesh_file is None:
                    print("먼저 'y'로 YOLO 실행하세요.")
                    continue

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

                center_pose = pose @ np.linalg.inv(to_origin)

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
                combo_pose = np.hstack([vis_bgr_pose, depth_color])
                show_mode = "pose"

                if pose is not None:
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
                    tcp_pose = robot_recv.getActualTCPPose()
                    cam_pose6 = [last_pose_6d[k] for k in ("x", "y", "z", "roll", "pitch", "yaw")]
                
                    p_base_cur = convert_cam_to_base(cam_pose6[0:3],tcp_pose)
                    poses = _load_robot_poses_txt(ROBOT_POSES_TXT)
                    cap_pose = poses[0]
                    cap_pose = [cap_pose["x"], cap_pose["y"], cap_pose["z"],
                                cap_pose["rx"], cap_pose["ry"], cap_pose["rz"]]
                    ready_pose = poses[1]


            if key == ord("r"):
                print("로봇 이동 명령")
                if detected_class == "a":
                    robot_ctrl.moveL(
                    [ready_pose["x"], ready_pose["y"], ready_pose["z"],ready_pose["rx"], ready_pose["ry"], ready_pose["rz"]],
                    speed=0.10, acceleration=0.10,)
                    time.sleep(0.5)
                    RADIUS = -0.047

                    start_pose = [p_base_cur[0] + RADIUS,p_base_cur[1],0.0365,ready_pose["rx"], ready_pose["ry"], ready_pose["rz"]]
                    
                    c_x, c_y, c_z = p_base_cur
                    dt = 0.005
                    T = 3.0
                    # print(start_pose)
                    while True:
                        k2 = cv2.waitKey(0) & 0xFF   # 블로킹 대기

                        if k2 == ord("m"):

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

                            robot_ctrl.moveL(cap_pose, 0.10, 0.10)

                            break

                        elif k2 == 27:  # ESC
                            print("취소")
                            break

                elif detected_class == "b":

                    if pose is None:
                        print("pose가 없습니다. 먼저 's'로 포즈를 추정하세요.")
                        continue

                    robot_ctrl.moveL(
                    [ready_pose["x"], ready_pose["y"], ready_pose["z"],ready_pose["rx"], ready_pose["ry"], ready_pose["rz"]],
                    speed=0.10, acceleration=0.10,)
                    time.sleep(0.5)

                    cam_pose6_T = cam_pose6_to_T(cam_pose6)  #

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

                    # visualize corners
                    # for corner in corners_cam:
                    #     p = K @ corner
                    #     u = int(p[0] / p[2])
                    #     v = int(p[1] / p[2])
                    #     cv2.circle(vis_bgr_pose, (u, v), 5, (0, 0, 255), -1)
                    # cv2.imwrite("estimated_pose.png", vis_bgr_pose)
                    
                    # point rotation
                    axis_y = R[:, 1]           
                    scale = 0.1               

                    p0_3d = t                   
                    p1_3d = t + scale * axis_y  

                    # 3D -> 2D 투영 (동차좌표)
                    p0 = K @ p0_3d
                    p1 = K @ p1_3d

                    u0, v0 = p0[0] / p0[2], p0[1] / p0[2]
                    u1, v1 = p1[0] / p1[2], p1[1] / p1[2]

                    dx = u1 - u0
                    dy = v1 - v0
    
                    angle_rad = math.atan2(dy, dx)
                    angle_deg_img = math.degrees(angle_rad)

                    # print(angle_deg_img)

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

                    # print(corners_cam)
                    # print(rotated_corners_cam)

                    # visualize rotated corners
                    # for rc in rotated_corners_cam:
                    #     p = K @ rc
                    #     u = int(p[0] / p[2])
                    #     v = int(p[1] / p[2])
                    #     cv2.circle(vis_bgr_pose, (u, v), 5, (0, 255, 255), -1)
                    # cv2.imwrite("estimated_pose_rotated.png", vis_bgr_pose)

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
                        # print(f"edge {a}-{b}: rx={rv[0]:.4f}, ry={rv[1]:.4f}, rz={rv[2]:.4f}")

                    # rotated_corners_base, edge_rvs

                    OFFSET_M = 0.004  # 4mm
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
                        print(f"path {a}-{b}: start=({pa_off[0]:.5f}, {pa_off[1]:.5f}) end=({pb_off[0]:.5f}, {pb_off[1]:.5f})")

                    # [x, y, z, rx, ry, rz]
                    Z_HEIGHT = 0.0375
                    full_paths = []  
                    for idx, ((pa_off, pb_off), rv) in enumerate(zip(offset_paths, edge_rvs)):
                        start_6d = [pa_off[0], pa_off[1], Z_HEIGHT, rv[0], rv[1], rv[2]]
                        end_6d   = [pb_off[0], pb_off[1], Z_HEIGHT, rv[0], rv[1], rv[2]]
                        full_paths.append((start_6d, end_6d))
                        print(f"path {idx}: start={[f'{v:.5f}' for v in start_6d]} end={[f'{v:.5f}' for v in end_6d]}")

                    while True:
                        k2 = cv2.waitKey(0) & 0xFF   # 블로킹 대기

                        if k2 == ord("m"):

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
                            robot_ctrl.moveL([cur_pose[0], cur_pose[1], cur_pose[2]+0.05, cur_pose[3], cur_pose[4], cur_pose[5]], speed=0.20, acceleration=0.20)
                            time.sleep(0.5)
                            cur_pose = robot_recv.getActualTCPPose()
                            robot_ctrl.moveL([cur_pose[0]-0.1, cur_pose[1], cur_pose[2], full_paths[0][1][3], full_paths[0][1][4], full_paths[0][1][5]], speed=0.20, acceleration=0.20)
                            time.sleep(0.5)
                            robot_ctrl.moveL(
                            [ready_pose["x"], ready_pose["y"], ready_pose["z"],ready_pose["rx"], ready_pose["ry"], ready_pose["rz"]],
                            speed=0.20, acceleration=0.20,)
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
                            robot_ctrl.moveL([cur_pose[0], cur_pose[1], cur_pose[2]+0.05, cur_pose[3], cur_pose[4], cur_pose[5]], speed=0.20, acceleration=0.20)
                            time.sleep(0.5)
                            cur_pose = robot_recv.getActualTCPPose()
                            robot_ctrl.moveL([cur_pose[0]-0.1, cur_pose[1], cur_pose[2], full_paths[2][1][3], full_paths[2][1][4], full_paths[2][1][5]], speed=0.20, acceleration=0.20)
                            time.sleep(0.5)
                            robot_ctrl.moveL(
                            [ready_pose["x"], ready_pose["y"], ready_pose["z"],ready_pose["rx"], ready_pose["ry"], ready_pose["rz"]],
                            speed=0.20, acceleration=0.20,)
                            time.sleep(0.5)

                            # 캡처 포즈로 복귀
                            robot_ctrl.moveL(cap_pose, 0.10, 0.10)

                            break

                        elif k2 == 27:  # ESC
                            print("취소")
                            break


    finally:
        cv2.destroyAllWindows()
    
    keepalive.stop()

    return last_pose_6d
