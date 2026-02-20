import os
import sys
import time
import warnings
from types import SimpleNamespace

import cv2
import numpy as np
from base_coordi import convert_cam_to_base
from circle_move import circle_move_0, move_offset_0, circle_move_1, move_offset_1
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

def _normalize_depth_for_view(depth_m: np.ndarray, d_min_m: float = 0.1, d_max_m: float = 2.0) -> np.ndarray:

    if depth_m is None or depth_m.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    d = depth_m.copy()
    invalid = ~np.isfinite(d) | (d <= 0)
    if np.all(invalid):
        return np.zeros(d.shape, dtype=np.uint8)
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

def compute_rotvec_lookat_vertical(cur, center):
    cur = np.array(cur)
    center = np.array(center)

    dir_vec = center - cur
    dir_vec[2] = 0.0
    dir_vec /= (np.linalg.norm(dir_vec) + 1e-12)

    z = np.array([0.0, 0.0, -1.0])
    x = dir_vec

    y = np.cross(z, x)
    y /= (np.linalg.norm(y) + 1e-12)

    x = np.cross(y, z)

    R = np.column_stack((x, y, z))

    rotvec, _ = cv2.Rodrigues(R)
    return rotvec.flatten()

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
    window_name: str = "Pose Est",
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
        [0.41750, 0.124807, 0.173820, 3.1416, 0.0, 0.0],
        speed=0.15,acceleration=0.15,)
    
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
                    "y: Detection | s: Pose | c: recapture | q: quit | r: ready robot | m: move robot",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )

                cv2.imshow(window_name, combo_idle)

            elif show_mode == "yolo" and combo_yolo is not None:
                cv2.imshow(window_name, combo_yolo)

            elif show_mode == "pose" and combo_pose is not None:
                cv2.imshow(window_name, combo_pose)

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
                    print(p_base_cur)
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
                    speed=0.1, acceleration=0.1,)
                    time.sleep(0.5)
                    RADIUS = -0.047

                    start_pose = [p_base_cur[0] + RADIUS,p_base_cur[1],0.0365,ready_pose["rx"], ready_pose["ry"], ready_pose["rz"]]
                    
                    c_x, c_y, c_z = p_base_cur
                    dt = 0.01
                    T = 5.0
                    print(start_pose)
                    while True:
                        k2 = cv2.waitKey(0) & 0xFF   # 블로킹 대기

                        if k2 == ord("m"):

                            # 반시계 방향
                            robot_ctrl.moveL(start_pose, speed=0.05, acceleration=0.05)
                            time.sleep(0.5)
                            final_pose = circle_move_0(c_x, c_y, dt, T, RADIUS, robot_ctrl, start_pose)
                            move_offset_0(final_pose, dt, robot_ctrl, robot_recv)

                            # 시계 방향
                            robot_ctrl.moveL(start_pose, speed=0.05, acceleration=0.05)
                            time.sleep(0.5)
                            final_pose = circle_move_1(c_x, c_y, dt, T, RADIUS, robot_ctrl, start_pose)
                            move_offset_1(final_pose, dt, robot_ctrl, robot_recv)

                            robot_ctrl.moveL(cap_pose, 0.05, 0.05)

                            break

                        elif k2 == 27:  # ESC
                            print("취소")
                            break
                        
                elif detected_class == "b":
                    print("classb")


    finally:
        cv2.destroyAllWindows()
    
    keepalive.stop()

    return last_pose_6d
