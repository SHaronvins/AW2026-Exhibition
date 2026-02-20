# pose_cal.py
import re
import numpy as np
from scipy.spatial.transform import Rotation


# ===============================
# Hand–Eye (TCP -> Camera)
# ===============================
T_TCP_CAM = np.array([
    [0.00926773, -0.99941872, -0.03280759, 0.07539392],
    [0.99285868, 0.01309933, -0.11857510, 0.22510953],
    [0.11893593, -0.03147438, 0.99240295, -0.35042392],
    [0.00000000, 0.00000000, 0.00000000, 1.00000000],
], dtype=np.float64)


# --------------------------------------------------
# txt 파싱
# --------------------------------------------------

def _parse_kv(text: str) -> dict:
    out = {}
    for m in re.finditer(r'([a-zA-Z_]+)\s*=\s*([-+0-9.eE]+)', text):
        out[m.group(1).lower()] = float(m.group(2))
    return out


def load_reference_pose_txt(path: str):
    """
    return:
        cam_pose6_ref  (x,y,z,roll,pitch,yaw)
        rob_pose6_ref  (x,y,z,rx,ry,rz)
    """
    txt = open(path, "r", encoding="utf-8").read()

    parts = re.split(r'(?i)#\s*robot pose', txt)
    cam_txt = parts[0]
    rob_txt = parts[1] if len(parts) > 1 else ""

    cam = _parse_kv(cam_txt)
    rob = _parse_kv(rob_txt)

    # cam_pose6 = np.array([
    #     cam["x"], cam["y"], cam["z"],
    #     cam["roll"], cam["pitch"], cam["yaw"]
    # ], dtype=np.float64)

    cam_pose_xyz = np.array([
        cam["x"], cam["y"], cam["z"]
    ], dtype=np.float64)

    rob_pose6 = np.array([
        rob["x"], rob["y"], rob["z"],
        rob["rx"], rob["ry"], rob["rz"]
    ], dtype=np.float64)

    return cam_pose_xyz, rob_pose6


# --------------------------------------------------
# 6D -> 4x4
# --------------------------------------------------

def cam_pose6_to_T(pose6):
    """ x,y,z,roll,pitch,yaw  -> T_cam_obj """
    x, y, z, r, p, y_ = map(float, pose6)

    R = Rotation.from_euler("xyz", [r, p, y_], degrees=False).as_matrix()

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T


def robot_pose6_to_T(pose6):
    """ x,y,z,rx,ry,rz (rotvec) -> T_base_tcp """
    x, y, z, rx, ry, rz = map(float, pose6)

    R = Rotation.from_rotvec([rx, ry, rz]).as_matrix()

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T


def convert_cam_to_base(
    cam_pose_xyz,
    robot_pose6,
):
    # TCP pose → transformation
    T_base_tcp = robot_pose6_to_T(robot_pose6)

    # cam point (homogeneous)
    p_cam = np.asarray(cam_pose_xyz, dtype=np.float64).reshape(3)
    p_cam_h = np.array([p_cam[0], p_cam[1], p_cam[2], 1.0])

    # base 변환
    p_base_h = T_base_tcp @ T_TCP_CAM @ p_cam_h
    p_base = p_base_h[:3]

    return p_base

# def convert_ref_and_cur_to_base(
#     reference_txt_path: str,
#     cur_cam_pose_xyz,
#     cur_robot_pose6,
# ):

#     cam_pose6_ref = None
#     rob_pose6_ref = None

#     with open(reference_txt_path, "r") as f:
#         for line in f:
#             line = line.strip()

#             # 빈줄 / 주석 스킵
#             if not line or line.startswith("#"):
#                 continue

#             vals = [float(v) for v in line.replace(" ", "").split(",")]

#             if cam_pose6_ref is None:
#                 cam_pose6_ref = vals    
#             else:
#                 rob_pose6_ref = vals
#                 break

#     if cam_pose6_ref is None or rob_pose6_ref is None:
#         raise RuntimeError(f"Reference txt format invalid: {reference_txt_path}")

#     cam_pose_xyz_ref = cam_pose6_ref[:3]
#     T_base_tcp_ref = robot_pose6_to_T(rob_pose6_ref)

#     p_cam_ref = np.asarray(cam_pose_xyz_ref, dtype=np.float64).reshape(3)
#     p_cam_h_ref = np.array(
#         [p_cam_ref[0], p_cam_ref[1], p_cam_ref[2], 1.0],
#         dtype=np.float64,
#     )

#     p_base_ref_h = T_base_tcp_ref @ T_TCP_CAM @ p_cam_h_ref
#     p_base_ref = p_base_ref_h[:3]
#     T_base_tcp_cur = robot_pose6_to_T(cur_robot_pose6)

#     p_cam_cur = np.asarray(cur_cam_pose_xyz, dtype=np.float64).reshape(3)
#     p_cam_h_cur = np.array(
#         [p_cam_cur[0], p_cam_cur[1], p_cam_cur[2], 1.0],
#         dtype=np.float64,
#     )

#     p_base_cur_h = T_base_tcp_cur @ T_TCP_CAM @ p_cam_h_cur
#     p_base_cur = p_base_cur_h[:3]

#     return p_base_ref, p_base_cur
