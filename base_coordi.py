# pose_cal.py
import re
import numpy as np
from scipy.spatial.transform import Rotation


# ===============================
# Hand–Eye (TCP -> Camera)
# ===============================
T_TCP_CAM = np.array([
    [0.99365086, -0.00336135, -0.11245740, 0.02518853],
    [-0.07998690, 0.68182636, -0.72712785, 0.09027294],
    [0.07912055, 0.73150633, 0.67722849, -0.19702249],
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

# def convert_cam_to_base_4_corners(
#     cam_pose_xyz_list,
#     robot_pose6,
# ):
#     return [convert_cam_to_base(cam_pose_xyz, robot_pose6) for cam_pose_xyz in cam_pose_xyz_list]

# def base_6dof(cam_pose_6dof, robot_pose6):
#     T_base_tcp = robot_pose6_to_T(robot_pose6)
#     T_cam_obj = cam_pose6_to_T(cam_pose_6dof)
#     T_base_obj = T_base_tcp @ T_TCP_CAM @ T_cam_obj
#     # make 6 dof
#     R_base_obj = T_base_obj[:3, :3]
#     p_base = T_base_obj[:3, 3]
#     # 6축
#     return p_base, R_base_obj