from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
import time
import math
import threading

import numpy as np
import cv2

# 수직 테스트
# rv = np.array([2.337283, -2.094930, -0.075545], float)

# # 현재 상태
# R, _ = cv2.Rodrigues(rv)
# z = R[:, 2]
# tilt_deg = math.degrees(math.acos(np.clip(-z[2], -1.0, 1.0)))

# print("===== 현재 상태 =====")
# print("rotvec:", rv)
# print("Z_tcp:", z)
# print("tilt(deg):", tilt_deg)
# exit()

ROBOT_IP = "192.168.2.100"

rtde_c = RTDEControlInterface(ROBOT_IP)
rtde_r = RTDEReceiveInterface(ROBOT_IP)

class RTDEKeepAliveThread(threading.Thread):
    def __init__(self, rtde_recv, interval=1.0):
        super().__init__(daemon=True)
        self.interval = interval
        self.running = True
        self.rtde_r = rtde_recv

    def run(self):
        print("[RTDE KeepAlive] started")
        while self.running:
            try:
                _ = self.rtde_r.getActualTCPPose()
            except Exception as e:
                print("[RTDE KeepAlive ERROR]", e)
            time.sleep(self.interval)

    def stop(self):
        self.running = False

keepalive = RTDEKeepAliveThread(rtde_r, interval=0.1)
keepalive.start()

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

rtde_c.moveL([0.440,-0.080,0.060,2.221441, -2.221441, 0.0], 
             0.03, 0.3)

center = rtde_r.getActualTCPPose()
cx, cy, cz, rx, ry, rz = center

RADIUS = -0.04455

start_circle_pose = [
    cx + RADIUS,
    cy,
    cz,
    rx, ry, rz
]

rtde_c.moveL(start_circle_pose, 0.03, 0.03)
time.sleep(0.5)

dt = 0.01
T = 8.0
steps = int(T / dt)

for i in range(steps + 1):

    theta = math.pi * i / steps # 시계 반대 방향
    # theta = -math.pi * i / steps # 시계 방향

    x = cx + RADIUS * math.cos(theta)
    y = cy + RADIUS * math.sin(theta)
    z = cz

    theta_ro = theta - math.pi / 2
    x_ro = cx + RADIUS * math.cos(theta_ro)
    y_ro = cy + RADIUS * math.sin(theta_ro)

    rv = compute_rotvec_lookat_vertical(
        [x_ro, y_ro, z],
        [cx, cy, cz]
    )

    pose = [x, y, z, rv[0], rv[1], rv[2]]

    rtde_c.servoL(
        pose,
        0.1,
        0.1,
        dt,
        0.1,    
        100       
    )

    time.sleep(dt)

final_pose = pose  

for _ in range(50): 
    rtde_c.servoL(final_pose, 0.03, 0.03, dt, 0.15, 100)
    time.sleep(dt)

rtde_c.servoStop(0.3)
time.sleep(0.5)

cur_pose1 = rtde_r.getActualTCPPose()
offset_pose1 = [cur_pose1[0] + 0.005,cur_pose1[1],cur_pose1[2] + 0.06,cur_pose1[3],cur_pose1[4],cur_pose1[5]]
rtde_c.moveL(offset_pose1, 0.07, 0.07)
time.sleep(0.3)

cur_pose2 = rtde_r.getActualTCPPose()
offset_pose2 = [cur_pose2[0]-0.04955,cur_pose2[1],cur_pose2[2],3.14,0,cur_pose2[5]]
rtde_c.moveL(offset_pose2, 0.07, 0.07)
time.sleep(0.3)

cur_pose3 = rtde_r.getActualTCPPose()
offset_pose3 = [cur_pose3[0]-0.04955,cur_pose3[1],cur_pose3[2],2.221,-2.221,cur_pose3[5]]
rtde_c.moveL(offset_pose3, 0.07, 0.07)
time.sleep(0.3)


rtde_c.moveL(start_circle_pose, 0.03, 0.03)
time.sleep(0.5)

dt = 0.01
T = 8.0
steps = int(T / dt)

for i in range(steps + 1):

    # theta = math.pi * i / steps # 시계 반대 방향
    theta = -math.pi * i / steps # 시계 방향

    x = cx + RADIUS * math.cos(theta)
    y = cy + RADIUS * math.sin(theta)
    z = cz

    theta_ro = theta - math.pi / 2
    x_ro = cx + RADIUS * math.cos(theta_ro)
    y_ro = cy + RADIUS * math.sin(theta_ro)

    rv = compute_rotvec_lookat_vertical(
        [x_ro, y_ro, z],
        [cx, cy, cz]
    )

    pose = [x, y, z, rv[0], rv[1], rv[2]]
    # print(pose)

    rtde_c.servoL(
        pose,
        0.1,
        0.1,
        dt,
        0.1,    
        100       
    )

    time.sleep(dt)

final_pose = pose  

for _ in range(50): 
    rtde_c.servoL(final_pose, 0.03, 0.03, dt, 0.15, 100)
    time.sleep(dt)

rtde_c.servoStop(0.3)
time.sleep(0.5)

cur_pose1 = rtde_r.getActualTCPPose()
offset_pose1 = [cur_pose1[0] + 0.005,cur_pose1[1],cur_pose1[2] + 0.03,cur_pose1[3],cur_pose1[4],cur_pose1[5]]
rtde_c.moveL(offset_pose1, 0.05, 0.05)
time.sleep(0.3)

cur_pose2 = rtde_r.getActualTCPPose()
offset_pose2 = [cur_pose2[0]-0.04955,cur_pose2[1],cur_pose2[2],0,3.14,cur_pose2[5]]
rtde_c.moveL(offset_pose2, 0.05, 0.05)
time.sleep(0.3)

cur_pose3 = rtde_r.getActualTCPPose()
offset_pose3 = [cur_pose3[0]-0.04955,cur_pose3[1],cur_pose3[2],2.221,-2.221,cur_pose3[5]]
rtde_c.moveL(offset_pose3, 0.05, 0.05)
time.sleep(0.3)
