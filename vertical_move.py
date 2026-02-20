from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
import time
import math
import threading
# tcp의 orientation이 지면과 수직이 되도록함

ROBOT_IP = "192.168.2.100"   # 🔴 UR 로봇 IP

rtde_c = RTDEControlInterface(ROBOT_IP)
rtde_r = RTDEReceiveInterface(ROBOT_IP)

class RTDEKeepAliveThread(threading.Thread):
    def __init__(self, rtde_recv, interval=1.0):
        super().__init__(daemon=True)
        self.interval = interval
        self.running = True
        self.rtde_r = rtde_recv   # ⭐ 기존 객체 공유

    def run(self):
        print("[RTDE KeepAlive] started")
        while self.running:
            try:
                tcp = self.rtde_r.getActualTCPPose()
            except Exception as e:
                print("[RTDE KeepAlive ERROR]", e)

            time.sleep(self.interval)

    def stop(self):
        self.running = False

keepalive = RTDEKeepAliveThread(rtde_r, interval=0.1)
keepalive.start()

rtde_c.moveL([0.455189,-0.090116,0.056210,2.337283,-2.094930,-0.075545], 
             0.1, 0.1)

center = rtde_r.getActualTCPPose()
cx, cy, cz, rx, ry, rz = center

RADIUS = -0.03

start_circle_pose = [
    cx + RADIUS,
    cy,
    cz,
    rx, ry, rz
]

print("Move to circle start point...")
rtde_c.moveL(start_circle_pose, 0.05, 0.05)
time.sleep(0.5)

# -------------------------
# servoL 원운동
# -------------------------
print("Start smooth circular motion")

dt = 0.01
T = 6.0
steps = int(T / dt)

center_minus_r = [cx, cy, cz - RADIUS]


import numpy as np
import cv2

def rotvec_lookat(cur_xyz, target_xyz, up=np.array([0,0,1.0])):
    cur_xyz = np.array(cur_xyz, dtype=float)
    target_xyz = np.array(target_xyz, dtype=float)
    up = np.array(up, dtype=float)

    z = cur_xyz - target_xyz
    z = z / (np.linalg.norm(z) + 1e-12)

    if abs(np.dot(z, up) / (np.linalg.norm(up)+1e-12)) > 0.999:
        up = np.array([0,1.0,0])

    x = np.cross(up, z)
    x = x / (np.linalg.norm(x) + 1e-12)

    y = np.cross(z, x)

    R = np.column_stack([x, y, z])

    rotvec, _ = cv2.Rodrigues(R)
    return rotvec.reshape(3)

for i in range(steps + 1):

    theta = math.pi * i / steps

    x = cx + RADIUS * math.cos(theta)
    y = cy + RADIUS * math.sin(theta)
    z = cz

    cur_xyz = [x, y, z]
    target = center_minus_r
    rv = rotvec_lookat(cur_xyz, target, up=np.array([0,0,1.0]))
    pose = [x, y, z, rv[0], rv[1], rv[2]]
    # pose = [x, y, z, rotvec[0], rotvec[1], rotvec[2]]

    print(pose)
    # 출력되는 포즈가 수직으로 세우는거임
    exit()

    # pose = [x, y, z, rx, ry, rz]

    # rtde_c.servoL(
    #     pose,
    #     0.1,
    #     0.1,
    #     dt,
    #     0.1,     # 좀 더 크게
    #     100       # gain 줄임
    # )

    time.sleep(dt)

exit()
final_pose = pose  # 위 for 루프의 마지막 원 위치

for _ in range(50): 
    rtde_c.servoL(final_pose, 0.03, 0.03, dt, 0.15, 100)
    time.sleep(dt)

rtde_c.servoStop(0.3)

print("Circular motion finished.")

keepalive.stop()
