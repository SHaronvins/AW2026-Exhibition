import math
import time
import cv2
import numpy as np

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

def circle_move_0(c_x, c_y, dt, T, RADIUS, robot_ctrl, start_pose):
    steps = int(T / dt)
    for i in range(steps + 1):
        theta = math.pi * i / steps

        rotation_x = c_x + RADIUS * math.cos(theta)
        rotation_y = c_y + RADIUS * math.sin(theta)
        rotation_z = start_pose[2]

        theta_ro = theta - math.pi / 2
        x_ro = c_x + RADIUS * math.cos(theta_ro)
        y_ro = c_y + RADIUS * math.sin(theta_ro)
        rv = compute_rotvec_lookat_vertical(
        [x_ro, y_ro, rotation_z],
        [c_x, c_y, rotation_z])

        pose = [rotation_x, rotation_y, rotation_z, 
                rv[0], rv[1], rv[2]]
        # print(pose)

        robot_ctrl.servoL(
            pose,
            0.1,
            0.1,
            dt,
            0.1,    
            150       
        )

        time.sleep(dt)

    final_pose = pose 
    
    return final_pose

def move_offset_0(final_pose, dt, robot_ctrl, robot_recv):
    for _ in range(50): 
        robot_ctrl.servoL(final_pose, 0.03, 0.03, dt, 0.15, 100)
        time.sleep(dt)

    robot_ctrl.servoStop(0.3)
    time.sleep(0.5)

    cur_pose1 = robot_recv.getActualTCPPose()
    offset_pose1 = [cur_pose1[0] + 0.005,cur_pose1[1],cur_pose1[2] + 0.06,cur_pose1[3],cur_pose1[4],cur_pose1[5]]
    robot_ctrl.moveL(offset_pose1, 0.20, 0.20)
    time.sleep(0.3)

    cur_pose2 = robot_recv.getActualTCPPose()
    offset_pose2 = [cur_pose2[0]-0.055,cur_pose2[1],cur_pose2[2],3.14,0,cur_pose2[5]]
    robot_ctrl.moveL(offset_pose2, 0.20, 0.20)
    time.sleep(0.3)

    cur_pose3 = robot_recv.getActualTCPPose()
    offset_pose3 = [cur_pose3[0]-0.055,cur_pose3[1],cur_pose3[2],2.221,-2.221,cur_pose3[5]]
    robot_ctrl.moveL(offset_pose3, 0.20, 0.20)
    time.sleep(0.3)

def circle_move_1(c_x, c_y, dt, T, RADIUS, robot_ctrl, start_pose):
    steps = int(T / dt)
    for i in range(steps + 1):
        theta = -math.pi * i / steps

        rotation_x = c_x + RADIUS * math.cos(theta)
        rotation_y = c_y + RADIUS * math.sin(theta)
        rotation_z = start_pose[2]

        theta_ro = theta - math.pi / 2
        x_ro = c_x + RADIUS * math.cos(theta_ro)
        y_ro = c_y + RADIUS * math.sin(theta_ro)
        rv = compute_rotvec_lookat_vertical(
        [x_ro, y_ro, rotation_z],
        [c_x, c_y, rotation_z])

        pose = [rotation_x, rotation_y, rotation_z, 
                rv[0], rv[1], rv[2]]
        # print(pose)

        robot_ctrl.servoL(
            pose,
            0.1,
            0.1,
            dt,
            0.1,    
            150       
        )

        time.sleep(dt)
    
    final_pose = pose 
    return final_pose

def move_offset_1(final_pose, dt, robot_ctrl, robot_recv):
    for _ in range(50): 
        robot_ctrl.servoL(final_pose, 0.03, 0.03, dt, 0.15, 100)
        time.sleep(dt)

    robot_ctrl.servoStop(0.3)
    time.sleep(0.5)

    cur_pose4 = robot_recv.getActualTCPPose()
    offset_pose4 = [cur_pose4[0] + 0.005,cur_pose4[1],cur_pose4[2] + 0.06,cur_pose4[3],cur_pose4[4],cur_pose4[5]]
    robot_ctrl.moveL(offset_pose4, 0.20, 0.20)
    time.sleep(0.3)

    cur_pose5 = robot_recv.getActualTCPPose()
    offset_pose5 = [cur_pose5[0]-0.055,cur_pose5[1],cur_pose5[2],0,3.14,cur_pose5[5]]
    robot_ctrl.moveL(offset_pose5, 0.20, 0.20)
    time.sleep(0.3)

    cur_pose6 = robot_recv.getActualTCPPose()
    offset_pose6 = [cur_pose6[0]-0.055,cur_pose6[1],cur_pose6[2],2.221,-2.221,cur_pose6[5]]
    robot_ctrl.moveL(offset_pose6, 0.20, 0.20)
    time.sleep(0.3)