"""
UR 로봇의 현재 TCP 좌표를 txt 파일로 저장하는 스크립트
실행할 때마다 기존 파일에 추가로 저장됩니다.
"""

from datetime import datetime

try:
    from rtde_receive import RTDEReceiveInterface
except ImportError:
    print("Error: ur_rtde 패키지가 설치되어 있지 않습니다.")
    print("pip install ur_rtde")
    exit(1)



UR5E_IP = "192.168.2.100"
SAVE_FILE = "robot_poses.txt"

import os

if not os.path.exists(SAVE_FILE):
    with open(SAVE_FILE, "w") as f:
        f.write("x,y,z,rx,ry,rz\n")

def save_tcp_pose():
    """현재 TCP 좌표를 파일에 저장"""
    robot_recv = None
    
    try:
        print(f"로봇 연결 중... ({UR5E_IP})")
        robot_recv = RTDEReceiveInterface(UR5E_IP)
        
        # 현재 TCP 좌표 가져오기 (베이스 기준)
        tcp_pose = robot_recv.getActualTCPPose()
        
        x, y, z = tcp_pose[0], tcp_pose[1], tcp_pose[2]
        rx, ry, rz = tcp_pose[3], tcp_pose[4], tcp_pose[5]
        
        # 현재 시간
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 파일에서 기존 포즈 개수 확인
        try:
            with open(SAVE_FILE, 'r') as f:
                lines = f.readlines()
                # 'point_' 로 시작하는 줄 개수 세기
                pose_count = sum(1 for line in lines if line.strip().startswith('point_'))
        except FileNotFoundError:
            pose_count = 0
        
        # 새 포즈 번호
        pose_num = pose_count + 1
        
        # 파일에 추가 (append 모드)
        with open(SAVE_FILE, 'a') as f:
            # f.write(f"# {timestamp}\n")
            f.write(f"{x:.6f},{y:.6f},{z:.6f},{rx:.6f},{ry:.6f},{rz:.6f}\n")
            # f.write("\n")
        
        print(f"\n=== TCP 좌표 저장 완료 (point_{pose_num}) ===")
        print(f"  x  = {x:.6f} m")
        print(f"  y  = {y:.6f} m")
        print(f"  z  = {z:.6f} m")
        print(f"  rx = {rx:.6f} rad")
        print(f"  ry = {ry:.6f} rad")
        print(f"  rz = {rz:.6f} rad")
        print(f"\n저장 파일: {SAVE_FILE}")
        
    except Exception as e:
        print(f"오류 발생: {e}")
    finally:
        if robot_recv is not None:
            robot_recv.disconnect()


if __name__ == "__main__":
    save_tcp_pose()
