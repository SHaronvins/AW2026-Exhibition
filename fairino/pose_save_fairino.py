from windows.fairino_sdk import RPC

ROBOT_IP = "192.168.57.2"

SAVE_PATH = "robot_poses.txt"   # 저장 파일

# -----------------------------
# RPC 연결
# -----------------------------
robot = RPC(ROBOT_IP)
print("RPC 객체 생성 완료")

ret = robot.GetSDKVersion()
if ret[0] != 0:
    print("❌ SDK 호출 실패")
    exit()

print("✅ 로봇 연결 성공!")

# -----------------------------
# TCP 한 번 읽어서 저장
# -----------------------------
ret = robot.GetActualTCPPose(0)
print(ret)

if ret[0] != 0:
    print("❌ TCP 읽기 실패:", ret[0])
    exit()

# [mm, deg] 그대로 저장
# ret 형식: (에러코드, [x, y, z, rx, ry, rz])
x, y, z, rx, ry, rz = ret[1]

line = f"{x:.3f},{y:.3f},{z:.3f},{rx:.3f},{ry:.3f},{rz:.3f}\n"

with open(SAVE_PATH, "a") as f:
    f.write(line)

print("✅ TCP 저장 완료:")
print(line.strip())
