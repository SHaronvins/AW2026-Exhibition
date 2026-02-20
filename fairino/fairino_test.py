from windows.fairino_sdk import RPC

ROBOT_IP = "192.168.57.2"

robot = RPC(ROBOT_IP)
print("RPC 객체 생성 완료")

ret = robot.GetSDKVersion()
if ret[0] != 0:
    raise SystemExit(f"❌ SDK 호출 실패: {ret[0]}")
print("✅ 로봇 연결 성공!")
print("SDK:", ret[1])

# 현재 TCP
ret = robot.GetActualTCPPose(0)
if ret[0] != 0:
    raise SystemExit(f"❌ TCP 읽기 실패: {ret[0]}")
tcp = list(ret[1])
print("현재 TCP:", tcp)

# 목표 TCP (Base X +100mm)
P_target = tcp.copy()
P_target[0] += 50.0
print("목표 TCP:", P_target)

# 현재 Joint
ret = robot.GetActualJointPosDegree(0)
if ret[0] != 0:
    raise SystemExit(f"❌ Joint 읽기 실패: {ret[0]}")
J_cur = list(ret[1])
print("현재 Joint:", J_cur)

# 현재 세이프티 상태 출력
try:
    safety = robot.GetSafetyCode()
    print("현재 SafetyCode:", safety)
except Exception as e:
    print("SafetyCode 읽기 실패:", e)

print("MoveL 호출 시작")
err = robot.MoveL(
    P_target,                 # desc_pos
    0,                        # tool
    0,                        # user
    [0, 0, 0, 0, 0, 0],       # joint_pos -> 0이면 내부에서 IK 사용
    10.0,                     # vel (%): 5% → 20%로 조금 빠르게
    0.0,                      # acc
    100.0,                    # ovl
    -1.0,                     # blendR (블록킹)
    0,                        # blendMode
    [0, 0, 0, 0],             # exaxis_pos
    0,                        # search
    0,                        # offset_flag
    [0, 0, 0, 0, 0, 0],       # offset_pos
)
ret = robot.GetActualTCPPose(0)
print("MoveL 반환:", err)
print("➡ Base X +5cm 이동 명령 완료")
tcp = list(ret[1])
print(tcp)