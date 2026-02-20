import cv2
import numpy as np

# =========================
# Sharpness / Texture utils
# =========================

def sharpness_map_tenengrad(gray):
    """
    Tenengrad sharpness map:
    mag(x,y) = Gx^2 + Gy^2
    """
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
  
    return gx * gx + gy * gy


def frame_score_from_map(sharp_map, top_percent=5):
    """
    Frame sharpness score using top percentile mean
    """
    v = sharp_map.reshape(-1)
    thr = np.percentile(v, 100 - top_percent)
    sel = v[v >= thr]
    return float(sel.mean()) if sel.size > 0 else 0.0


def texture_energy(gray):
    """
    Texture existence check
    (low value => flat surface like white paper)
    """
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return float((gx * gx + gy * gy).mean())


# =========================
# Parameters (현장 튜닝용)
# =========================

RTSP_URL = "rtsp://169.254.4.176:554/h264"

# ROI 설정 (전체 프레임 기준 비율)
ROI_X1, ROI_Y1 = 0.25, 0.25
ROI_X2, ROI_Y2 = 0.75, 0.75

# 임계값 (반드시 현장에서 1번 튜닝)
TEXTURE_THRESH = 20.0      # 이하면 "NO TEXTURE"
SHARP_THRESH   = 10000.0     # 이하면 BLUR

TOP_PERCENT = 5            # 샤프니스 상위 퍼센트 사용


# =========================
# RTSP Capture
# =========================

cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    raise RuntimeError("❌ RTSP 연결 실패")

print("✅ RTSP 연결 성공")

# =========================
# Main loop
# =========================

while True:
    cap.grab()
    ret, frame = cap.retrieve()
    if not ret:
        print("❌ 프레임 수신 실패")
        break

    h, w = frame.shape[:2]

    # ROI 계산
    x1 = int(w * ROI_X1)
    y1 = int(h * ROI_Y1)
    x2 = int(w * ROI_X2)
    y2 = int(h * ROI_Y2)

    roi = frame[y1:y2, x1:x2]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # 1️⃣ 텍스처 존재 여부
    tex = texture_energy(gray)

    if tex < TEXTURE_THRESH:
        status = "NO TEXTURE"
        score = 0.0
        sharp_map = np.zeros_like(gray, dtype=np.float32)
        color = (0, 255, 255)  # 노랑
    else:
        # 2️⃣ 샤프니스 계산
        sharp_map = sharpness_map_tenengrad(gray)
        score = frame_score_from_map(sharp_map, TOP_PERCENT)

        if score >= SHARP_THRESH:
            status = "FOCUS OK"
            color = (0, 255, 0)
        else:
            status = "BLUR"
            color = (0, 0, 255)

    # =========================
    # Visualization
    # =========================

    # Sharpness heatmap
    sm_norm = cv2.normalize(sharp_map, None, 0, 255, cv2.NORM_MINMAX)
    sm_norm = sm_norm.astype(np.uint8)
    sm_color = cv2.applyColorMap(sm_norm, cv2.COLORMAP_JET)

    # ROI 표시
    view = frame.copy()
    cv2.rectangle(view, (x1, y1), (x2, y2), color, 2)

    # 텍스트
    cv2.putText(view, f"Texture: {tex:.1f}",
                (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.putText(view, f"Sharpness: {score:.1f}",
                (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.putText(view, f"Status: {status}",
                (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)

    # =========================
    # Side-by-side Visualization
    # =========================

    # 왼쪽: 원본 프레임 + ROI 박스
    view_left = frame.copy()
    cv2.rectangle(view_left, (x1, y1), (x2, y2), color, 2)

    cv2.putText(view_left, f"Texture: {tex:.1f}",
                (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(view_left, f"Sharpness: {score:.1f}",
                (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(view_left, f"Status: {status}",
                (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)

    # 오른쪽: ROI Sharpness Heatmap (프레임 크기로 확대)
    heatmap_big = cv2.resize(sm_color, (w, h))

    cv2.putText(heatmap_big, "Sharpness Heatmap (ROI)",
                (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 255), 2)

    # 좌우 결합
    view = cv2.hconcat([view_left, heatmap_big])

    cv2.imshow("Focus Check (RTSP)", view)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
