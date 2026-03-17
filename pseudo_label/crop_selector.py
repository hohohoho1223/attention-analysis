# crop_selector.py
# 역할: 동영상별로 학생 패널 5개의 좌표를 마우스 드래그로 측정하고 JSON으로 저장
# 사용법: 실행 후 각 영상의 첫 프레임에서 학생 패널을 순서대로 드래그

import cv2 # open computer vision
import json
import os
from pathlib import Path

# ── 설정 ──────────────────────────────────────────
VIDEO_DIR = "./data/Zoom_Record"   # 동영상 파일들이 있는 폴더 경로
COORD_DIR = "./data/Zoom_Record/crop_coords"        # 좌표 JSON 저장 폴더
NUM_STUDENTS = 5                 # 선택할 학생 패널 수
# ─────────────────────────────────────────────────

os.makedirs(COORD_DIR, exist_ok=True)

# 동영상 폴더에서 .mp4 파일 목록 수집 (path에 있는 확장자가 포함된 것을 모두 검색(glob))
video_files = sorted(Path(VIDEO_DIR).glob("*.mp4")) 

# 클릭한 y 좌표를 저장할 변수
clicked_y = []

def on_click(event, x, y, flags, param):
    """마우스 클릭 시 y 좌표만 수집"""
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_y.append(y)
        print(f"  클릭: y={y}")

for video_path in video_files:
    coord_path = Path(COORD_DIR) / f"{video_path.stem}.json" # stem : 확장자를 뺀 부분 ex. crop_coords/lecture_01.json

    # 이미 좌표가 저장된 영상은 스킵 (재실행 시 중복 작업 방지)
    if coord_path.exists():
        print(f"[SKIP] {video_path.name} - 좌표 있음")
        continue

    # 영상의 첫번째 프레임만 읽어서 좌표 선택용 이미지로 사용
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read() # [bool, MatLike] # grab은 다음 파일 가져오고 읽지는 않음
    cap.release() # 닫기

    if not ret: # 영상을 읽어왔는지 여부
        print(f"[ERROR] {video_path.name} 프레임 읽기 실패")
        continue

    coords = []
    print(f"\n[{video_path.name}] 학생 패널 {NUM_STUDENTS}개를 순서대로 드래그하세요.")
    print("  드래그 후 Space 또는 Enter로 확정 / c로 다시 선택")

    # 1단계: 아무 패널 하나만 드래그 → x, w, h 기준값 확보
    print("  1단계: 아무 패널 하나를 드래그하세요 (x, w, h 기준값)")
    roi = cv2.selectROI(f"기준 드래그 — {video_path.name}", frame,
                        fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()

    base_x, _, base_w, base_h = roi  # y는 버림 (_로 무시)
    print(f"  기준값 — x={base_x}, w={base_w}, h={base_h}")

    # 2단계: 5개 패널의 좌상단 y만 클릭으로 수집
    print("  2단계: 각 패널 좌상단을 순서대로 클릭하세요 (5번)")
    clicked_y.clear()  # 이전 영상 클릭값 초기화

    display = frame.copy()
    cv2.line(display, (base_x, 0), (base_x, frame.shape[0]), (0, 100, 255), 1)  # x 위치 가이드선
    cv2.putText(display, "각 패널 좌상단을 순서대로 클릭 (5번)",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)

    cv2.imshow(f"y 좌표 클릭 — {video_path.name}", display)
    cv2.setMouseCallback(f"y 좌표 클릭 — {video_path.name}", on_click)

    # 5번 클릭될 때까지 대기하면서 박스 미리보기 업데이트
    while len(clicked_y) < NUM_STUDENTS:
        preview = display.copy()
        for i, y in enumerate(clicked_y):
            cv2.rectangle(preview,
                        (base_x, y),
                        (base_x + base_w, y + base_h), (0, 200, 0), 2)
            cv2.putText(preview, f"S{i+1}", (base_x+4, y+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
        cv2.imshow(f"y 좌표 클릭 — {video_path.name}", preview)
        cv2.waitKey(100)

    cv2.destroyAllWindows()

    # 3단계: x, w, h 고정 + y만 클릭값으로 조합
    coords = [
        {"x": int(base_x), "y": int(y), "w": int(base_w), "h": int(base_h)}
        for y in sorted(clicked_y)  # 위→아래 순서 정렬
    ]

    # 5개 좌표를 JSON 파일로 저장 (파일명: 영상명.json)
    with open(coord_path, "w") as f:
        json.dump({"video": video_path.name, "students": coords}, f, indent=2)
    print(f"[SAVED] {coord_path}")

print("\n모든 영상 좌표 측정 완료!")