import pandas as pd
import shutil
import os
from pathlib import Path

# ── 경로 설정 ──────────────────────────────────────────
CSV_PATH    = r"C:\Users\Spudu\Downloads\daisee\DAiSEE\sampled\TrainLabels_filtered.csv"
VIDEO_SRC   = r"C:\Users\Spudu\Downloads\daisee\DAiSEE\sampled\Train"
OUTPUT_ROOT = r"C:\Users\Spudu\Downloads\daisee_small\train"

# ── 매핑 (Engagement 0→1, 3→3, 나머지 스킵) ──────────
def map_label(row):
    e = row['Engagement']
    if e == 0:
        return 1   # unfocus
    elif e == 2 or e == 1:
        return "others"   # partial
    else:
        return 3   # focus

# ── 실행 ───────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
df_use = df[df['used'] == 1].copy()
df_use['class'] = df_use.apply(map_label, axis=1)

# 출력 폴더 생성
for c in [1, "others", 3]:
    os.makedirs(os.path.join(OUTPUT_ROOT, str(c)), exist_ok=True)

# 파일 복사
not_found = []
for _, row in df_use.iterrows():
    clip_id = row['ClipID']
    target_dir = os.path.join(OUTPUT_ROOT, str(row['class']))

    # 하위 폴더 전체에서 파일명으로 검색
    matches = list(Path(VIDEO_SRC).rglob(clip_id))
    if not matches:
        not_found.append(clip_id)
        continue

    src = matches[0]
    dst = os.path.join(target_dir, src.name)
    shutil.copy2(src, dst)
    print(f"[OK] {src.name} → class {row['class']}")

print(f"\n완료: {len(df_use) - len(not_found)}개 복사, {len(not_found)}개 미발견")
if not_found:
    print("미발견 파일:", not_found)