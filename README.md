# 🦁 멋쟁이사자처럼 - 학습 집중도 분석 시스템 (CV Focus Project)

웹캠 이미지를 실시간으로 분석하여 학생들의 수업 집중도를 파악하고, 관리자가 이를 모니터링할 수 있는 AI 기반 웹 서비스입니다.

---

## ⚙️ 로컬 환경 실행 방법 (How to Run)

이 프로젝트는 **프론트엔드(Node.js)**와 **AI 백엔드(FastAPI)** 두 개의 서버를 동시에 켜야 완벽하게 작동합니다. 터미널 창을 2개 띄워주세요!

### 1️⃣ 프론트엔드 서버 실행 (웹 화면)
프론트엔드 폴더(`CV_Focus_Project` 최상위)에서 터미널을 열고 아래 명령어를 실행합니다.

패키지 설치 및 로컬 서버 실행 (최초 1회만)
```bash
cd frontend
npm install
node index.js
```

👉 실행 후 브라우저에서 `http://localhost:3000` 으로 접속합니다.

### 2️⃣ AI 백엔드 서버 실행 (FastAPI)
CNN 모델과 룰베이스 로직이 돌아가는 파이썬 서버입니다. 백엔드 폴더(`main.py`가 있는 곳)에서 터미널을 열고 아래 명령어를 실행합니다.

필요한 파이썬 라이브러리 설치 및 FastAPI 서버 실행(다른 터미널에서 최초 1회만)
```bash
cd ai_server
pip install -r requirements.txt
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

👉 `✅ [AI 서버] 모델 로드 완벽 성공!` 메시지가 뜨면 프론트엔드와 통신할 준비가 완료된 것입니다.

---

## 🎯 핵심 파이프라인
* **Student:** 웹캠 접속 → MediaPipe 랜드마크 추출 → 눈/입 크롭 이미지 FastAPI 서버로 전송
* **AI Server (FastAPI):** CNN 모델 분석 → 집중도 확률 프론트엔드로 반환
* **Firebase DB:** 0.5초 단위 점수 결산 → 수업 종료 시 일간 평균 및 타임라인 데이터 누적 저장
* **Admin:** 실시간 전체 학생 모니터링 및 월별/일별 상세 그래프 조회
