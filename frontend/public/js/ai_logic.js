// public/js/ai_logic.js
// 파이썬 로직을 JS로 완벽 번역한 AI 엔진 코어

// ==========================================
// 1. Head Pose Estimator (고개 각도 추정)
// ==========================================
class PoseAngles {
    constructor(yaw = 0.0, pitch = 0.0, roll = 0.0) {
        this.yaw = yaw;
        this.pitch = pitch;
        this.roll = roll;
    }
}

class HeadPoseEstimator {
    // OpenCV solvePnP 대신 MediaPipe의 3D(z축) 데이터를 활용한 초고속 기하학 추정법
    estimate(faceLandmarks) {
        if (!faceLandmarks) return new PoseAngles();

        const topFace = faceLandmarks[10];
        const bottomFace = faceLandmarks[152];
        const leftFace = faceLandmarks[234];
        const rightFace = faceLandmarks[454];
        const leftEye = faceLandmarks[33];
        const rightEye = faceLandmarks[263];

        // Pitch (상하)
        const pitch = Math.atan2(
            bottomFace.z - topFace.z, 
            bottomFace.y - topFace.y
        ) * (180 / Math.PI) - 10; // 보정값 -10

        // Yaw (좌우)
        const yaw = Math.atan2(
            rightFace.z - leftFace.z, 
            rightFace.x - leftFace.x
        ) * (180 / Math.PI);

        // Roll (갸우뚱)
        const roll = Math.atan2(
            rightEye.y - leftEye.y, 
            rightEye.x - leftEye.x
        ) * (180 / Math.PI);

        return new PoseAngles(-yaw, pitch, roll); // 파이썬과 방향 맞추기 위해 yaw 반전
    }
}

// ==========================================
// 2. Eye Focus Analyzer (눈 깜빡임/시선 추적)
// ==========================================
class EyeFocusResult {
    constructor(gaze_direction="Unknown", blink_bpm=0, eye_focus_score=100.0, eye_status_msg="Disabled") {
        this.gaze_direction = gaze_direction;
        this.blink_bpm = blink_bpm;
        this.eye_focus_score = eye_focus_score;
        this.eye_status_msg = eye_status_msg;
    }
}

class EyeFocusAnalyzer {
    constructor() {
        this.LEFT_EYE = [362, 385, 387, 263, 373, 380];
        this.RIGHT_EYE = [33, 160, 158, 133, 153, 144];
        this.earBuffer = [];
        this.blinkTimestamps = [];
        this.eyeClosed = false;
        this.startTime = Date.now() / 1000;
        this.baseEar = 0.25; // 초기 캘리브레이션 임시값
    }

    _euclideanDistance(p1, p2) {
        return Math.hypot(p1.x - p2.x, p1.y - p2.y);
    }

    _calculateEAR(landmarks, eyeIndices) {
        const pts = eyeIndices.map(i => landmarks[i]);
        const v1 = this._euclideanDistance(pts[1], pts[5]);
        const v2 = this._euclideanDistance(pts[2], pts[4]);
        const h = this._euclideanDistance(pts[0], pts[3]);
        if (h <= 0) return 0.25;
        return (v1 + v2) / (2.0 * h);
    }

    analyze(faceLandmarks) {
        if (!faceLandmarks) return new EyeFocusResult("Unknown", 0, 0, "No Face");

        const earL = this._calculateEAR(faceLandmarks, this.LEFT_EYE);
        const earR = this._calculateEAR(faceLandmarks, this.RIGHT_EYE);
        const avgEar = (earL + earR) / 2.0;

        // 스무딩 (deque 길이 5)
        this.earBuffer.push(avgEar);
        if (this.earBuffer.length > 5) this.earBuffer.shift();
        const smoothEar = this.earBuffer.reduce((a, b) => a + b) / this.earBuffer.length;

        // BPM 계산
        const currentTime = Date.now() / 1000;
        const blinkThreshold = this.baseEar * 0.75;

        if (smoothEar < blinkThreshold) {
            if (!this.eyeClosed) {
                this.blinkTimestamps.push(currentTime);
                this.eyeClosed = true;
            }
        } else {
            this.eyeClosed = false;
        }

        // 60초 지난 기록 삭제
        while (this.blinkTimestamps.length > 0 && currentTime - this.blinkTimestamps[0] > 60) {
            this.blinkTimestamps.shift();
        }
        const blinkBpm = this.blinkTimestamps.length;

        // 시선 방향 (간단화된 기하학 버전)
        const leftEyeInner = faceLandmarks[133];
        const leftEyeOuter = faceLandmarks[33];
        const leftIris = faceLandmarks[468]; // 홍채 중앙
        const eyeWidth = leftEyeInner.x - leftEyeOuter.x;
        const irisPos = leftIris.x - leftEyeOuter.x;
        const gazeRatio = irisPos / eyeWidth;
        
        let gazeDirection = "Center";
        if (gazeRatio < 0.4) gazeDirection = "Right";
        else if (gazeRatio > 0.6) gazeDirection = "Left";

        // 상태 점수 부여
        let msg = "Focused (Optimal)";
        let score = 100.0;
        const elapsedTime = currentTime - this.startTime;

        if (gazeDirection !== "Center") { msg = "Distracted (Looking Away)"; score = 40.0; }
        else if (blinkBpm > 15) { msg = "Anxious/Distracted (High BPM)"; score = 60.0; }
        else if (elapsedTime > 10 && blinkBpm < 3) { msg = "Spacing Out (Low BPM)"; score = 50.0; }

        return new EyeFocusResult(gazeDirection, blinkBpm, score, msg);
    }
}

// ==========================================
// 3. Attention Analyzer (최종 점수/상태 산출)
// ==========================================
class AttentionConfig {
    constructor() {
        this.yaw_threshold = 50.0;
        this.pitch_threshold = 30.0;
        this.roll_threshold = 25.0;
        this.focused_yaw_threshold = 30.0;
        this.focused_pitch_threshold = 15.0;
        this.focused_roll_threshold = 10.0;
        this.partial_focus_time = 3.0;
        this.lost_focus_time = 5.0;
        this.no_face_time = 5.0;
        this.smoothing_alpha = 0.35;
        this.recovery_speed = 1.5;
    }
}

class AttentionState {
    constructor() {
        this.state = "FOCUSED";
        this.score = 100.0;
        this.head_duration = 0.0;
        this.body_duration = 0.0;
        this.fixation_break_duration = 0.0;
        this.no_face_duration = 0.0;
        this.face_detected = false;
        this.smoothed_yaw = 0.0;
        this.smoothed_pitch = 0.0;
        this.smoothed_roll = 0.0;
        this.smoothed_body_tilt = 0.0;
    }
}

class AttentionAnalyzer {
    constructor() {
        this.config = new AttentionConfig();
        this.state = new AttentionState();
    }

    _smoothValue(prev, curr) {
        return this.config.smoothing_alpha * curr + (1.0 - this.config.smoothing_alpha) * prev;
    }

    _isFocusedRange() {
        return Math.abs(this.state.smoothed_yaw) <= this.config.focused_yaw_threshold &&
               Math.abs(this.state.smoothed_pitch) <= this.config.focused_pitch_threshold &&
               Math.abs(this.state.smoothed_roll) <= this.config.focused_roll_threshold;
    }

    _isOverThreshold() {
        return Math.abs(this.state.smoothed_yaw) > this.config.yaw_threshold ||
               Math.abs(this.state.smoothed_pitch) > this.config.pitch_threshold ||
               Math.abs(this.state.smoothed_roll) > this.config.roll_threshold;
    }

    _isScreenFixated(gazeDirection) {
        const yaw = this.state.smoothed_yaw;
        const headLeft = yaw < -this.config.focused_yaw_threshold;
        const headRight = yaw > this.config.focused_yaw_threshold;
        const eyeCenter = gazeDirection === "Center";

        if (!headLeft && !headRight) return eyeCenter;
        if (headLeft && gazeDirection === "Right") return true; // 보상
        if (headRight && gazeDirection === "Left") return true; // 보상
        return false;
    }

    update(poseAngles, bodyTilt, faceDetected, dt, eyeResult, isDrowsy = false) {
        this.state.face_detected = faceDetected;

        // 🌟 CNN 방어막이 졸음을 감지했을 때의 무자비한 페널티!
        if (isDrowsy) {
            this.state.state = "DROWSY";
            this.state.score = Math.max(0.0, this.state.score - 8.0 * dt); // 초당 8점씩 깎임
            this.state.eye_status_msg = "DROWSY DETECTED: Penalty!";
            this.state.no_face_duration = 0.0;
            return this.state;
        }

        if (faceDetected) {
            this.state.no_face_duration = 0.0;
            this.state.smoothed_yaw = this._smoothValue(this.state.smoothed_yaw, poseAngles.yaw);
            this.state.smoothed_pitch = this._smoothValue(this.state.smoothed_pitch, poseAngles.pitch);
            this.state.smoothed_roll = this._smoothValue(this.state.smoothed_roll, poseAngles.roll);
            this.state.smoothed_body_tilt = this._smoothValue(this.state.smoothed_body_tilt, bodyTilt);

            const isFixated = this._isScreenFixated(eyeResult.gaze_direction);
            const headWarning = this._isOverThreshold() && !isFixated;

            if (headWarning) this.state.head_duration += dt;
            else if (this._isFocusedRange()) this.state.head_duration = Math.max(0, this.state.head_duration - dt * this.config.recovery_speed);

            if (!isFixated) this.state.fixation_break_duration += dt;
            else this.state.fixation_break_duration = Math.max(0, this.state.fixation_break_duration - dt * this.config.recovery_speed);

        } else {
            this.state.no_face_duration += dt;
            this.state.head_duration += dt;
            this.state.fixation_break_duration += dt;
        }

        // 상태 판별
        if (this.state.no_face_duration >= this.config.no_face_time) this.state.state = "ABSENT";
        else if (!faceDetected) this.state.state = "LOST_FOCUS";
        else if (this.state.fixation_break_duration >= this.config.lost_focus_time) this.state.state = "LOST_FOCUS";
        else if (this.state.fixation_break_duration >= this.config.partial_focus_time) this.state.state = "PARTIAL_FOCUS";
        else if (Math.abs(this.state.smoothed_body_tilt) > 20.0 || eyeResult.eye_focus_score < 70) this.state.state = "PARTIAL_FOCUS";
        else this.state.state = "FOCUSED";

        // 점수 계산 로직
        let score = 100.0;
        const lostDriver = Math.max(this.state.fixation_break_duration, this.state.no_face_duration);
        const partialDriver = this.state.fixation_break_duration;

        if (this.state.state === "ABSENT") score = 20.0 - Math.min(this.state.no_face_duration, 3.0) * 7.0;
        else if (this.state.state === "LOST_FOCUS") score = 55.0 - Math.min(lostDriver, 5.0) * 5.0;
        else if (this.state.state === "PARTIAL_FOCUS") score = 80.0 - Math.min(partialDriver, 4.0) * 3.0;

        if (this.state.state === "PARTIAL_FOCUS" && eyeResult.eye_focus_score < 100) score -= (100 - Math.max(0, eyeResult.eye_focus_score)) * 0.08;
        if (this.state.state === "FOCUSED" && eyeResult.eye_focus_score < 100) score -= (100 - Math.max(0, eyeResult.eye_focus_score)) * 0.03;

        this.state.score = Math.max(0, Math.min(100, score));
        return this.state;
    }
}