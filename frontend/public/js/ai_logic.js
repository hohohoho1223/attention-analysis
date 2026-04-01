// public/js/ai_logic.js
// 2번 구조: 최종 상태 판단은 ai_server /analyze 가 담당하고,
// 프론트는 프레임 전송 + 응답 표시만 담당한다.

class RemoteAttentionState {
    constructor(payload = {}) {
        this.state = payload.state ?? "FOCUSED";
        this.score = payload.score ?? 100.0;
        this.is_fixated = payload.is_fixated ?? false;
        this.face_detected = payload.face_detected ?? false;
        this.gaze_direction = payload.gaze_direction ?? "Unknown";
        this.blink_bpm = payload.blink_bpm ?? 0;
        this.eye_focus_score = payload.eye_focus_score ?? 100.0;
        this.eye_status_msg = payload.eye_status_msg ?? "Eye analysis disabled";
        this.absent_count = payload.absent_count ?? 0;

        this.pose = payload.pose ?? { yaw: 0.0, pitch: 0.0, roll: 0.0 };
        this.body = {
            visible: payload.body?.visible ?? false,
            shoulder_tilt: payload.body?.shoulder_tilt ?? 0.0,
            smoothed_shoulder_tilt: payload.body?.smoothed_shoulder_tilt ?? 0.0,
        };
        this.durations = payload.durations ?? {
            head: 0.0,
            body: 0.0,
            fixation_break: 0.0,
            no_face: 0.0,
        };
        this.cnn = payload.cnn ?? {
            is_drowsy: false,
            left_eye: 0.0,
            right_eye: 0.0,
            mouth: 0.0,
            smooth: 0.0,
        };
    }
}

class RemoteAttentionEngine {
    constructor({
        baseUrl = "http://127.0.0.1:8000",
        sessionId = "frontend-default",
        processScale = 0.75,
        refineLandmarks = true,
        detectionConfidence = 0.5,
        trackingConfidence = 0.5,
        enableDefenseModel = true,
    } = {}) {
        this.baseUrl = baseUrl.replace(/\/$/, "");
        this.sessionId = sessionId;
        this.processScale = processScale;
        this.refineLandmarks = refineLandmarks;
        this.detectionConfidence = detectionConfidence;
        this.trackingConfidence = trackingConfidence;
        this.enableDefenseModel = enableDefenseModel;
        this.lastState = new RemoteAttentionState();
    }

    async analyzeFrame(videoOrCanvasEl, fps = 2.0) {
        const frame = this._encodeFrame(videoOrCanvasEl);
        const payload = {
            frame,
            session_id: this.sessionId,
            process_scale: this.processScale,
            refine_landmarks: this.refineLandmarks,
            detection_confidence: this.detectionConfidence,
            tracking_confidence: this.trackingConfidence,
            fps,
            enable_defense_model: this.enableDefenseModel,
        };

        const response = await fetch(`${this.baseUrl}/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });

        if (!response.ok) {
            const text = await response.text();
            throw new Error(`AI analyze failed: ${response.status} ${text}`);
        }
        
        // main.py return 값을 받는 구간 { state, score, face_detected, gaze_direction, blink_bpm, eye_focus_score, eye_status_msg, pose, body, durations, cnn } 형태의 JSON이므로 그대로 RemoteAttentionState로 매핑 
        const data = await response.json();
        this.lastState = new RemoteAttentionState(data);
        return this.lastState;
    }

    _encodeFrame(videoOrCanvasEl) {
        const canvas = document.createElement("canvas");
        const width = videoOrCanvasEl.videoWidth || videoOrCanvasEl.width;
        const height = videoOrCanvasEl.videoHeight || videoOrCanvasEl.height;

        if (!width || !height) {
            throw new Error("video/canvas 크기를 읽을 수 없습니다.");
        }

        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(videoOrCanvasEl, 0, 0, width, height);
        return canvas.toDataURL("image/jpeg", 0.9);
    }
}

// 하위 호환용 별칭
window.RemoteAttentionState = RemoteAttentionState;
window.RemoteAttentionEngine = RemoteAttentionEngine;