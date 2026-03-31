var StudentReportCRUD = {
    // 0. 유틸리티 헬퍼
    formatDate: function(date) {
        if (typeof date === 'string') return date; 
        return date.toISOString().split('T')[0]; // Date 객체를 'YYYY-MM-DD'로
    },

    calculateGrade: function(value) {
        if (value >= 80) return 'A';
        if (value >= 70) return 'B';
        if (value >= 60) return 'C';
        if (value >= 50) return 'D';
        return 'F';
    },

    // 1. Daily Reports
    // daily data 조회
    getDailyData: async function(uid, dateString) {
        try {
            const docRef = db.collection(COL_USERS).doc(uid).collection(COL_DAILY_REPORT).doc(dateString);
            const doc = await docRef.get();
            return doc.exists ? doc.data().totalScore : null;
        } catch (error) {
            console.error("[Daily_report] totalScore 조회 실패: ", error);
            return null;
        }
    },

    // daily Report 전체 조회
    getDailyReport: async function(uid, dateString) {
        try {
            const docRef = db.collection(COL_USERS).doc(uid)
                            .collection(COL_DAILY_REPORT).doc(dateString);
            const doc = await docRef.get();
            return doc.exists ? doc.data() : null;
        } catch (error) {
            console.error("[Daily_report] 전체 조회 실패: ", error);
            return null;
        }
    },

    // timeline data 조회
    getTimelineScore: async function(uid, date) {
        var docRef = db.collection(COL_USERS).doc(uid)
                    .collection(COL_DAILY_TIMELINE).doc(date);
        try {
            var doc = await docRef.get();
            if (!doc.exists) return [];
            return doc.data().timelineScore || [];
        } catch (error) {
            console.error("[Daily_timeline] timelineScore 읽기 실패: ", error);
            return [];
        }
    },

    // daily report 생성 (수업 종료 시 호출)
    createDailyReport: async function(uid, date, userData, timelineScoreArray, absentCount) {
        if (!timelineScoreArray || timelineScoreArray.length === 0) {
            console.warn("정산할 데이터가 없습니다.");
            return null;
        }

        // 전체 평균 계산
        var totalScore = Math.round(
            scores.reduce((a, b) => a + b, 0) / scores.length
        );
        // 등급 판정
        var grade = this.calculateGrade(totalScore);

        try {
            await db.collection(COL_USERS).doc(uid)
                    .collection(COL_DAILY_REPORT).doc(date)
                    .set({
                        name: userData.name,
                        courseName: userData.courseName,
                        grade: grade,
                        totalScore: totalScore,
                        absentCount: absentCount,
                        createdAt: firebase.firestore.FieldValue.serverTimestamp() // 🌟 생성일자 추가 추천
                    });
            
            return { totalScore, grade };
        } catch (error) {
            console.error("[Daily_report] 저장 실패: ", error);
            return false;
        }
    },

    // monthly 조회
    getMonthDailyReports: async function(uid, year, month) {
        // 날짜 범위 생성
        const monthStr = String(month).padStart(2, '0');
        const lastDay = new Date(year, month, 0).getDate(); //월말이 며칠인지

        const startDate = `${year}-${monthStr}-01`;
        const endDate = `${year}-${monthStr}-${String(lastDay).padStart(2, '0')}`;

        try {
            const querySnapshot = await db.collection(COL_USERS).doc(uid)
                .collection(COL_DAILY_REPORT)
                .where(firebase.firestore.FieldPath.documentId(), '>=', startDate)
                .where(firebase.firestore.FieldPath.documentId(), '<=', endDate)
                .get();

            const monthlyData = {};
            querySnapshot.forEach(doc => {
                // 문서 ID(날짜)를 키로 해서 저장 { "2026-03-05": { grade: 'A', totalScore: 95 }, ... }
                monthlyData[doc.id] = doc.data();
            });

            return monthlyData; 
        } catch (error) {
            console.error("[Monthly_View] 조회 실패: ", error);
            return {};
        }
    },
    
    // weekly, monthly 평균 계산
    calculateRangeAverage: async function(uid, rangeDates) {
        // 병렬 호출(Promise.all 활용)
        const results = await Promise.all(
            rangeDates.map(date => this.getDailyData(uid, date))
        );

        const validScores = results.filter(score => score !== null);
        
        if (validScores.length === 0) return { score: 0, grade: 'N/A', count: 0 };

        const avg = Math.round(validScores.reduce((a, b) => a + b, 0) / validScores.length);
        return {
            score: avg,
            grade: this.calculateGrade(avg),
            count: validScores.length // 몇 일치 데이터인지 확인용
        };
    }
}


var DailyTLCRUD = {
    // 1. 1분마다 점수를 '추가'하는 기능
    addBulkScoresToList: async function(uid, date, scoresArray) {
        if (!scoresArray || scoresArray.length === 0) return;

        var docRef = db.collection('users').doc(uid)
                    .collection('daily_timeline').doc(date);

        try {
            await docRef.set({
                uid: uid,
                timelineScore: firebase.firestore.FieldValue.arrayUnion(...scoresArray),
                updatedAt: firebase.firestore.FieldValue.serverTimestamp()
            }, { merge: true });
            
            return true;
        } catch (error) {
            console.error("[Daily_timeline] 벌크 리스트 저장 실패: ", error);
            return false;
        }
    },

    // 2. timelineScore 전체 조회 (차트용)
    getTimelineScore: async function(uid, date) {
        const docRef = db.collection('users').doc(uid)
                         .collection('daily_timeline').doc(date);
        try {
            const doc = await docRef.get();
            // 🌟 데이터가 없으면 빈 리스트 [] 반환
            if (!doc.exists) return [];
            return doc.data().timelineScore || [];
        } catch (error) {
            console.error("[Daily_timeline] 읽기 실패: ", error);
            return [];
        }
    }
};