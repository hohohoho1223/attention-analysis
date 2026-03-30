var DailyTLCRUD = {
    // 1. 1분마다 점수를 '추가'하는 기능
    addTempScore: async function(uid, date, score) {
        var docRef = db.collection('users').doc(uid)
                       .collection('daily_timeline').doc(date); // 특정 행만 다루어야 하므로 converter 사용하지 않음

        try {
            // 문서가 없으면 생성(merge), 있으면 tempScore 배열에 점수 추가
            await docRef.set({
                uid: uid,
                tempScore: firebase.firestore.FieldValue.arrayUnion(score),
            }, { merge: true });
            
            return true;
        } catch (error) {
            console.error("[Daily_timeline] tempScore 저장 실패: ", error);
            return false;
        }
    },

    // 2. 10분마다 또는 수업 종료 시 점수를 '정산'하는 기능
    updateTimelineScore: async function(uid, date) {
        var docRef = db.collection('users').doc(uid)
                       .collection('daily_timeline').doc(date);  // 특정 행만 다루어야 하므로 converter 사용하지 않음
        try {
            await db.runTransaction(async function(transaction) {
                var doc = await transaction.get(docRef);
                var tempScore = doc.exists ? (doc.data().tempScore || []) : [];
                if (tempScore.length === 0) return;

                var avg = Math.round(
                    tempScore.reduce(function(a, b) { return a + b; }, 0) / tempScore.length
                );
                transaction.update(docRef, {
                    timelineScore: firebase.firestore.FieldValue.arrayUnion(avg),
                    tempScore: [] // tempScore 리셋
                }, { merge: true });
            });
            return true;
        } catch (error) {
            console.error("[Daily_timeline] timelineScore 저장 실패: ", error);
            return false;
        }
    }
};