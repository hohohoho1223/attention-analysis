var RealtimeCRUD = {

    // 관리자 : 모니터링
    subscribeRealtimeList: (courseId, callback) => {
    return db.collection('courses').doc(courseId)
                .collection('realtime')
                .withConverter(realtimeConverter)
                .onSnapshot((snapshot) => {
                    const list = snapshot.docs.map(doc => doc.data());
                    callback(list);
                }, (error) => {
                    console.error("실시간 목록 구독 실패:", error);
                });
    },

    // 학생 : 개인의 상태 전송
    updateStatus: async (courseId, realtimeStatus) => {
        try {
            const docRef = db.collection('courses').doc(courseId)
                                .collection('realtime').doc(realtimeStatus.uid);
            
            // 컨버터를 사용하여 객체를 DB 형식으로 변환 후 저장
            await docRef.withConverter(realtimeConverter).set(realtimeStatus);
            return true;
        } catch (error) {
            console.error("실시간 상태 업데이트 실패:", error);
            return false;
        }
    },

    // 학생: 본인 이름 삭제 (현재 로직에서는 사용하지 않음)
    deleteStatus: async (courseId, uid) => {
        try {
            await db.collection('courses').doc(courseId)
                    .collection('realtime').doc(uid).delete();
            return true;
        } catch (error) {
            console.error("실시간 상태 삭제 실패:", error);
            return false;
        }
    }

};