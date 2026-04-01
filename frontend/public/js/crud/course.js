var CourseCRUD = {
    // 1. 모든 과정 목록 조회
    getCourses: async function() {
        try {
            const snapshot = await db.collection(COL_COURSES)
                                     .orderBy("createdAt", "desc")
                                     .withConverter(courseConverter)
                                     .get();
            return snapshot.docs.map(doc => doc.data());
        } catch (error) {
            console.error("❌ 과정 목록 로드 실패:", error);
            return [];
        }
    },

    // 2. 과정 생성 (Create)
    addCourse: async function(courseName, instructor) {
        try {
            const newDocRef = db.collection(COL_COURSES).doc(); // 빈 문서 참조 생성 (id만 예약) 
        
            // 생성된 ID를 객체에 넣어줍니다.
            const newCourse = new Course(// id로 객체 생성
                newDocRef.id, // id 자동생성
                courseName, 
                instructor, 
                true, // isActive
                firebase.firestore.FieldValue.serverTimestamp() // createdAt
            );
            await newDocRef.withConverter(courseConverter).set(newCourse); // 그 객체를 DB에 저장
            return true;
        } catch (error) {
            console.error("❌ 과정 생성 실패:", error);
            return false;
        }
    },

    // 3. 과정 정보 수정 (Update)
    updateCourse: async function(courseId, updateData) {
        try {
            // 수정 보안: id와 생성일은 데이터 뭉치에서 제거 (구조 분해 할당)
            const { _courseId, _createdAt, ...pureData } = updateData;

            await db.collection(COL_COURSES).doc(courseId).update(pureData);
            console.log(`✅ [${courseId}] 과정 수정 완료`);
            return true;
        } catch (error) {
            console.error("❌ 과정 수정 실패:", error);
            return false;
        }
    },

    // 4. 과정 삭제 (Delete)
    deleteCourse: async function(courseId) {
        try {
            await db.collection(COL_COURSES).doc(courseId).delete();
            return true;
        } catch (error) {
            console.error("❌ 과정 삭제 실패:", error);
            return false;
        }
    },

    // 5. 과정별 학생 조회
    getStudentsByCourse: async (courseId) => {
        try {
            const snapshot = await db.collection(COL_USERS)
                .where("courseId", "==", courseId) // 1. 과정 필터
                .where("role", "==", "student")    // 2. 역할 필터
                .get();

            // 가져온 결과만 배열로 변환
            return snapshot.docs.map(doc => ({ uid: doc.id, ...doc.data() }));
        } catch (error) {
            console.error("과정별 학생 목록 조회 실패:", error);
            return [];
        }
    }
};


// 전체 학생의 타임라인 평균 계산
var CourseTLCRUD = {
    // 1. 과정별 전체 학생의 타임라인 점수 조회 (필터링 포함)
    getStudentTimelines: async function(courseId, date) {
        // CourseCRUD에서 학생 리스트를 가져온다고 가정
        const students = await CourseCRUD.getStudentsByCourse(courseId);
        
        const timelinePromises = students.map(async (student) => {
            // 기존 StudentReportCRUD의 메서드 활용
            return await StudentReportCRUD.getTimelineScore(student.uid, date);
        });

        const allTimelines = await Promise.all(timelinePromises);
        // 데이터가 있는 학생들의 타임라인만 필터링해서 반환
        return allTimelines.filter(tl => tl.length > 0);
    },

    // 2. 인덱스별 평균 계산 ([] + [] -> [])
    calculateAverageTimeline: function(allTimelines) {
        if (allTimelines.length === 0) return [];
        
        // 가장 긴 타임라인 길이를 기준으로 루프 수행
        const maxLength = Math.max(...allTimelines.map(tl => tl.length));
        const avgTimeline = [];

        for (let i = 0; i < maxLength; i++) {
            let sum = 0;
            let count = 0;
            allTimelines.forEach(tl => {
                if (tl[i] !== undefined) {
                    sum += tl[i];
                    count++;
                }
            });
            // 해당 시간대(index)의 반 전체 평균 push
            avgTimeline.push(count > 0 ? Math.round(sum / count) : 0);
        }
        return avgTimeline;
    },

    // 3. 계산된 데이터를 CourseTL DB에 저장 (Converter 사용)
    saveCourseTimeline: async function(courseId, date, timelineScore) {
        if (timelineScore.length === 0) return;

        // 전체 평균, 최고, 최저점 미리 계산
        const avgScore = Math.round(timelineScore.reduce((a, b) => a + b, 0) / timelineScore.length);
        const maxScore = Math.max(...timelineScore);
        const minScore = Math.min(...timelineScore);

        // 강사님이 만든 Model 인스턴스 생성
        const newTL = new CourseTimeline(
            date, 
            false, // isStarted: 기록 종료이므로 false
            avgScore, 
            maxScore, 
            minScore, 
            timelineScore, 
            timelineScore.length
        );

        const docRef = db.collection(COL_COURSES).doc(courseId)
                         .collection('course_timeline').doc(date)
                         .withConverter(courseTLConverter); // 컨버터 적용

        await docRef.set(newTL);
        return newTL;
    },

    // 4. 저장된 CourseTL 조회하기
    getCourseTimeline: async function(courseId, date) {
        const docRef = db.collection(COL_COURSES).doc(courseId)
                         .collection('course_timeline').doc(date)
                         .withConverter(courseTLConverter);
        
        const doc = await docRef.get();
        return doc.exists ? doc.data() : null;
    }
};