// DB - courses & courses/daily

// 1) models
// 과정 목록
class Course {
    constructor(courseId, courseName, instructor, isActive = true, createdAt = null){
        // 수정 불가
        this._courseId = courseId;
        this._createdAt = createdAt;
        // 수정 가능
        this.courseName = courseName;
        this.instructor = instructor;
        this.isActive = isActive;
    } 

    //getter
    get courseId() {return this._courseId;}
    get createdAt() {return this._createdAt;}
}

// 수업별 타임라인
class CourseTimeline {
    constructor(date, isStarted, avgScore, maxScore, minScore, timelineScore = [], studentCount = 0, flushStatus = 'waiting', expectedCount = 0, completedUids = []){
        this._date = date;

        this.isStarted = isStarted;
        this.avgScore = avgScore;
        this.maxScore = maxScore;
        this.minScore = minScore;
        this.timelineScore = timelineScore;
        this.studentCount = studentCount;
        this.flushStatus = flushStatus;     // 'waiting' | 'calculating' | 'done'
        this.expectedCount = expectedCount; // 수업 종료 시점 접속자 수
        this.completedUids = completedUids; // 저장 완료한 학생 uid 배열
    } 
    get date() { return this._date; }
}

// 2) Converters
const courseConverter = {
    fromFirestore: (snapshot, options)=>{ 
        const data = snapshot.data(options);
        return new Course(
            snapshot.id, 
            data.courseName,
            data.instructor,
            data.isActive ?? true,
            data.createdAt || null,
    );
  },

  toFirestore: (course) => { 
      return { 
          // courseId
      courseName: course.courseName,
      instructor: course.instructor,
      isActive: course.isActive,
      createdAt: course.createdAt || firebase.firestore.FieldValue.serverTimestamp(),
    }
}
}

const courseTLConverter = {
  fromFirestore: (snapshot, options)=>{ 
    const data = snapshot.data(options);
    return new CourseTimeline(
      snapshot.id, 
      data.isStarted ?? true,
      data.avgScore || 0,
      data.maxScore || 0,
      data.minScore || 0,
      data.timelineScore || [],
      data.studentCount || 0,
      data.flushStatus || 'waiting',  
      data.expectedCount || 0,        
      data.completedUids || [],       
    );
  },

  toFirestore: (courseTL) => { 
    return { 
      // date
      isStarted: courseTL.isStarted,
      avgScore: courseTL.avgScore,
      maxScore: courseTL.maxScore,
      minScore: courseTL.minScore,
      timelineScore: courseTL.timelineScore,
      studentCount: courseTL.studentCount,
      flushStatus: courseTL.flushStatus,     
      expectedCount: courseTL.expectedCount, 
      completedUids: courseTL.completedUids, 
    }
  }
}