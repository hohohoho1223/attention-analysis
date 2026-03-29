// DB - courses & courses/daily

// 1) models
// 과정 목록
class Course {
    constructor(courseId, courseName, instructor, isActive = true, createdAt = null){
        this.courseId = courseId;
        this.courseName = courseName;
        this.instructor = instructor;
        this.isActive = isActive;
        this.createdAt = createdAt;
    } 
}

// 수업별 타임라인
class CourseTimeline {
    constructor(date, avgScore, maxScore, minScore, timelineScore = [], studentCount = 0){
        this.date = date;
        this.avgScore = avgScore;
        this.maxScore = maxScore;
        this.minScore = minScore;
        this.timelineScore = timelineScore;
        this.studentCount = studentCount;
    } 
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
      data.avgScore || 0,
      data.maxScore || 0,
      data.minScore || 0,
      data.timelineScore || [],
      data.studentCount || 0,
    );
  },

  toFirestore: (courseTL) => { 
    return { 
      // date
      avgScore: courseTL.avgScore,
      maxScore: courseTL.maxScore,
      minScore: courseTL.minScore,
      timelineScore: courseTL.timelineScore,
      studentCount: courseTL.studentCount,
    }
  }
}