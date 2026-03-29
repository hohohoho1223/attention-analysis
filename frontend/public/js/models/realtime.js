// DB - courses/daily/realtime

// 1) model
// 수업 모니터링용
class RealtimeStatus {
    constructor(uid, name, course, status, currentScore, absentCount, lastUpdated = null){
        this.uid = uid;
        this.name = name;
        this.course = course;
        this.status = status;
        this.currentScore = currentScore;
        this.absentCount = absentCount;
        this.lastUpdated = lastUpdated;
    } 
}

// 2) converter
const realtimeConverter = {
  fromFirestore: (snapshot, options)=>{
    const data = snapshot.data(options); 
    return new RealtimeStatus(
      snapshot.id, // uid
      data.name,
      data.course,
      data.status || "N/A",
      data.currentScore || 0,
      data.absentCount || 0,
      data.lastUpdated,
    );
  },

  toFirestore: (realtime) => { 
    return { 
      uid: realtime.uid, // 조회 편의성으로 포함
      name: realtime.name,
      course: realtime.course,
      status: realtime.status,
      currentScore: realtime.currentScore,
      absentCount: realtime.absentCount,
      lastUpdated: firebase.firestore.FieldValue.serverTimestamp(),
    }
  }
}