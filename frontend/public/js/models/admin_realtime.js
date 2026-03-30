// DB - courses/{courseId}/realtime/{uid}

// 1) model
// 수업 모니터링용
class Realtime {
    constructor(uid, name, course, status, currentScore, absentCount, lastUpdated = null){
      // 수정 불가  
      this._uid = uid;
      this._name = name;
      this._course = course;

      this.status = status;
      this.currentScore = currentScore;
      this.absentCount = absentCount;
      this.lastUpdated = lastUpdated;
    } 
    get uid() {return this._uid;}
    get name() { return this._name; }
    get course() { return this._course; }
}

// 2) converter
const realtimeConverter = {
  fromFirestore: (snapshot, options)=>{
    const data = snapshot.data(options); 
    return new Realtime(
      snapshot.id, // uid
      data.name,
      data.course,
      data.status || "N/A",
      data.currentScore ?? null,
      data.absentCount ?? 0,
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