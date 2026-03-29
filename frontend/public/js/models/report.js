// DB - users/daily_report

// 1) models
// 리포트용
class DailyReport {
    constructor(date, name, course, grade, totalScore, absentCount){
      // 수정 불가
      this._date = date;
      this._name = name;
      this._course = course;

      this.grade = grade;
      this.totalScore = totalScore;
      this.absentCount = absentCount;

      // 이후 추가 가능한 지표
      // totalStudyTime : 실제 학습 시간, 
      // participationRate : 수업 참여도
    }
    get date() { return this._date; }
    get name() { return this._name; }
    get course() { return this._course; }
}

// 상세분석 타임라인용 
  // 1분 간격으로 임시저장 & 10분 간격으로 타임라인 점수(avg) 생성
class DailyTimeline {
    constructor(date, uid, tempScore=[], timelineScore = []){
      this._date = date;
      this._uid = uid; // 부모 폴더 uid(조회용)
      
      this.tempScore = tempScore; // 1분 간격 임시 저장 (10분마다 리셋)
      this.timelineScore = timelineScore; // 10분 간격으로 타임라인 생성
      // 이후 추가 가능한 지표
      // absentCount
    }
    get date() { return this._date; }
    get uid() { return this._uid; }
}


// 2) converters
const dailyReportConverter = {
  fromFirestore: (snapshot, options)=>{ 
    const data = snapshot.data(options);
    return new DailyReport(
      snapshot.id, // date
      data.name,
      data.course,
      data.grade || "N/A",
      data.totalScore || 0,
      data.absentCount || 0,
    );
  },

  toFirestore: (dailyReport) => { 
    return { 
      //date = 파일명
      name: dailyReport.name,
      course: dailyReport.course,
      grade: dailyReport.grade,
      totalScore: dailyReport.totalScore,
      absentCount: dailyReport.absentCount,
    }
  }
}

const dailyTLConverter = {
  fromFirestore: (snapshot, options)=>{
    const data = snapshot.data(options); 
    return new DailyTimeline(
      snapshot.id, // date
      data.uid,
      data.tempScore,
      data.timelineScore,
    );
  },

  toFirestore: (dailyTL) => { 
    return { 
      //date = 파일명
      uid: dailyTL.uid, //조회 편의상 추가
      tempScore: dailyTL.tempScore,
      timelineScore: dailyTL.timelineScore,
    }
  }
}