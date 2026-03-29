// 🔥 Firebase 설정 (CDN 방식)
// ⚠️ 중요: 아래 설정값들을 실제 Firebase 프로젝트 설정으로 교체해야 합니다!
// 
// 📋 Firebase 설정 방법:
// 1. https://console.firebase.google.com/ 접속
// 2. 프로젝트 선택 (또는 새 프로젝트 생성)
// 3. 프로젝트 설정 > 일반 탭 > "Firebase SDK 스니펫" > "구성" 선택
// 4. 아래 값들을 복사해서 교체
//
const firebaseConfig = {
  apiKey: "AIzaSyCGJ9PzArXNTZx-1uqpJp6AaZLXXo9Cpjw",
  authDomain: "cv-focus-system.firebaseapp.com",
  projectId: "cv-focus-system",
  storageBucket: "cv-focus-system.firebasestorage.app",
  messagingSenderId: "635999713897",
  appId: "1:635999713897:web:e4af6ae707a1174439c0f8",
  measurementId: "G-ZE8BC4E9X8"
};

// collection 이름 정의
const COL_USERS = "users";
const COL_DAILY_REPORT = "daily_report";
const COL_DAILY_TIMELINE = "daily_timeline";
const COL_COURSES = "courses";
const COL_DAILY = "daily";
const COL_REALTIME = "realtime";

// Firebase 초기화 (Firestore용 CDN 방식)
let db;
let auth; // 🌟 구글 로그인을 위한 Auth 엔진
const googleProvider = new firebase.auth.GoogleAuthProvider();

try {
  firebase.initializeApp(firebaseConfig);
  db = firebase.firestore();
  auth = firebase.auth();
  console.log("✅ 나만의 Firebase Firestore & Auth 초기화 완벽 성공!");
} catch (error) {
  console.error("❌ Firebase 초기화 실패:", error);
}

// Firebase Firestore 관련 함수들 (CDN 방식)
const firebaseUtils = {
  // Firebase 연결 상태 확인
  isFirebaseConnected() {
    return db && firebaseConfig.projectId !== "your-project-name";
  },

  // 날짜를 YYYYMMDD 형식으로 변환
  formatDateForFirebase(dateString) {
    return dateString.replace(/-/g, ''); // 하이픈 제거: 2025-10-08 → 20251008
  },

  // 일별 데이터 가져오기 (Firestore)
  async getDailyData(dateString) {
    // Firebase 연결 확인
    if (!this.isFirebaseConnected()) {
      console.warn("⚠️ Firebase가 설정되지 않았습니다. 일별 테스트 데이터를 반환합니다.");
      return Math.floor(Math.random() * 30) + 70; // 70-99 범위의 임시 테스트 데이터
    }

    try {
      const firebaseDateString = this.formatDateForFirebase(dateString);
      const docRef = db.collection('helmo').doc('results').collection('daily').doc(firebaseDateString);
      const doc = await docRef.get();
      
      if (doc.exists) {
        const data = doc.data();
        return data.result || null;
      } else {
        console.log(`해당 날짜(${firebaseDateString})의 일별 데이터가 없습니다.`);
        return null;
      }
    } catch (error) {
      console.error("일별 데이터를 가져오는 중 오류 발생:", error);
      console.error("🔥 Firebase Firestore 설정을 확인해주세요!");
      return null;
    }
  },

  // 주간 데이터 계산 및 저장 (일간 데이터 기반)
  async getWeeklyData(startDate, endDate) {
    // Firebase 연결 확인
    if (!this.isFirebaseConnected()) {
      console.warn("⚠️ Firebase가 설정되지 않았습니다. 주간 테스트 데이터를 반환합니다.");
      return Math.floor(Math.random() * 20) + 75; // 75-94 범위의 임시 테스트 데이터
    }

    try {
      // 해당 주의 일간 데이터들을 수집
      const dailyResults = [];
      const current = new Date(startDate);
      const end = new Date(endDate);
      
      console.log(`📊 주간 데이터 계산: ${startDate} ~ ${endDate}`);
      
      while (current <= end) {
        const dateString = `${current.getFullYear()}-${(current.getMonth() + 1).toString().padStart(2, '0')}-${current.getDate().toString().padStart(2, '0')}`;
        const dailyData = await this.getDailyData(dateString);
        
        if (dailyData) {
          dailyResults.push(dailyData);
          console.log(`  ✅ ${dateString}: ${dailyData}`);
        } else {
          console.log(`  ❌ ${dateString}: 데이터 없음`);
        }
        
        current.setDate(current.getDate() + 1);
      }
      
      if (dailyResults.length > 0) {
        // 주간 평균 계산
        const average = dailyResults.reduce((sum, val) => sum + val, 0) / dailyResults.length;
        const weeklyResult = Math.round(average);
        
        // 주간 데이터를 Firebase에 저장 (주의 시작 날짜를 키로 사용)
        const weekStartKey = this.formatDateForFirebase(startDate);
        await this.saveWeeklyData(weekStartKey, weeklyResult);
        
        console.log(`📈 주간 평균: ${weeklyResult} (${dailyResults.length}일 기준)`);
        return weeklyResult;
      } else {
        console.log("해당 주의 일간 데이터가 없습니다.");
        return null;
      }
    } catch (error) {
      console.error("주간 데이터를 계산하는 중 오류 발생:", error);
      return null;
    }
  },

  // 주간 데이터 저장
  async saveWeeklyData(weekStartKey, result) {
    try {
      const docRef = db.collection('helmo').doc('results').collection('weekly').doc(weekStartKey);
      await docRef.set({
        result: result,
        updatedAt: new Date().toISOString(),
        calculatedFrom: 'daily'
      });
      console.log(`💾 주간 데이터 저장 완료: ${weekStartKey} = ${result}`);
    } catch (error) {
      console.error("주간 데이터 저장 중 오류:", error);
    }
  },

  // 월간 데이터 계산 및 저장 (일간 데이터 기반)
  async getMonthlyData(year, month) {
    // Firebase 연결 확인
    if (!this.isFirebaseConnected()) {
      console.warn("⚠️ Firebase가 설정되지 않았습니다. 월간 테스트 데이터를 반환합니다.");
      return Math.floor(Math.random() * 15) + 80; // 80-94 범위의 임시 테스트 데이터
    }

    try {
      // 해당 월의 일간 데이터들을 수집
      const dailyResults = [];
      const daysInMonth = new Date(year, month, 0).getDate();
      
      console.log(`📊 월간 데이터 계산: ${year}년 ${month}월 (${daysInMonth}일)`);
      
      for (let day = 1; day <= daysInMonth; day++) {
        const dateString = `${year}-${month.toString().padStart(2, '0')}-${day.toString().padStart(2, '0')}`;
        const dailyData = await this.getDailyData(dateString);
        
        if (dailyData) {
          dailyResults.push(dailyData);
          console.log(`  ✅ ${dateString}: ${dailyData}`);
        } else {
          console.log(`  ❌ ${dateString}: 데이터 없음`);
        }
      }
      
      if (dailyResults.length > 0) {
        // 월간 평균 계산
        const average = dailyResults.reduce((sum, val) => sum + val, 0) / dailyResults.length;
        const monthlyResult = Math.round(average);
        
        // 월간 데이터를 Firebase에 저장 (YYYYMM 형식)
        const monthKey = `${year}${month.toString().padStart(2, '0')}`;
        await this.saveMonthlyData(monthKey, monthlyResult);
        
        console.log(`📈 월간 평균: ${monthlyResult} (${dailyResults.length}일 기준)`);
        return monthlyResult;
      } else {
        console.log("해당 월의 일간 데이터가 없습니다.");
        return null;
      }
    } catch (error) {
      console.error("월간 데이터를 계산하는 중 오류 발생:", error);
      return null;
    }
  },

  // 월간 데이터 저장
  async saveMonthlyData(monthKey, result) {
    try {
      const docRef = db.collection('helmo').doc('results').collection('monthly').doc(monthKey);
      await docRef.set({
        result: result,
        updatedAt: new Date().toISOString(),
        calculatedFrom: 'daily'
      });
      console.log(`💾 월간 데이터 저장 완료: ${monthKey} = ${result}`);
    } catch (error) {
      console.error("월간 데이터 저장 중 오류:", error);
    }
  },

  // 데이터 저장 함수들 (Firestore)
  async setDailyData(dateString, resultValue) {
    try {
      const firebaseDateString = this.formatDateForFirebase(dateString);
      const docRef = db.collection('helmo').doc('results').collection('daily').doc(firebaseDateString);
      await docRef.set({ result: resultValue });
      console.log(`일별 데이터가 성공적으로 저장되었습니다: ${firebaseDateString} = ${resultValue}`);
      return true;
    } catch (error) {
      console.error("일별 데이터 저장 중 오류 발생:", error);
      return false;
    }
  },



  // 등급 계산 함수
  calculateGrade(value) {
    if (value >= 90) return 'A';
    else if (value >= 80) return 'B';
    else if (value >= 70) return 'C';
    else if (value >= 60) return 'D';
    else return 'F';
  },

  // 일별 결과 포맷팅
  formatDailyResult(resultValue) {
    if (resultValue === null || resultValue === undefined) {
      return '결과 없음';
    }
    
    return `결과 값: ${resultValue}`;
  },

  // 주간 결과 포맷팅
  formatWeeklyResult(averageValue) {
    if (averageValue === null || averageValue === undefined) {
      return '등급 / 평균 착용률';
    }
    
    const grade = this.calculateGrade(averageValue);
    return `등급: ${grade} / 평균 착용률: ${averageValue}%`;
  },

  // 월간 결과 포맷팅
  formatMonthlyResult(averageValue) {
    if (averageValue === null || averageValue === undefined) {
      return '등급 / 월 평균 착용률';
    }
    
    const grade = this.calculateGrade(averageValue);
    return `등급: ${grade} / 월 평균 착용률: ${averageValue}%`;
  }
};