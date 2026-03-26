// 🔧 Firebase 연결 및 데이터 관리 유틸리티 함수들
// 주간/월간 데이터는 일간 데이터를 기반으로 자동 계산됩니다

// 🔧 개별 일간 데이터 추가 함수
async function addDailyData(dateString, resultValue) {
  try {
    const result = await firebaseUtils.setDailyData(dateString, resultValue);
    
    if (result) {
      console.log(`✅ ${dateString}: ${resultValue}점 저장 완료`);
      return true;
    }
  } catch (error) {
    console.error('❌ 일간 데이터 저장 실패:', error);
    return false;
  }
}

// 🔍 Firebase Firestore 연결 테스트 함수
async function testFirebaseConnection() {
  if (!firebaseUtils.isFirebaseConnected()) {
    console.log("❌ Firebase가 연결되지 않았습니다.");
    console.log("🔧 firebase-config.js에서 실제 Firebase 설정값으로 교체해주세요.");
    return false;
  }
  
  console.log("✅ Firebase Firestore 연결 성공!");
  console.log("🔥 프로젝트 ID:", firebase.app().options.projectId);
  
  // 연결 테스트를 위한 간단한 읽기 시도
  try {
    const testQuery = await db.collection('helmo').doc('results').get();
    console.log("📖 Firestore 읽기 테스트 성공!");
    return true;
  } catch (error) {
    console.error("❌ Firestore 읽기 테스트 실패:", error);
    return false;
  }
}

// 🌍 브라우저 콘솔에서 사용 가능하도록 전역 함수로 등록
window.testFirebaseConnection = testFirebaseConnection;
window.addDailyData = addDailyData;