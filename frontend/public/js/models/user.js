// DB - Users
// CND sytle : html에서 바로 쓸 수 있는 형태

// 1) Model
class User{
  constructor(uid, email, name, course, role = "student", createdAt = null){
    this.uid = uid;
    this.email = email;
    this.name = name;
    this.course = course;
    this.role = role;
    this.createdAt = createdAt; // DB에서 가져온 Timestamp 객체
  }
}

// 2) converter (firebase에서 자동으로 변환)
const userConverter = {
  
  // DB에서 자동 불러오기
  fromFirestore: (snapshot, options)=>{ // snapshot: firebase 문서, options: 서버에 아직 기록되지 않은 로컬 데이터를 어떻게 처리할지 결정하는 설정값
    const data = snapshot.data(options);
    
    // role 검증
    let userRole = data.role;

    if (!userRole) {
      console.warn(`[DB Warning] ${data.email}의 role 데이터가 없어 'student'로 임시 할당합니다.`);
      userRole = "student";
    }

    return new User(
      snapshot.id,
      data.email,
      data.name,
      data.course,
      userRole,
      data.createdAt
    );
  },

  // DB에 자동 쓰기
  toFirestore: (user) => { // user: crud에서 호출한 User class
    return { //  firebase에 저장하기 위해 일반 객체로 변환 
      //uid는 파일명으로 변환되므로 db 저장하지 않음
      email: user.email,
      name: user.name,
      course: user.course,
      role: user.role || "student", // null값 방어용
      createdAt: user.createdAt || firebase.firestore.FieldValue.serverTimestamp() // 값이 없을 시 신규 생성
    }
  }
}
