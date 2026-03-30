var UserCRUD = {
    // 1. 조회 (Read)
    getUser: async function(uid) {
        try {
            var docRef = db.collection(COL_USERS).doc(uid).withConverter(userConverter);
            var doc = await docRef.get();
            return doc.exists ? doc.data() : null; // 있으면 유저 정보, 없으면 null return
        } catch (error) {
            console.error("❌ 유저 조회 중 오류 발생:", error);
            return null;
        }
    },

    // 2. 생성 (Create)
    createUser: async function(user) {
        try {
            await db.collection(COL_USERS).doc(user.uid)
                .withConverter(userConverter)
                .set(user);
            console.log("✅ 신규 유저 등록 완료:", user.name);
            return true;
        } catch (error) {
            console.error("❌ 유저 생성 중 오류 발생:", error);
            return false;
        }
    },

    // 3. 수정 (Update)
    updateUser: async function(uid, updateData) {
        try {
            // uid, email, createdAt 수정 불가
            const {uid : _, email: __, createdAt:___, ...pureUpdateData} = updateData

            // 업데이트할 것이 없을 경우 빠져나오기
            if (Object.keys(pureUpdateData).length === 0) return null;

            await db.collection(COL_USERS).doc(uid).update(pureUpdateData);
            console.log("✅ 유저 정보 수정 완료");
            return pureUpdateData; // 수정 정보 반환
        } catch (error) {
            console.error("❌ 유저 수정 중 오류 발생:", error);
            return false;
        }
    },

    // 4. 삭제 (Delete)
    deleteUser: async function(uid) {
        try {
            await db.collection(COL_USERS).doc(uid).delete();
            console.log("✅ 유저 데이터 삭제 완료");
            return true;
        } catch (error) {
            console.error("❌ 유저 삭제 중 오류 발생:", error);
            return false;
        }
    }
};