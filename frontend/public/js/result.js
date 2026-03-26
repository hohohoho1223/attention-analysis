// URL에서 날짜 정보를 가져와 표시
const urlParams = new URLSearchParams(window.location.search);
const dateParam = urlParams.get('date');
const resultDateEl = document.getElementById("resultDate");

if (dateParam) {
    const dateObj = new Date(dateParam);
    const formattedDate = `${dateObj.getFullYear()}년 ${dateObj.getMonth() + 1}월 ${dateObj.getDate()}일`;
    resultDateEl.textContent = formattedDate;
} else {
    resultDateEl.textContent = "날짜 정보 없음";
}

// localStorage에서 선택된 이미지들을 가져와서 테이블에 표시
function loadSelectedImages() {
    const selectedImagesData = localStorage.getItem('selectedImages');
    
    if (selectedImagesData) {
        const selectedImages = JSON.parse(selectedImagesData);
        const tableRows = document.querySelectorAll('.result-table tbody tr');
        
        // 모든 테이블 행의 결과 이미지를 초기화
        tableRows.forEach(row => {
            const resultImgCell = row.cells[4]; // 결과 이미지 셀 (5번째 열)
            resultImgCell.innerHTML = '<span class="no-image">이미지 없음</span>';
        });
        
        // 선택된 이미지들을 해당 시간대에 맞게 배치
        selectedImages.forEach(imageData => {
            const slotId = parseInt(imageData.slotId);
            
            if (slotId < tableRows.length) {
                const targetRow = tableRows[slotId];
                const resultImgCell = targetRow.cells[4];
                
                resultImgCell.innerHTML = `
                    <img src="${imageData.imageData}" alt="분석 결과 이미지" class="result-img">
                `;
            }
        });
        
        // 사용 후 localStorage 정리 (선택사항)
        // localStorage.removeItem('selectedImages');
    }
}

// 페이지 로드 시 이미지 로드
document.addEventListener('DOMContentLoaded', loadSelectedImages);

// 문의 버튼 이벤트 처리
const inquiryButtons = document.querySelectorAll('.inquiry-btn');
inquiryButtons.forEach(btn => {
    btn.addEventListener('click', (e) => {
        const row = e.target.closest('tr');
        const timeSlot = row.cells[0].textContent;
        alert(`${timeSlot} 시간대에 대한 문의를 접수했습니다.`);
    });
});

function pdfbtnClicked(){
  const date = new URLSearchParams(window.location.search).get("date");

  fetch(`/result-pdf?date=${date}`)
    .then(response => response.blob())
    .then(blob => {
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `기록_${date}.pdf`; // 다운로드될 파일 이름
      document.body.appendChild(a);
      a.click(); // 자동 클릭으로 다운로드 시작
      a.remove();
      window.URL.revokeObjectURL(url); // 메모리 정리
    });
};