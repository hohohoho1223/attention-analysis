// 월별 분석 상세보기 페이지 JavaScript

// URL에서 년도와 월 정보 가져오기
const urlParams = new URLSearchParams(window.location.search);
const year = urlParams.get('year') || new Date().getFullYear();
const month = urlParams.get('month') || (new Date().getMonth() + 1);

// 페이지 로드 시 초기화
document.addEventListener('DOMContentLoaded', function() {
    initializePage();
});

// 페이지 초기화
function initializePage() {
    // 제목 설정
    const monthYearElement = document.getElementById('monthLabel');
    if (monthYearElement) {
        monthYearElement.textContent = `${year}년 ${month}월`;
    }
    
    // 월별 데이터 로드 (임시 데이터)
    loadMonthlyData();
}

// 월별 데이터 로드 (임시 구현)
function loadMonthlyData() {
    // 임시로 더미 데이터 표시
    const dummyData = generateDummyData();
    displayMonthlyData(dummyData);
}

// 더미 데이터 생성
function generateDummyData() {
    const daysInMonth = new Date(year, month, 0).getDate();
    const monthlyData = [];
    
    for (let day = 1; day <= daysInMonth; day++) {
        // 랜덤하게 몇 개의 날짜만 데이터가 있는 것처럼 시뮬레이션
        if (Math.random() > 0.3) { // 70% 확률로 데이터 존재
            const rate = Math.floor(Math.random() * 40) + 60; // 60-100% 범위
            monthlyData.push({
                day: day,
                date: `${year}-${month.toString().padStart(2, '0')}-${day.toString().padStart(2, '0')}`,
                rate: rate,
                grade: getGrade(rate),
                status: rate >= 80 ? '양호' : rate >= 60 ? '보통' : '주의'
            });
        }
    }
    
    return monthlyData;
}

// 등급 계산
function getGrade(rate) {
    if (rate >= 90) return 'A';
    else if (rate >= 80) return 'B';
    else if (rate >= 70) return 'C';
    else if (rate >= 60) return 'D';
    else return 'F';
}

// 월별 데이터 표시
function displayMonthlyData(data) {
    if (data.length === 0) {
        // 데이터가 없는 경우
        document.querySelector('.text-gray-600').textContent = '해당 월의 데이터가 없습니다.';
        return;
    }
    
    // 요약 통계 계산
    const rates = data.map(item => item.rate);
    const avgRate = Math.round(rates.reduce((sum, rate) => sum + rate, 0) / rates.length);
    const maxRate = Math.max(...rates);
    const minRate = Math.min(...rates);
    
    // 월별 요약 표시 (기존 month_result.html의 구조에 맞춰서)
    console.log(`월별 분석 데이터:`);
    console.log(`- 총 기록 일수: ${data.length}일`);
    console.log(`- 평균 착용률: ${avgRate}%`);
    console.log(`- 최고 착용률: ${maxRate}%`);
    console.log(`- 최저 착용률: ${minRate}%`);
    
    // 차트 데이터 준비 (기존 Chart.js 사용)
    const chartData = prepareChartData(data);
    updateChart(chartData);
}

// 차트 데이터 준비
function prepareChartData(data) {
    const labels = data.map(item => `${item.day}일`);
    const rates = data.map(item => item.rate);
    
    return {
        labels: labels,
        datasets: [{
            label: '착용률 (%)',
            data: rates,
            borderColor: 'rgb(59, 130, 246)',
            backgroundColor: 'rgba(59, 130, 246, 0.1)',
            tension: 0.1
        }]
    };
}

// 차트 업데이트 (기존 Chart.js 코드 활용)
function updateChart(chartData) {
    // 기존 month_result.html의 차트 업데이트 로직 사용
    // Chart.js가 이미 초기화되어 있다고 가정
    if (window.monthlyChart) {
        window.monthlyChart.data = chartData;
        window.monthlyChart.update();
    }
}

// 메인 화면으로 이동
function goToCalendarPage() {
    window.location.href = '/';
}

// 이전 달
function prevMonth() {
    let newMonth = parseInt(month) - 1;
    let newYear = parseInt(year);
    
    if (newMonth < 1) {
        newMonth = 12;
        newYear -= 1;
    }
    
    window.location.href = `month_result.html?year=${newYear}&month=${newMonth}`;
}

// 다음 달
function nextMonth() {
    let newMonth = parseInt(month) + 1;
    let newYear = parseInt(year);
    
    if (newMonth > 12) {
        newMonth = 1;
        newYear += 1;
    }
    
    window.location.href = `month_result.html?year=${newYear}&month=${newMonth}`;
}