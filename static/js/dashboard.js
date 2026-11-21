
/* =========================================================
   STOCK MARKET ML DASHBOARD
   Complete dashboard + ML + Decision Support
========================================================= */


/* =========================================================
   1. INITIAL DATA
========================================================= */

const initialData = window.initialStockData || {};

const initialSymbol = window.initialStockSymbol || "ACB";

const initialLatestStock = window.initialLatestStock || {};

const stockSelect = document.getElementById("stockSelect");

const timeFilter = document.getElementById("timeFilter");

let currentStockData = initialData;

let predictionData = [];

let selectedPrediction = null;


/* =========================================================
   2. FORMAT UTILITIES
========================================================= */

function formatNumber(value, decimals = 2) {

    if (
        value === null ||
        value === undefined ||
        value === ""
    ) {
        return "--";
    }

    const number = Number(value);

    if (!Number.isFinite(number)) {
        return value;
    }

    return new Intl.NumberFormat(
        "vi-VN",
        {
            maximumFractionDigits: decimals
        }
    ).format(number);

}


function formatPercent(value, decimals = 2) {

    if (
        value === null ||
        value === undefined ||
        !Number.isFinite(Number(value))
    ) {
        return "--";
    }

    return `${Number(value).toFixed(decimals)}%`;

}


function formatDate(dateValue) {

    if (!dateValue) {
        return "--";
    }

    const date = new Date(dateValue);

    if (isNaN(date.getTime())) {
        return dateValue;
    }

    return date.toLocaleDateString(
        "vi-VN"
    );

}


function safeNumber(value, fallback = 0) {

    const number = Number(value);

    return Number.isFinite(number)
        ? number
        : fallback;

}


/* =========================================================
   3. UPDATE STOCK DETAILS
========================================================= */

function updateStockDetails(data, symbol) {

    if (!data) return;


    const detailSymbol = document.getElementById(
        "detailSymbol"
    );

    const detailClose = document.getElementById(
        "detailClose"
    );

    const detailOpen = document.getElementById(
        "detailOpen"
    );

    const detailHigh = document.getElementById(
        "detailHigh"
    );

    const detailLow = document.getElementById(
        "detailLow"
    );

    const detailVolume = document.getElementById(
        "detailVolume"
    );

    const detailDate = document.getElementById(
        "detailDate"
    );


    if (detailSymbol) {

        detailSymbol.textContent =
            symbol || data.symbol || "--";

    }


    if (detailClose) {

        detailClose.textContent = formatNumber(
            data.latest_price ?? data.close
        );

    }


    if (detailOpen) {

        detailOpen.textContent = formatNumber(
            data.latest_open ?? data.open
        );

    }


    if (detailHigh) {

        detailHigh.textContent = formatNumber(
            data.latest_high ?? data.high
        );

    }


    if (detailLow) {

        detailLow.textContent = formatNumber(
            data.latest_low ?? data.low
        );

    }


    if (detailVolume) {

        detailVolume.textContent = formatNumber(
            data.latest_volume ?? data.volume,
            0
        );

    }


    if (detailDate) {

        detailDate.textContent =
            data.latest_date ?? data.date ?? "--";

    }

}


/* =========================================================
   4. FILTER DATA BY TIME
========================================================= */

function filterStockData(data, range) {

    if (
        !data ||
        !data.dates ||
        data.dates.length === 0
    ) {
        return data;
    }


    if (range === "all") {
        return data;
    }


    let months = 0;


    switch (range) {

        case "1m":
            months = 1;
            break;

        case "3m":
            months = 3;
            break;

        case "6m":
            months = 6;
            break;

        case "1y":
            months = 12;
            break;

        default:
            return data;

    }


    const latestDate = new Date(
        data.dates[data.dates.length - 1]
    );


    const startDate = new Date(latestDate);

    startDate.setMonth(
        startDate.getMonth() - months
    );


    const indexes = data.dates
        .map((date, index) => ({
            date: new Date(date),
            index
        }))
        .filter(item => item.date >= startDate)
        .map(item => item.index);


    return {

        dates: indexes.map(i => data.dates[i]),

        prices: indexes.map(i => data.prices[i]),

        volumes: indexes.map(i => data.volumes[i]),

        ma5: indexes.map(i => data.ma5[i]),

        ma20: indexes.map(i => data.ma20[i])

    };

}


/* =========================================================
   5. CHART CONFIGURATION
========================================================= */

const chartCommonOptions = {

    responsive: true,

    maintainAspectRatio: false,

    interaction: {
        intersect: false,
        mode: "index"
    },

    plugins: {

        legend: {
            display: true,
            position: "top"
        },

        tooltip: {
            enabled: true
        }

    },

    scales: {

        x: {

            ticks: {
                maxTicksLimit: 10
            }

        },

        y: {
            beginAtZero: false
        }

    }

};


/* =========================================================
   6. PRICE CHART
========================================================= */

const priceChartElement = document.getElementById(
    "priceChart"
);


let priceChart = null;


if (
    priceChartElement &&
    typeof Chart !== "undefined"
) {

    priceChart = new Chart(
        priceChartElement,
        {

            type: "line",

            data: {

                labels: initialData.dates || [],

                datasets: [

                    {

                        label: "Giá đóng cửa",

                        data: initialData.prices || [],

                        borderColor: "#60a5fa",

                        backgroundColor:
                            "rgba(96, 165, 250, 0.10)",

                        borderWidth: 2,

                        tension: 0.3,

                        fill: true,

                        pointRadius: 0,

                        pointHoverRadius: 5

                    },


                    {

                        label: "MA5",

                        data: initialData.ma5 || [],

                        borderColor: "#f59e0b",

                        borderWidth: 2,

                        tension: 0.3,

                        fill: false,

                        pointRadius: 0

                    },


                    {

                        label: "MA20",

                        data: initialData.ma20 || [],

                        borderColor: "#a78bfa",

                        borderWidth: 2,

                        tension: 0.3,

                        fill: false,

                        pointRadius: 0

                    }

                ]

            },

            options: chartCommonOptions

        }

    );

}


/* =========================================================
   7. VOLUME CHART
========================================================= */

const volumeChartElement = document.getElementById(
    "volumeChart"
);


let volumeChart = null;


if (
    volumeChartElement &&
    typeof Chart !== "undefined"
) {

    volumeChart = new Chart(
        volumeChartElement,
        {

            type: "bar",

            data: {

                labels: initialData.dates || [],

                datasets: [

                    {

                        label: "Khối lượng giao dịch",

                        data: initialData.volumes || [],

                        backgroundColor:
                            "rgba(34, 197, 94, 0.55)",

                        borderColor: "#22c55e",

                        borderWidth: 1

                    }

                ]

            },

            options: {

                ...chartCommonOptions,

                scales: {

                    x: {

                        ticks: {
                            maxTicksLimit: 10
                        }

                    },

                    y: {

                        beginAtZero: true

                    }

                }

            }

        }

    );

}


/* =========================================================
   8. UPDATE CHARTS
========================================================= */

function updateCharts(data) {

    if (!data) return;


    const selectedRange = timeFilter
        ? timeFilter.value
        : "all";


    const filteredData = filterStockData(
        data,
        selectedRange
    );


    if (priceChart) {

        priceChart.data.labels =
            filteredData.dates || [];

        priceChart.data.datasets[0].data =
            filteredData.prices || [];

        priceChart.data.datasets[1].data =
            filteredData.ma5 || [];

        priceChart.data.datasets[2].data =
            filteredData.ma20 || [];

        priceChart.update();

    }


    if (volumeChart) {

        volumeChart.data.labels =
            filteredData.dates || [];

        volumeChart.data.datasets[0].data =
            filteredData.volumes || [];

        volumeChart.update();

    }

}


/* =========================================================
   9. TIME FILTER EVENT
========================================================= */

if (timeFilter) {

    timeFilter.addEventListener(
        "change",
        function () {

            updateCharts(currentStockData);

        }
    );

}


/* =========================================================
   10. CHANGE STOCK
========================================================= */

if (stockSelect) {

    stockSelect.addEventListener(
        "change",
        async function () {

            const symbol = this.value;


            try {

                console.log(
                    "Đang tải dữ liệu:",
                    symbol
                );


                const response = await fetch(
                    `/api/stock/${symbol}`
                );


                if (!response.ok) {

                    throw new Error(
                        "Không thể tải dữ liệu cổ phiếu"
                    );

                }


                const data = await response.json();


                updateStockDetails(
                    data,
                    symbol
                );


                currentStockData = data;


                updateCharts(
                    currentStockData
                );


                updateActiveStock(
                    symbol
                );


                console.log(
                    `Đã cập nhật dữ liệu ${symbol}`
                );


            } catch (error) {

                console.error(
                    "Lỗi tải cổ phiếu:",
                    error
                );


                alert(
                    "Không thể tải dữ liệu cổ phiếu. Vui lòng thử lại."
                );

            }

        }
    );

}


/* =========================================================
   11. UPDATE ACTIVE STOCK LIST
========================================================= */

function updateActiveStock(symbol) {

    const stockItems = document.querySelectorAll(
        ".stock-list-item"
    );


    stockItems.forEach(item => {

        item.classList.remove("active");


        if (item.dataset.symbol === symbol) {

            item.classList.add("active");

        }

    });

}


/* =========================================================
   12. CLICK STOCK LIST
========================================================= */

const stockListItems = document.querySelectorAll(
    ".stock-list-item"
);


stockListItems.forEach(item => {

    item.addEventListener(
        "click",
        function () {

            const symbol = this.dataset.symbol;


            if (!stockSelect) return;


            stockSelect.value = symbol;


            stockSelect.dispatchEvent(
                new Event("change")
            );

        }
    );

});


/* =========================================================
   13. ML PREDICTION ELEMENTS
========================================================= */

const predictionTableBody = document.getElementById(
    "predictionTableBody"
);

const predictionLoading = document.getElementById(
    "predictionLoading"
);

const predictionError = document.getElementById(
    "predictionError"
);

const predictionWarning = document.getElementById(
    "predictionWarning"
);

const predictionModel = document.getElementById(
    "predictionModel"
);

const predictionThreshold = document.getElementById(
    "predictionThreshold"
);

const predictionDate = document.getElementById(
    "predictionDate"
);

const predictionFreshness = document.getElementById(
    "predictionFreshness"
);

const refreshPredictionBtn = document.getElementById(
    "refreshPredictionBtn"
);


/* =========================================================
   14. PREDICTION TABLE
========================================================= */

function renderPredictionTable(stocks) {

    if (!predictionTableBody) return;


    predictionTableBody.innerHTML = "";


    stocks.forEach(stock => {

        const row = document.createElement("tr");

        row.dataset.symbol = stock.symbol;

        row.style.cursor = "pointer";


        const probabilityClass =
            stock.prediction === 1
                ? "probability-positive"
                : "probability-neutral";


        const signalClass =
            stock.prediction === 1
                ? "signal-positive"
                : "signal-neutral";


        row.innerHTML = `

            <td>

                <span class="rank-badge">
                    ${stock.rank}
                </span>

            </td>


            <td>

                <strong class="prediction-symbol">
                    ${stock.symbol}
                </strong>

            </td>


            <td>
                ${formatNumber(stock.price)}
            </td>


            <td>

                <div class="probability-cell">

                    <strong class="${probabilityClass}">
                        ${formatPercent(stock.probability_up)}
                    </strong>


                    <div class="probability-bar">

                        <div
                            class="probability-bar-fill"
                            style="width:${Math.min(
                                Math.max(stock.probability_up, 0),
                                100
                            )}%">
                        </div>

                    </div>

                </div>

            </td>


            <td>

                <span class="signal-badge ${signalClass}">
                    ${stock.signal}
                </span>

            </td>


            <td>
                ${formatNumber(stock.rsi)}
            </td>


            <td>
                ${formatNumber(stock.macd, 4)}
            </td>

        `;


        row.addEventListener(
            "click",
            function () {

                selectedPrediction = stock;

                renderStockAnalysis(
                    stock
                );


                const panel = document.getElementById(
                    "stock-analysis-panel"
                );


                if (panel) {

                    panel.scrollIntoView({
                        behavior: "smooth",
                        block: "start"
                    });

                }


                updateActivePredictionRow(
                    stock.symbol
                );

            }
        );


        predictionTableBody.appendChild(row);

    });

}


/* =========================================================
   15. ACTIVE PREDICTION ROW
========================================================= */

function updateActivePredictionRow(symbol) {

    const rows = document.querySelectorAll(
        "#predictionTableBody tr"
    );


    rows.forEach(row => {

        row.classList.remove(
            "prediction-row-active"
        );


        if (row.dataset.symbol === symbol) {

            row.classList.add(
                "prediction-row-active"
            );

        }

    });

}


/* =========================================================
   16. LOAD ML PREDICTIONS
========================================================= */

async function loadPredictions() {

    if (!predictionTableBody) return;


    if (predictionLoading) {

        predictionLoading.classList.remove(
            "hidden"
        );

    }


    if (predictionError) {

        predictionError.classList.add(
            "hidden"
        );

    }


    predictionTableBody.innerHTML = "";


    try {

        const response = await fetch(
            "/api/predictions"
        );


        const data = await response.json();


        if (!response.ok) {

            throw new Error(
                data.error || "Không tải được dữ liệu dự đoán"
            );

        }


        predictionData =
            data.predictions || [];


        /* MODEL */

        if (predictionModel) {

            predictionModel.textContent =
                data.model || "Logistic Regression";

        }


        /* THRESHOLD */

        if (predictionThreshold) {

            predictionThreshold.textContent =
                formatPercent(
                    safeNumber(data.threshold) * 100
                );

        }


        /* DATE */

        if (predictionDate) {

            predictionDate.textContent =
                data.latest_data_date || "--";

        }


        /* FRESHNESS */

        if (predictionFreshness) {

            if (data.data_is_stale) {

                predictionFreshness.textContent =
                    `Dữ liệu cũ ${data.data_age_days} ngày`;

                predictionFreshness.classList.add(
                    "stale-text"
                );

            } else {

                predictionFreshness.textContent =
                    "Dữ liệu tương đối mới";

                predictionFreshness.classList.remove(
                    "stale-text"
                );

            }

        }


        if (predictionWarning) {

            predictionWarning.classList.toggle(
                "hidden",
                !data.data_is_stale
            );

        }


        /* TABLE */

        renderPredictionTable(
            predictionData
        );


        /* SELECT DEFAULT */

        if (
            predictionData.length > 0 &&
            !selectedPrediction
        ) {

            const defaultStock =
                predictionData.find(
                    item => item.symbol === initialSymbol
                ) || predictionData[0];


            selectedPrediction =
                defaultStock;


            renderStockAnalysis(
                defaultStock
            );


            updateActivePredictionRow(
                defaultStock.symbol
            );

        }


    } catch (error) {

        console.error(
            "Prediction API error:",
            error
        );


        if (predictionError) {

            predictionError.textContent =
                `Lỗi tải dự đoán: ${error.message}`;

            predictionError.classList.remove(
                "hidden"
            );

        }

    } finally {

        if (predictionLoading) {

            predictionLoading.classList.add(
                "hidden"
            );

        }

    }

}


/* =========================================================
   17. REFRESH PREDICTIONS
========================================================= */

if (refreshPredictionBtn) {

    refreshPredictionBtn.addEventListener(
        "click",
        function () {

            loadPredictions();

        }
    );

}


/* =========================================================
   18. TECHNICAL ANALYSIS
========================================================= */

function renderStockAnalysis(stock) {

    if (!stock) return;


    const panel = document.getElementById(
        "stock-analysis-panel"
    );


    if (panel) {

        panel.classList.remove(
            "hidden"
        );

    }


    /* BASIC */

    const title = document.getElementById(
        "analysisStockTitle"
    );

    if (title) {

        title.textContent =
            `Phân tích ${stock.symbol}`;

    }


    const probability = document.getElementById(
        "analysisProbability"
    );

    if (probability) {

        probability.textContent =
            formatPercent(stock.probability_up);

    }


    const threshold = document.getElementById(
        "analysisThreshold"
    );

    if (threshold) {

        threshold.textContent =
            formatPercent(stock.threshold);

    }


    /* SIGNAL */

    const signalBadge = document.getElementById(
        "analysisSignalBadge"
    );


    if (signalBadge) {

        if (stock.prediction === 1) {

            signalBadge.textContent =
                "Đạt ngưỡng mô hình";

            signalBadge.className =
                "signal-badge signal-positive";

        } else {

            signalBadge.textContent =
                "Chưa đạt ngưỡng";

            signalBadge.className =
                "signal-badge signal-neutral";

        }

    }


    /* RSI */

    const rsi = safeNumber(
        stock.rsi
    );


    setText(
        "analysisRSI",
        formatNumber(rsi)
    );


    let rsiText = "";


    if (rsi < 30) {

        rsiText =
            "RSI dưới 30: có thể ở vùng quá bán. Điều này không đảm bảo giá sẽ đảo chiều.";

    } else if (rsi > 70) {

        rsiText =
            "RSI trên 70: có thể ở vùng quá mua.";

    } else if (rsi >= 40 && rsi <= 60) {

        rsiText =
            "RSI ở vùng trung tính.";

    } else {

        rsiText =
            "RSI đang ở vùng trung gian.";

    }


    setText(
        "analysisRSIText",
        rsiText
    );


    /* MOVING AVERAGE */

    const price = safeNumber(
        stock.price
    );

    const ma5 = safeNumber(
        stock.ma5
    );

    const ma20 = safeNumber(
        stock.ma20
    );


    let maTrend = "";

    let maText = "";


    if (
        price > ma5 &&
        price > ma20
    ) {

        maTrend =
            "Trên MA5 & MA20";

        maText =
            "Giá đang cao hơn cả MA5 và MA20, xu hướng kỹ thuật ngắn hạn tương đối tích cực.";

    } else if (
        price < ma5 &&
        price < ma20
    ) {

        maTrend =
            "Dưới MA5 & MA20";

        maText =
            "Giá đang thấp hơn cả MA5 và MA20, xu hướng kỹ thuật ngắn hạn đang yếu.";

    } else {

        maTrend =
            "Xu hướng hỗn hợp";

        maText =
            "Giá đang nằm giữa MA5 và MA20, các tín hiệu trung bình động chưa đồng thuận.";

    }


    setText(
        "analysisMATrend",
        maTrend
    );

    setText(
        "analysisMAText",
        maText
    );


    /* MACD */

    const macd = safeNumber(
        stock.macd
    );

    const macdSignal = safeNumber(
        stock.macd_signal
    );


    setText(
        "analysisMACD",
        formatNumber(macd, 4)
    );


    let macdText = "";


    if (macd > macdSignal) {

        macdText =
            "MACD cao hơn Signal, động lượng có dấu hiệu tích cực.";

    } else {

        macdText =
            "MACD thấp hơn Signal, động lượng có dấu hiệu yếu.";

    }


    setText(
        "analysisMACDText",
        macdText
    );


    /* VOLUME */

    const volume = safeNumber(
        stock.volume
    );

    const volumeMA5 = safeNumber(
        stock.volume_ma5
    );


    setText(
        "analysisVolume",
        formatNumber(volume, 0)
    );


    let volumeText = "";


    if (volume > volumeMA5) {

        volumeText =
            "Khối lượng cao hơn trung bình 5 phiên.";

    } else {

        volumeText =
            "Khối lượng thấp hơn trung bình 5 phiên.";

    }


    setText(
        "analysisVolumeText",
        volumeText
    );


    /* =====================================================
       DECISION SUPPORT SCORE
    ===================================================== */

    let scoreML = 0;
    let scoreMA = 0;
    let scoreMACD = 0;
    let scoreRSI = 0;
    let scoreVolume = 0;
    let scoreFreshness = 0;


    /* ML: 30 */

    const probabilityUp = safeNumber(
        stock.probability_up
    );

    const thresholdValue = safeNumber(
        stock.threshold
    );


    if (
        probabilityUp >=
        thresholdValue + 5
    ) {

        scoreML = 30;

    } else if (
        probabilityUp >= thresholdValue
    ) {

        scoreML = 20;

    } else if (
        probabilityUp >=
        thresholdValue - 5
    ) {

        scoreML = 10;

    } else {

        scoreML = 0;

    }


    /* MA: 25 */

    if (
        price > ma5 &&
        price > ma20
    ) {

        scoreMA = 25;

    } else if (
        price > ma5 ||
        price > ma20
    ) {

        scoreMA = 12;

    } else {

        scoreMA = 0;

    }


    /* MACD: 15 */

    scoreMACD =
        macd > macdSignal
            ? 15
            : 0;


    /* RSI: 15 */

    if (
        rsi >= 40 &&
        rsi <= 65
    ) {

        scoreRSI = 15;

    } else if (
        rsi >= 30 &&
        rsi < 40
    ) {

        scoreRSI = 10;

    } else if (
        rsi > 65 &&
        rsi <= 70
    ) {

        scoreRSI = 8;

    } else {

        scoreRSI = 3;

    }


    /* VOLUME: 10 */

    scoreVolume =
        volume > volumeMA5
            ? 10
            : 5;


    /* FRESHNESS: 5 */

    let dataAge = null;


    if (stock.date) {

        const stockDate = new Date(
            stock.date
        );

        const today = new Date();


        dataAge = Math.floor(
            (
                today.getTime() -
                stockDate.getTime()
            ) /
            (1000 * 60 * 60 * 24)
        );

    }


    if (
        dataAge !== null &&
        dataAge <= 3
    ) {

        scoreFreshness = 5;

    } else if (
        dataAge !== null &&
        dataAge <= 7
    ) {

        scoreFreshness = 3;

    } else {

        scoreFreshness = 0;

    }


    /* TOTAL */

    const totalScore =
        scoreML +
        scoreMA +
        scoreMACD +
        scoreRSI +
        scoreVolume +
        scoreFreshness;


    /* =====================================================
       SCORE INTERPRETATION
    ===================================================== */

    let decisionStatus = "";

    let decisionDescription = "";


    if (totalScore >= 80) {

        decisionStatus =
            "Các yếu tố tương đối đồng thuận";

        decisionDescription =
            "Nhiều yếu tố đang hỗ trợ tín hiệu. Tuy nhiên, điểm số không phải xác suất sinh lời và cần xem xét rủi ro thị trường.";

    } else if (totalScore >= 60) {

        decisionStatus =
            "Tín hiệu ở mức trung bình";

        decisionDescription =
            "Một số yếu tố đang hỗ trợ nhưng vẫn tồn tại các yếu tố chưa đồng thuận. Nên tiếp tục đánh giá thêm.";

    } else if (totalScore >= 40) {

        decisionStatus =
            "Tín hiệu yếu";

        decisionDescription =
            "Các yếu tố hỗ trợ chưa đủ rõ ràng. Cần thận trọng và theo dõi thêm.";

    } else {

        decisionStatus =
            "Chưa có sự đồng thuận";

        decisionDescription =
            "Các yếu tố hiện tại chưa tạo được tín hiệu hỗ trợ rõ ràng.";

    }


    /* =====================================================
       RISK ANALYSIS
    ===================================================== */

    let risks = [];


    if (
        stock.probability_up <
        stock.threshold
    ) {

        risks.push(
            "Xác suất ML chưa đạt ngưỡng đã thiết lập."
        );

    }


    if (
        price < ma5 &&
        price < ma20
    ) {

        risks.push(
            "Giá đang nằm dưới cả MA5 và MA20."
        );

    }


    if (
        macd < macdSignal
    ) {

        risks.push(
            "MACD thấp hơn đường Signal."
        );

    }


    if (
        rsi > 70
    ) {

        risks.push(
            "RSI cao, có thể đang ở vùng quá mua."
        );

    }


    if (
        rsi < 30
    ) {

        risks.push(
            "RSI thấp. Vùng quá bán không đảm bảo giá sẽ đảo chiều."
        );

    }


    if (
        volume < volumeMA5
    ) {

        risks.push(
            "Khối lượng thấp hơn mức trung bình 5 phiên."
        );

    }


    if (
        dataAge !== null &&
        dataAge > 7
    ) {

        risks.push(
            `Dữ liệu đã cũ ${dataAge} ngày.`
        );

    }


    if (risks.length === 0) {

        risks.push(
            "Không phát hiện cảnh báo cơ bản từ các chỉ báo đang sử dụng."
        );

    }


    let riskLevel = "Thấp";


    if (risks.length >= 4) {

        riskLevel = "Cao";

    } else if (risks.length >= 2) {

        riskLevel = "Trung bình";

    }


    /* =====================================================
       DISPLAY SCORE
    ===================================================== */

    setText(
        "decisionScore",
        totalScore
    );


    const scoreFill = document.getElementById(
        "decisionScoreFill"
    );


    if (scoreFill) {

        scoreFill.style.width =
            `${totalScore}%`;

    }


    setText(
        "decisionStatus",
        decisionStatus
    );


    setText(
        "decisionRisk",
        `Rủi ro: ${riskLevel}`
    );


    setText(
        "decisionExplanation",
        decisionDescription
    );


    /* SCORE BREAKDOWN */

    setText(
        "scoreML",
        `${scoreML}/30`
    );

    setText(
        "scoreMA",
        `${scoreMA}/25`
    );

    setText(
        "scoreMACD",
        `${scoreMACD}/15`
    );

    setText(
        "scoreRSI",
        `${scoreRSI}/15`
    );

    setText(
        "scoreVolume",
        `${scoreVolume}/10`
    );

    setText(
        "scoreFreshness",
        `${scoreFreshness}/5`
    );


    /* RISK LIST */

    const riskList = document.getElementById(
        "decisionRiskList"
    );


    if (riskList) {

        riskList.innerHTML = "";


        risks.forEach(risk => {

            const li = document.createElement(
                "li"
            );

            li.textContent = risk;

            riskList.appendChild(li);

        });

    }


    /* SUMMARY */

    const summaryList = document.getElementById(
        "analysisSummaryList"
    );


    if (summaryList) {

        summaryList.innerHTML = "";


        const summaries = [

            `Xác suất tăng do ML: ${formatPercent(probabilityUp)}.`,

            `Decision Support Score: ${totalScore}/100.`,

            `RSI: ${formatNumber(rsi)}.`,

            maText,

            macdText,

            volumeText,

            `Ngày dữ liệu: ${stock.date || "--"}.`

        ];


        summaries.forEach(summary => {

            const li = document.createElement(
                "li"
            );

            li.textContent = summary;

            summaryList.appendChild(li);

        });

    }

}


/* =========================================================
   19. HELPER SET TEXT
========================================================= */

function setText(elementId, value) {

    const element = document.getElementById(
        elementId
    );


    if (element) {

        element.textContent =
            value ?? "--";

    }

}


/* =========================================================
   20. INITIALIZE DASHBOARD
========================================================= */

updateStockDetails(
    initialLatestStock,
    initialSymbol
);


updateCharts(
    initialData
);


updateActiveStock(
    initialSymbol
);


loadPredictions();


console.log(
    "Stock Market ML Dashboard initialized successfully."
);
