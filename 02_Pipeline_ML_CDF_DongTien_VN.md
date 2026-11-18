# Pipeline Kỹ thuật: Machine Learning $\to$ CDF Tương lai $\to$ Stochastic Dominance $\to$ Dòng tiền Cổ phiếu VN

Tài liệu này chuyển hóa phần ghi chú nửa dưới trang viết tay:
`ML -> prediction -> { CDF trong tương lai ; SD } -> ứng dụng: dòng tiền đang / sẽ đi đâu | data: Yahoo finance, bảng đ/tử`

---

## 1. Tại sao không dùng ML dự báo 1 con số (Point Prediction)?

Các mô hình ML truyền thống (Linear Regression, XGBoost MSE, LSTM MSE) dự báo giá trị kỳ vọng có điều kiện:

\[
\hat{y}_{i,t+1} = \hat{\mathbb{E}}[R_{i,t+1} \mid X_{i,t}]
\]

Một con số duy nhất **không có hình dạng phân phối**, nên **không thể áp dụng FSD, SSD hay ASD**. Để nối ML với Stochastic Dominance, bước bắt buộc là dùng **Distributional / Quantile Machine Learning** để dự báo **hàm phân phối tích lũy tương lai** $\hat{F}_{i,t+1}(r \mid X_{i,t}, Z_t)$.

---

## 2. Ba phương pháp Q1 tốt nhất để ML sinh ra "CDF trong tương lai"

### Phương pháp A (Khuyên dùng làm Baseline mạnh & Dễ giải thích): Quantile Regression Forests (QRF) — Meinshausen (2006, *JMLR* Q1)
- **Nguyên lý:** Rừng cây ngẫu nhiên không chỉ lấy trung bình ở các lá (leaves), mà giữ lại toàn bộ phân phối có trọng số $w_j(x)$ của các quan sát trong lá:

\[
\hat{F}(r \mid X = x) = \sum_{j=1}^{n} w_j(x)\,\mathbf{1}_{\{R_j \le r\}}
\]

- **Ưu điểm cho NCKH sinh viên:** Có sẵn qua thư viện Python (`quantile-forest` hoặc tự tính trọng số lá từ `RandomForestRegressor`), không cần GPU lớn, tự động đảm bảo **CDF đơn điệu không giảm** (zero quantile crossing), chạy cực ổn trên dữ liệu bảng cổ phiếu Việt Nam.

### Phương pháp B (Chuẩn Q1 SOTA Tài chính): Two-Stage Quantile Neural Network + Cubic B-Spline — Baruník, Hronec & Tobek (2026, *IJF* Q1)
- **Nguyên lý:**
  - **Bước 1 (Loss hàm Pinball):** Mạng nơ-ron đa đầu ra dự báo đồng thời $K$ phân vị $\tau_1 < \tau_2 < \dots < \tau_K$ (ví dụ 19 hoặc 37 phân vị từ $0{,}01$ đến $0{,}99$) bằng hàm mất mát bất đối xứng:

\[
\rho_{\tau}(\xi) = \mathbf{1}_{\{\xi \ge 0\}}\,\tau\,\xi + \mathbf{1}_{\{\xi < 0\}}\,(\tau - 1)\,\xi
\]

  - **Bước 2 (Chuẩn hóa biến động):** Giai đoạn 1 dự báo phân vị lợi suất chuẩn hóa theo độ biến động cắt ngang $\sigma_t$ (EWMA với $\lambda = 0{,}94$); Giai đoạn 2 nhân lại hệ số điều chỉnh biến động thị trường $\hat{\sigma}_t^M$ để khử look-ahead bias và nắm bắt chế độ biến động (volatility regime).
  - **Bước 3 (Nội suy Cubic B-Spline):** Nội suy mượt từ các điểm phân vị $\bigl(\hat{Q}_i(\tau_k), \tau_k\bigr)_{k=1}^K$ thành đường CDF liên tục $\hat{F}_{i,t+1}(r)$ trên lưới dày bằng `scipy.interpolate.splrep` / `BSpline` (nhớ kẹp đơn điệu nếu có quantile crossing!).

### Phương pháp C (Bổ trợ / Nâng cao có bảo chứng toán học): LightGBM Quantile + Conformal Predictive Distributions (Vovk et al., 2019 *ML* Q1; Chernozhukov et al., 2021 *PNAS* Q1)
- **Nguyên lý:** Train LightGBM với `objective='quantile'` cho lưới $\tau_k$, sau đó hiệu chỉnh ngoài mẫu bằng Conformal Calibration trên tập hiệu chuẩn (calibration split theo thời gian) để đảm bảo CDF dự báo có **độ phủ xác suất chuẩn (calibrated CDF)** khi thị trường Việt Nam biến động mạnh (như giai đoạn 2022 hoặc tin vĩ mô đột ngột).

---

## 3. Pipeline 5 Bước từ Dữ liệu đến "Dòng tiền đang / sẽ đi đâu"

### Bước 1: Thu thập & Làm sạch Dữ liệu Việt Nam (`Yahoo Finance` + `Bảng điện tử`)
1. **Nhóm biến Giá & Kỹ thuật (từ Yahoo Finance `.VN` / `vnstock`):**
   - Lợi suất trễ ($R_{t-1}, R_{t-5}, R_{t-21}, R_{t-63}$), momentum, đảo chiều ngắn hạn, EWMA volatility ($\lambda = 0{,}94$), độ lệch quá khứ (skewness), max daily return (proxy cổ phiếu vé số — Fong, 2013).
   - **Cảnh báo kỹ thuật quan trọng:** Giá trên Yahoo Finance cho mã `.VN` đôi khi chưa điều chỉnh cổ tức/chia tách chuẩn hoặc thiếu ngày giao dịch cận biên; **bắt buộc đối chiếu chéo** với nguồn Việt Nam (`vnstock`, CafeF, SSI iBoard, FireAnt).
2. **Nhóm biến Dòng tiền & Bảng điện tử (từ Bảng điện tử HOSE/HNX):**
   - **Khối ngoại ròng (`Foreign Net Flow`):** $\text{FNF}_{i,t} = \frac{\text{BuyVal}_{\text{foreign}} - \text{SellVal}_{\text{foreign}}}{\text{TotalVal}_{i,t}}$ (theo Vo, 2017, *IRFA* Q1).
   - **Mất cân bằng lệnh chủ động (`Order Flow Imbalance` / Net Active Buy):** Tỷ trọng khối lượng khớp lệnh chủ động mua trừ chủ động bán (nếu có dữ liệu tick/intraday từ bảng điện tử) hoặc proxy *Money Flow Index (MFI)* / *Chaikin Money Flow (CMF)* / *Amihud Illiquidity*.
   - **Thanh khoản bất thường (`Abnormal Turnover`):** Tỷ số giá trị giao dịch ngày $t$ chia trung bình 20 phiên.
3. **Nhóm biến Cơ bản & Vĩ mô (nếu mở rộng khung tuần/tháng):** P/E, P/B, ROE, tăng trưởng lợi nhuận quý, lãi suất liên ngân hàng VN, tỷ giá USD/VND, tăng trưởng tín dụng.

### Bước 2: Dự báo Phân phối Lợi suất Tương lai (Out-of-Sample Walk-Forward)
- Chia dữ liệu bằng **Expanding / Rolling Window** theo thời gian (tuyệt đối **không** dùng `train_test_split` ngẫu nhiên để tránh data leakage!).
- Tại thời điểm $t$, dự báo CDF lợi suất kỳ $t+1$ (ví dụ 5 phiên / T+5 hoặc 22 phiên / 1 tháng) cho từng cổ phiếu $i$ trong rổ (ví dụ VN30 hoặc Top 100 cổ phiếu thanh khoản nhất HOSE) và cho chỉ số chuẩn VN-Index / VN30.

### Bước 3: Chấm điểm Ưa thích ("Yêu / Không yêu") qua FSD, SSD, Almost SD
Trên lưới điểm chung $\{r_1, \dots, r_M\}$ của $\hat{F}_{i,t+1}(r)$ và $\hat{F}_{\text{Bench}, t+1}(r)$:
- Tính **tỷ lệ vi phạm $\varepsilon_1^{*}(i \succ \text{Bench})$** (AFSD) và **$\varepsilon_2^{*}(i \succ \text{Bench})$** (ASSD theo Tzeng et al., 2013), cùng điều kiện $\hat{\mathbb{E}}[R_{i,t+1}] \ge \hat{\mathbb{E}}[R_{\text{Bench}, t+1}]$.
- Gán nhãn hoặc xếp hạng từng cổ phiếu / từng nhóm ngành:
  - **Nhóm "Yêu / Ưa thích" (ASD-Preferred):** $\varepsilon_2^{*} \le \varepsilon_{\text{ngưỡng}}$ (khảo sát độ nhạy $\varepsilon \in \{0{,}01; 0{,}03; 0{,}059; 0{,}10\}$).
  - **Nhóm "Không yêu" (Dominated):** bị benchmark hoặc đa số cổ phiếu khác chi phối theo ASSD.

### Bước 4: Liên kết với "Dòng tiền đang / sẽ đi đâu" (Capital Flow Analysis)
Đây là đóng góp thực tiễn sáng giá nhất của đề tài! Cần kiểm định hai chiều rõ ràng:
1. **Dòng tiền ĐANG đi đâu ($t$):**
   - Đo dòng tiền thực tế hiện tại theo cổ phiếu/ngành: Net Foreign Flow, Net Proprietary Flow (tự doanh), Abnormal Money Flow / Turnover Share ($\text{Vol}_{i,t} / \sum_j \text{Vol}_{j,t}$).
   - Dùng **Test Stochastic Dominance (Linton et al., 2005 / Davidson & Duclos, 2000)** để kiểm tra xem nhóm cổ phiếu **đang hút tiền mạnh** có thực sự **SSD/ASSD-dominate** nhóm bị rút tiền không? (Hay dòng tiền cá nhân ở Việt Nam đang chạy vào "lottery stocks" rủi ro cao, bị ASSD từ chối?).
2. **Dòng tiền SẼ đi đâu ($t+1$):**
   - Kiểm định xem **Điểm trội ngẫu nhiên dự báo từ ML ($S_{\text{ASD}, i, t}$)** có dự báo được **sự dịch chuyển dòng tiền ròng kỳ sau ($\Delta \text{Flow}_{i, t+1}$)** qua hồi quy Fama-MacBeth / Panel Fixed Effects hoặc mô hình phân loại ML hay không:

\[
\text{Flow}_{i,t+1} = \alpha_i + \gamma_t + \beta_1\, S_{\text{ASD}, i, t}^{\text{ML}} + \beta_2\, \text{Flow}_{i,t} + \mathbf{\delta}'\,\text{Controls}_{i,t} + \nu_{i,t+1}
\]

### Bước 5: Đánh giá Kinh tế & Kiểm chứng Giao dịch (Backtest Thực tế tại Việt Nam)
- Xây dựng danh mục Long-Only (vì VN chưa cho phép bán khống cổ phiếu cơ sở đại trà!) gồm Top-K cổ phiếu "Yêu / ASD-Preferred" hoặc tối ưu hóa tỷ trọng theo **ASSD / CSESD Linear Programming** (Cesarone & Puerto, 2025; Luo et al., 2025).
- Khấu trừ đầy đủ **phí môi giới + thuế bán $0{,}1\%$** và ràng buộc thanh khoản **T+2.5** (cổ phiếu mua ngày T chiều T+2 mới về tài khoản để bán).
- Đánh giá các chỉ số chuẩn Q1: **Annualized Return, Volatility, Sharpe, Sortino, Omega Ratio, Max Drawdown (MDD)**, và **Pinball Loss / CRPS** cho độ chính xác của CDF dự báo.
