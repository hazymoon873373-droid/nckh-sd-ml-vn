# Kế hoạch Nghiên cứu Khoa học (NCKH)
**Đề tài:** Dự đoán mức độ ưa thích cổ phiếu ("Yêu / Không yêu — Thích / Không thích") và xu hướng dịch chuyển dòng tiền trên thị trường chứng khoán Việt Nam bằng Machine Learning kết hợp Stochastic Dominance (FSD, SSD, Almost SD)

**Khung phương pháp (Framework Selection — Decomposition + PICO Hybrid):**
- **Population / Scope:** Cổ phiếu niêm yết trên HOSE (ưu tiên rổ VN100 / VN30 để đảm bảo thanh khoản và dữ liệu sạch trên Yahoo Finance + bảng điện tử) và dòng tiền (khối ngoại, tự doanh, tổ chức/cá nhân, thanh khoản/OFI).
- **Problem:** Dự báo điểm kỳ vọng lợi suất $\mathbb{E}[R_{t+1} \mid \mathcal{F}_t]$ bằng ML truyền thống bỏ qua rủi ro đuôi (fat tails), độ lệch (skewness) và yêu cầu phải giả định hàm hữu dụng (utility function) cụ thể.
- **Intervention / Solution:**
  1. Dùng **Distributional / Quantile Machine Learning** (Quantile Regression Forests, LightGBM Quantile, hoặc Two-stage Quantile Neural Network + Cubic B-Spline) để dự báo **toàn bộ CDF tương lai** $\hat{F}_{i,t+1}(r \mid \mathcal{F}_t)$ của từng cổ phiếu/nhóm ngành.
  2. Áp dụng kiểm định **FSD, SSD, và Almost SD (AFSD/ASSD)** trên các CDF dự báo so với benchmark (VN-Index / VN30) hoặc theo cặp để gán nhãn/điểm **"Ưa thích hợp lý" (Rational Preference Score)**.
  3. Kiểm chứng khả năng dự báo **dòng tiền đang/sẽ đi đâu** (net foreign flow, abnormal turnover, order flow imbalance, ETF/fund flows).
- **Comparison:** So sánh với (a) Mean-Variance Markowitz, (b) Point-prediction ML (XGBoost/LSTM dự báo mean return), (c) Pure historical SD (không dùng ML dự báo CDF tương lai), (d) VN-Index Buy & Hold.
- **Outcome:** Hiệu quả danh mục ngoài mẫu (Sharpe, Sortino, Omega, Max Drawdown), độ chính xác phân phối (CRPS / Pinball Loss), và khả năng giải thích/dự báo dòng tiền ròng kỳ $t+1$.

## Khoảng trống Nghiên cứu (Research Gap)
1. Trên thế giới đã có:
   - *Distributional ML dự báo CDF cổ phiếu*: Baruník, Hronec & Tobek (2026, *IJF* Q1), Meinshausen (2006, *JMLR* Q1), Chernozhukov et al. (2021, *PNAS* Q1).
   - *Tối ưu hóa & xếp hạng danh mục theo SSD / Almost SD*: Leshno & Levy (2002, *MS* Q1), Tzeng et al. (2013, *MS* Q1), Luo, Chen & Jaillet (2025, *MS* Q1), Cesarone & Puerto (2025, *EJOR* Q1), Babapour-Azar & Khanjani-Shiraz (2024, *FRL* Q1).
   - *Dòng tiền & Stochastic Dominance*: Clark & Kassimatis (2013, *RQFA* Q2) cho dòng vốn quốc tế bằng MCSD.
2. Tại Việt Nam:
   - Các nghiên cứu ML ở Việt Nam mới chỉ dự báo **điểm giá / xu hướng tăng-giảm** (Phuoc et al., 2024, *HSSC* Q1; Thanh Nhon et al., 2025, *PLOS ONE* Q1) hoặc phân tích **hành vi/dòng tiền ngoại bằng kinh tế lượng truyền thống** (Vo, 2017, *IRFA* Q1; Vo & Phan, 2019, *PBFJ* Q1; Phan et al., 2023, *QREF* Q2).
   - **Chưa có công trình Q1/Q2 nào** kết hợp **Distributional ML dự báo CDF tương lai $\rightarrow$ Almost Stochastic Dominance $\rightarrow$ dự đoán mức độ ưa thích cổ phiếu & dịch chuyển dòng tiền tại thị trường cận biên/mới nổi Việt Nam** (nơi tỷ trọng nhà đầu tư cá nhân cao, lệch phải/trái mạnh, biên độ trần/sàn $\pm 7\%$ trên HOSE tạo ra phân phối cụt đuôi đặc thù).

## Các Giả thuyết Kiểm định (Falsifiable Hypotheses)
- **H1 (Distributional Advantage):** Dự báo CDF tương lai bằng Quantile ML (QRF / QNN / LightGBM-Quantile) đạt Pinball Loss & CRPS thấp hơn đáng kể so với giả định phân phối chuẩn lịch sử hoặc GARCH truyền thống trên dữ liệu HOSE.
- **H2 (ASD vs Strict FSD/SSD):** Trong thực nghiệm cổ phiếu Việt Nam, FSD và SSD thuần túy quá khắt khe (tạo tập không so sánh được lớn), trong khi **Almost Stochastic Dominance (AFSD/ASSD)** với ngưỡng vi phạm $\varepsilon$ hiệu chỉnh xác định được rõ ràng nhóm cổ phiếu "được ưa thích" (Preferred) và "không được ưa thích" (Non-preferred).
- **H3 (Capital Flow Alignment):** Cổ phiếu/nhóm ngành được ML-CDF xếp hạng **trội hơn theo ASSD** ở kỳ $t$ thu hút **dòng tiền ròng dương** (khối ngoại, thanh khoản bất thường, dòng tiền chủ động từ bảng điện tử) ở kỳ $t+1$ mạnh hơn nhóm bị chi phối, trừ các cổ phiếu mang tính "vé số" (lottery-like stocks theo Fong, 2013) hút dòng tiền cá nhân ngắn hạn dù bị SSD loại.
- **H4 (Out-of-sample Portfolio):** Chiến lược phân bổ vốn theo ML-predicted CDF + ASSD đạt tỷ số Sortino/Omega cao hơn và Max Drawdown thấp hơn so với chiến lược Mean-Variance và Point-ML truyền thống (sau khi tính phí giao dịch & thuế bán $0{,}1\%$ + T+2.5 tại Việt Nam).

## Nguồn dữ liệu thực nghiệm đề xuất
- **Yahoo Finance (`yfinance`):** Mã cổ phiếu Việt Nam đuôi `.VN` (ví dụ `VCB.VN`, `FPT.VN`, `HPG.VN`, `VNM.VN`, `^VNINDEX` hoặc ETF `E1VFVN30.VN`) — lưu ý kiểm tra cổ tức/chia tách (Adjusted Close) và đối chiếu nguồn Việt Nam vì Yahoo Finance đôi khi thiếu/lệch dữ liệu HOSE.
- **Bảng điện tử & Dữ liệu thị trường Việt Nam:** `vnstock` / FireAnt / SSI iBoard / CafeF / FiinQuant / Vietstock cho giao dịch khối ngoại ròng, tự doanh, khối lượng mua chủ động vs bán chủ động (Order Flow Imbalance), vốn hóa và báo cáo tài chính quý.
