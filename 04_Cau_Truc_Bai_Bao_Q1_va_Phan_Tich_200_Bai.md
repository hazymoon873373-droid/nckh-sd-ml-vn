# Giải phẫu Cấu trúc Bài Nghiên cứu Khoa học & Tổng hợp Scan 376 Bài Q1

> Tài liệu này tổng hợp hai phần phục vụ trực tiếp cho buổi làm NCKH ngày mai:
> 1. **Hiểu cấu trúc một bài nghiên cứu khoa học chuẩn Q1** (giải thích theo tư duy người thường + đối chiếu 3 bài Q1 thực tế trong `papers/` + ánh xạ vào khung Đề cương Khoa Kinh tế HCMUTE).
> 2. **Báo cáo scan 376 bài tạp chí Q1** (từ 24 tạp chí Q1 qua Crossref API, khử trùng lặp theo DOI trong `q1_200plus_corpus.csv`), phân cụm theo 6 nhánh kiến thức và bảng kế thừa biến/phương pháp.

---

## PHẦN A: Cấu trúc 1 Bài Nghiên cứu Khoa học (Hiểu dễ nhất)

### 1. Tư duy người thường: Một bài báo Q1 thực chất là "6 câu trả lời nối đuôi nhau"

Đừng nhìn cấu trúc học thuật như những chữ tiếng Anh khô khan (*Abstract, Literature Review, Methodology...*). Hãy tưởng tượng em đang đi thuyết trình gọi vốn trước một Hội đồng khó tính. Một bài báo Q1 luôn trả lời đúng **6 câu hỏi theo đúng trình tự tâm lý người đọc**:

| Phần trong bài báo Q1 | Câu hỏi đời thường nó trả lời | Tương ứng trong Đề cương/Luận văn Khoa Kinh tế HCMUTE |
|---|---|---|
| **0. Title & Abstract** (Tiêu đề & Tóm tắt) | *"Tóm lại bài này làm gì, bằng dữ liệu gì và kết quả ra sao trong 200 chữ?"* | Bìa + Tóm tắt nghiên cứu |
| **1. Introduction** (Giới thiệu) | *"Vì sao vấn đề này đáng tiền? Thế giới làm tới đâu rồi, còn lỗ hổng nào mà tôi phải nhảy vào làm?"* | **Chương 1: Tổng quan nghiên cứu** (1.1 Lý do, 1.2 Mục tiêu, 1.3 Phương pháp, 1.4 Đối tượng & Phạm vi, 1.5 Kết cấu) |
| **2. Literature Review & Hypotheses** (Tổng quan tài liệu & Giả thuyết) | *"Các bậc tiền bối Q1 đã chứng minh điều gì? Từ đó tôi đặt ra những giả thuyết kiểm chứng nào?"* | **Chương 2: Cơ sở lý thuyết và Mô hình** (2.1 Lý thuyết nền, 2.2 Tổng quan nghiên cứu trước, 2.3 Mô hình & Giả thuyết) |
| **3. Data & Methodology** (Dữ liệu & Phương pháp) | *"Tôi lấy số liệu ở đâu, làm sạch ra sao, biến số công thức thế nào để người khác làm lại y hệt được?"* | **Chương 3: Thiết kế nghiên cứu** (3.1 Quy trình, 3.2 Bảng thang đo/biến kế thừa từ các bài trước) |
| **4. Empirical Results & Robustness** (Kết quả thực nghiệm & Kiểm định bền) | *"Chạy thực tế có thắng cách cũ không? Đổi tham số, đổi giai đoạn khủng hoảng có bị sập không?"* | **Chương 4: Kết quả nghiên cứu & Thảo luận** |
| **5. Conclusion & Implications** (Kết luận & Hàm ý) | *"Nhà đầu tư/nhà quản lý dùng kết quả này làm gì ngoài đời? Bài còn hạn chế gì cho người sau làm tiếp?"* | **Chương 5: Kết luận & Hàm ý quản trị** + Tài liệu tham khảo (APA) |

---

### 2. Soi Cấu trúc Thật của 3 Bài Q1 Đang Có Trong Thư mục `papers/`

Để thấy người viết Q1 chia mục cụ thể ra sao, hãy nhìn 3 bài Q1 thật với 3 phong cách khác nhau (từ dễ đến chuẩn quốc tế):

1. **Mẫu 1 — Thực nghiệm ML trên sàn HOSE Việt Nam (16 trang, dễ theo nhất cho NCKH sinh viên):**
   - *Bài mẫu:* `Thanh Nhon, Do-Thi & Nguyen-Trang (2025, PLOS ONE Q1)` — [PLOS_ONE_2025_Vietnam_HOSE_ML_DEA.pdf](file:///C:/Users/hazym/.gemini/antigravity/scratch/nckh-sd-ml-vn/papers/PLOS_ONE_2025_Vietnam_HOSE_ML_DEA.pdf)
   - *Cấu trúc thật:*
     1. `Introduction` (p.1–2): Đặt bối cảnh thị trường VN thanh khoản thấp/biến động mạnh $\to$ nêu 3 đóng góp rõ ràng bằng 3 gạch đầu dòng.
     2. `Literature review` (p.2–4): Chia đúng 3 tiểu mục theo 3 thành phần của đề tài (ML dự báo lợi suất; DEA; Auto-feature engineering).
     3. `Data and methodology` (p.4–9): Trình bày bảng biến $X_1 \dots X_{12}, H, Y$ (Bảng 1, p.5), giải thích từng mô hình (GLM, DNN, DT, RF, GBT), cách chia Train (2019–2023) / Test (2024) và 3 câu hỏi nghiên cứu.
     4. `Result and discussion` (p.9–13): Báo cáo Cross-validation $\to$ Kết quả trên tập Test (RMSE, MAE, MAPE) $\to$ Giải thích ý nghĩa kinh tế (ví dụ giải thích tại sao biên lợi nhuận trước thuế phi tuyến với lợi suất cổ phiếu bất động sản như NLG).
     5. `Conclusion` (p.13) & `References` (p.14–16).

2. **Mẫu 2 — Tối ưu hóa Stochastic Dominance & Almost SSD (40 trang, chuẩn Operations Research / Finance):**
   - *Bài mẫu:* `Cesarone & Puerto (2025, EJOR Q1)` — [Cesarone_Puerto_2024_EJOR2025_ApproxSD_EnhancedIndexation.pdf](file:///C:/Users/hazym/.gemini/antigravity/scratch/nckh-sd-ml-vn/papers/Cesarone_Puerto_2024_EJOR2025_ApproxSD_EnhancedIndexation.pdf)
   - *Cấu trúc thật:*
     1. `Introduction` (p.2–4): So sánh Index Tracking vs Enhanced Indexation $\to$ nêu lý do FSD/SSD quá chặt và khoảng trống trong Almost SD.
     2. `Preliminary concepts on exact and approximate SD criteria` (p.4–8): Định nghĩa chính xác ZSD, FSD, SSD, ASSD, rời rạc hóa theo kịch bản và liên hệ với $\operatorname{CVaR}$.
     3. `Enhanced Indexation based on SD` & `Theoretical Analysis` (p.9–19): Xây dựng mô hình toán $\to$ chứng minh định lý $\to$ ví dụ minh họa bằng số nhỏ để người đọc hiểu.
     4. `Empirical Analysis` (p.20–31): Mô tả 4 bộ dữ liệu chuẩn (FTSE100, NASDAQ100, SP500, FF49), thiết lập **Rolling Window** (6 tháng in-sample / 1 tháng out-of-sample), rồi chấm **11 thước đo kinh tế** (Sharpe, Sortino, Max Drawdown, Ulcer Index, Omega, Rachev, Jensen's Alpha, VaR 1%, Appraisal, Information Ratio, Turnover).
     5. `Conclusions` (p.32–36) & `Appendix` (p.37–40).

3. **Mẫu 3 — Distributional ML dự báo CDF cổ phiếu (83 trang, chuẩn Top Finance / Econometrics):**
   - *Bài mẫu:* `Baruník, Hronec & Tobek (2026, IJF Q1)` — [Barunik_Hronec_Tobek_2024_IJF2026_QuantileNN_CDFs.pdf](file:///C:/Users/hazym/.gemini/antigravity/scratch/nckh-sd-ml-vn/papers/Barunik_Hronec_Tobek_2024_IJF2026_QuantileNN_CDFs.pdf)
   - *Cấu trúc thật:*
     1. `Introduction` (p.2–4): Tại sao dự báo điểm (mean) không đủ $\to$ cần dự báo toàn bộ phân phối/CDF.
     2. `Two-stage quantile neural networks` (p.4–11): Kiến trúc mạng 2 tầng (Stage I phân vị chuẩn hóa + Stage II điều chỉnh biến động thị trường) $\to$ Pinball Loss $\to$ Cubic B-Spline ra CDF mượt.
     3. `Empirical results` (p.12–27): Dữ liệu & 194 biến đặc trưng (153 anomalies + 18 stock volatilities + 23 biến thị trường) $\to$ Expanding Window không rò rỉ dữ liệu $\to$ Độ chính xác phân vị ngoài mẫu $\to$ Ứng dụng định giá tài sản cắt ngang.
     4. `Conclusion` (p.28) & `Appendices` (p.38–83): Đầy đủ danh sách biến, so sánh mô hình thay thế (FNN, GARCH, Tree-based ML), kiểm định siêu tham số và mô phỏng.

---

## PHẦN B: Báo cáo Scan 376 Bài Q1 Qua Crossref API

### 1. Thống kê Corpus Đã Scan ([q1_200plus_corpus.csv](file:///C:/Users/hazym/.gemini/antigravity/scratch/nckh-sd-ml-vn/q1_200plus_corpus.csv))

- **Tổng số bài Q1 duy nhất (khử trùng lặp theo DOI):** **376 bài** thuộc **24 tạp chí SJR Q1** (đối chiếu từ danh sách ISSN Q1 chuẩn hóa).
- **Phân bổ theo 6 nhánh chủ đề:**
  1. `SD_Theory_Tests` (Lý thuyết SD, kiểm định & tối ưu): **70 bài**
  2. `ML_Distributional_CDF` (Hồi quy phân vị, ML dự báo phân phối/CDF): **70 bài**
  3. `Investor_Preference_Flows` (Tâm lý, ưa thích nhà đầu tư, dòng tiền/lệnh): **68 bài**
  4. `ML_Stock_Portfolio` (Machine Learning dự báo cổ phiếu & tối ưu danh mục): **64 bài**
  5. `Vietnam_Emerging_Markets` (Thị trường mới nổi & Việt Nam): **61 bài**
  6. `Almost_SD` (Almost Stochastic Dominance & thư giãn điều kiện SSD): **43 bài**
- **Phân bổ theo giai đoạn thời gian:**
  - $\le 2005$: **77 bài** (đặt nền móng FSD/SSD, kiểm định phi tham số, Almost SD gốc).
  - $2006\text{–}2015$: **81 bài** (bùng nổ hành vi/tâm lý nhà đầu tư, hiệu chỉnh ASSD 2013, kiểm định SD qua quantile).
  - $2016\text{–}2020$: **71 bài** (ML tài chính, dòng vốn ngoại Việt Nam, SD spanning tests).
  - $2021\text{–}2026$: **147 bài** (Chiếm $39{,}1\%$ tổng corpus — bùng nổ **Distributional ML, Deep RL/NN + SSD, ASSD tối ưu hóa, và ML trên sàn HOSE Việt Nam**).
- **Top Tạp chí Q1 đóng góp nhiều nhất trong corpus:**
  *Finance Research Letters* (43), *Emerging Markets Review* (40), *Management Science* (35), *Expert Systems with Applications* (25), *Journal of Banking & Finance* (25), *International Journal of Forecasting* (24), *International Review of Financial Analysis* (20), *The Journal of Finance* (19), *Journal of Econometrics* (19), *Pacific-Basin Finance Journal* (19), *European Journal of Operational Research* (18), *PLOS ONE* (16), *Journal of Empirical Finance* (14), *JFQA* (12), *JFE* (10), *RFS* (8), *Quantitative Finance* (8), *Econometrica* (6), *REStud* (5), *Operations Research* (3), *JFM* (3), *EAAI* (2), *JBES* (2).
- **Phân tích chéo từ `cross_search_aggregator.py`:**
  - **Nhóm tác giả xuất hiện lặp nhiều nhất:** `Post, Thierry` (14 lần trên 2 nhánh), `Kopa, Miloš` (7 lần trên 2 nhánh), `Levy, Haim` (7 lần trên 2 nhánh), `Tzeng, Larry Y.` (5 lần), `Rentz, William F.` (4 lần trên 2 nhánh).
  - **Bài giao thoa đa nhánh (3 nhánh):** `Wang, Zhiguang (2025)` — *Machine learning for stock return prediction: Transformers or simple neural networks* (*Finance Research Letters*, DOI `10.1016/j.frl.2025.108783`), xuất hiện đồng thời ở `ML_Distributional_CDF`, `ML_Stock_Portfolio`, và `Vietnam_Emerging_Markets`.
  - **Trong 376 bài Q1, chỉ có đúng 6 bài nhắc trực tiếp tới Việt Nam** (chủ yếu về sở hữu nước ngoài/tổ chức, biến động và dự báo điểm lợi suất bằng ML trên HOSE) — **chưa có bài nào ghép cặp ML dự báo CDF $\to$ Almost SD $\to$ dòng tiền tại Việt Nam**. Đây là bằng chứng định lượng cực mạnh để em viết mục **1.1 Lý do chọn đề tài** và **2.2 Khoảng trống nghiên cứu**!

---

### 2. Tổng hợp 6 Cụm Tri thức Từ 376 Bài Q1 (Lắp thẳng vào từng Chương bài NCKH)

1. **Cụm 1 — Lý thuyết Nền về Lựa chọn dưới Rủi ro & SD (`Ch. 2.1`):**
   Đi từ Expected Utility đến FSD/SSD (Levy, 1992, *Management Science*), Marginal Conditional Stochastic Dominance để đo tác động gia tăng một tài sản vào danh mục (Shalit & Yitzhaki, 1994, *Management Science*), và các kiểm định nhất quán dựa trên hàm phân phối thực nghiệm CDF (Barrett & Donald, 2003, *Econometrica*; Linton et al., 2005, *REStud*).
2. **Cụm 2 — Almost Stochastic Dominance & Tối ưu hóa Khả thi (`Ch. 2.1 & 3.1`):**
   Leshno & Levy (2002, *MS*) mở ra ASD cho "đa số nhà đầu tư"; Tzeng, Huang & Shih (2013, *MS*) sửa điều kiện ASSD chuẩn; Kuosmanen (2004, *MS*) biến SSD thành lập trình tuyến tính; gần nhất là Müller et al. (2025, *Operations Research*) mở rộng ASD đa biến, Peng & Delage (2024, *Operations Research*) với Distributionally Robust SSD, Luo, Chen & Jaillet (2025, *MS*) tối ưu danh mục ASSD, và Cesarone & Puerto (2025, *EJOR*) dùng CSESD/OWA.
3. **Cụm 3 — ML Dự báo Phân phối / CDF qua Phân vị (`Ch. 2.2 & 3.2`):**
   Nền tảng hồi quy phân vị (Buchinsky, 1994, *Econometrica*; Xiao, 2009, *Journal of Econometrics*), nối thẳng sang Quantile Regression Forests (Meinshausen, 2006, *JMLR*), Neural Networks cho hồi quy phân vị bán tuyến tính (Zhong & Wang, 2024, *JBES*), và đỉnh cao hiện nay là Two-Stage Quantile Neural Networks + Cubic B-Spline (Baruník, Hronec & Tobek, 2026, *IJF*).
4. **Cụm 4 — Kết hợp ML + SD trong Chọn cổ phiếu & Danh mục (`Ch. 2.2 & 3.1`):**
   Ban, El Karoui & Lim (2018, *MS*) đưa ML vào tối ưu danh mục; Jiang (2021, *ESWA*) tổng hợp deep learning cho chứng khoán; Ma, Han & Wang (2021, *ESWA*) và Avramov, Cheng & Metzker (2023, *MS*) chứng minh ràng buộc kinh tế giúp ML ổn định hơn; gần đây **Babapour-Azar & Khanjani-Shiraz (2024, *FRL*)** dùng Neural Network giải bài toán danh mục SSD, **Jang & Seong (2023, *ESWA*)** kết hợp Deep RL + MPT, **Consigli et al. (2024, *EAAI*)** kết hợp RL + SSD, **Neugebauer (2026, *EJOR*)** dùng kiểm định SD bền vững, và **Wang (2025, *FRL*)** so sánh Transformer vs mạng đơn giản trong dự báo cổ phiếu.
5. **Cụm 5 — Tâm lý "Yêu/Thích", Hành vi Nhà đầu tư & Dòng tiền (`Ch. 2.1 & 2.2`):**
   Baker & Wurgler (2006, *JF*) và Huang et al. (2015, *RFS*) chứng minh tâm lý nhà đầu tư dẫn dắt lợi suất cắt ngang; Kumar & Lee (2006, *JF*) chỉ ra nhà đầu tư cá nhân tạo biến động đồng pha (retail comovement); Schmeling (2009, *JEF*) mở rộng ra quốc tế; Fong (2013, *JBF*) liên kết SD với cổ phiếu vé số (lottery stocks); **Guo et al. (2023, *FRL*)** phân tích độ nhạy dòng tiền quỹ theo nhóm nhà đầu tư và **Zhao & Zhang (2023, *FRL*)** chỉ ra sở hữu nước ngoài với niềm tin dị biệt tác động lên biến động.
6. **Cụm 6 — Thị trường Mới nổi & Bối cảnh Việt Nam (`Ch. 1.1 & 2.2`):**
   Cakici et al. (2013, *EMR*) và Hanauer & Kalsbach (2023, *EMR*) kiểm chứng nhân tố và ML ở thị trường mới nổi; Leippold, Wang & Zhou (2022, *JFE*) cho thấy ở thị trường nhiều nhà đầu tư cá nhân (Trung Quốc), biến thanh khoản và tín hiệu ngắn hạn quyết định sức mạnh ML; riêng Việt Nam có **Vo (2016, *IRFA*)** về sở hữu tổ chức, **Vo (2017, *IRFA*)** về dòng tiền giao dịch khối ngoại trên HOSE, **Vo & Ellis (2018, *EMR*)** về liên thông quốc tế, **Huang, Liu & Shu (2023, *PBFJ*)** về các bất thường định giá VN, **Thanh Nhon et al. (2025, *PLOS ONE*)** về ML + DEA trên HOSE, và **Tran, Wilson & Yang (2025, *EMR*)** về sở hữu tổ chức tại Việt Nam.

---

### 3. Bảng Kế thừa Biến & Phương pháp (Chuẩn Mục 3.2 Đề cương HCMUTE)

| Nhóm biến / Thành phần | Tên biến & Ký hiệu | Công thức / Thước đo cụ thể | Nguồn Q1 kế thừa |
|---|---|---|---|
| **Biến đầu vào (Features từ Yahoo Finance & BCTC)** | Lợi suất trễ, Động lượng, Độ biến động EWMA, Chỉ số hiệu quả / cơ bản | \(R_{i,t-k}\), \(\sigma_{i,t}\) (EWMA \(\lambda=0{,}94\)), ROE, P/B, Biên lợi nhuận trước thuế, Thanh khoản | Baruník et al. (2026, *IJF*); Leippold et al. (2022, *JFE*); Thanh Nhon et al. (2025, *PLOS ONE*) |
| **Biến trung gian (Đầu ra ML)** | CDF lợi suất tương lai \(\hat{F}_{i,t+1}(r)\) | Dự báo lưới phân vị \(\tau \in \{0{,}01 \dots 0{,}99\}\) bằng QRF / QNN + Cubic B-Spline | Meinshausen (2006, *JMLR*); Baruník et al. (2026, *IJF*) |
| **Biến Nhãn "Yêu / Không yêu"** | Chỉ số vi phạm ASSD \(\varepsilon_2^{*}(i \succ \text{Bench})\) | \(\text{Yêu} = 1\) nếu \(\hat{\mathbb{E}}[R_i] \ge \hat{\mathbb{E}}[R_{\text{Bench}}]\) và \(\varepsilon_2^{*} \le \varepsilon\) | Leshno & Levy (2002, *MS*); Tzeng et al. (2013, *MS*); Cesarone & Puerto (2025, *EJOR*) |
| **Biến Dòng tiền (Từ Bảng điện tử VN)** | Dòng tiền ngoại ròng (`FNF`), Mất cân bằng lệnh (`OFI`), Thanh khoản bất thường (`AbTurn`) | \(\text{FNF}_{i,t} = \frac{\text{Buy}_{\text{NN}} - \text{Sell}_{\text{NN}}}{\text{TotalVal}_{i,t}}\); \(\text{AbTurn}_{i,t} = \frac{\text{Val}_{i,t}}{\overline{\text{Val}}_{i,t-20:t-1}}\) | Vo (2017, *IRFA*); Kumar & Lee (2006, *JF*); Guo et al. (2023, *FRL*) |
| **Biến Kiểm soát "Cổ phiếu vé số"** | Lợi suất cực đại tháng trước (`MAX`) & Độ lệch (`Skew`) | \(\text{MAX}_{i,t} = \max_{d \in t} R_{i,d}\) (giải thích vì sao dòng tiền cá nhân lệch khỏi SSD) | Fong (2013, *JBF*); Huang et al. (2023, *PBFJ*) |
| **Thước đo đánh giá mô hình & danh mục** | Pinball Loss, CRPS, Sharpe, Sortino, Omega, Max Drawdown, Turnover | Đánh giá Out-of-sample theo Rolling/Expanding Window (trừ phí & thuế bán \(0{,}1\%\)) | Cesarone & Puerto (2025, *EJOR*); Baruník et al. (2026, *IJF*) |
