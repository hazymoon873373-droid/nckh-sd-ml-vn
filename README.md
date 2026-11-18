# VN-QUANT NCKH: Dự báo Cổ phiếu "Yêu / Không yêu" & Dòng tiền Việt Nam (ML × Almost Stochastic Dominance)

Nền tảng nghiên cứu khoa học định lượng (NCKH) kết hợp **Distributional / Quantile Machine Learning** và **Almost Stochastic Dominance (FSD, SSD, ASSD)** để dự báo phân phối lợi suất tương lai, phân loại mức độ ưa thích cổ phiếu và kiểm định dịch chuyển dòng tiền trên sàn HOSE (2021–2026).

## Mở nhanh Ứng dụng Web Công khai
- **Live GitHub Pages Web Terminal:** `https://hazymoon873373-droid.github.io/nckh-sd-ml-vn/`

## Cấu trúc Hệ thống Nghiên cứu
1. **Dữ liệu Thực nghiệm 5 năm (`1.305` phiên, `2021-11-18` → `2026-11-18`):** 38 cổ phiếu thanh khoản cao trên HOSE + chứng chỉ quỹ `E1VFVN30.VN` từ Yahoo Finance (`.VN`), tự động kiểm định và hiệu chỉnh 98 phiên lệch ngày chia tách/cổ tức vượt biên độ ±7%.
2. **Pipeline Machine Learning Phân phối (`web_demo/research_pipeline.py` & `docs/js/ml-engine.js`):**
   - `Two-Stage Vol-Scaled Quantile Ridge + Conformal Calibration` (Baruník, Hronec & Tobek, 2026, *IJF*; Vovk et al., 2019, *ML*)
   - `Quantile Regression Forests` (Meinshausen, 2006, *JMLR*)
   - `Quantile Gradient Boosting (Pinball GBDT)`
   - So sánh với `Historical Empirical CDF` và `Point-ML Ridge`.
3. **Kiểm định Stochastic Dominance & Tối ưu Danh mục (`docs/js/sd-math.js`):**
   - Triển khai chuẩn công thức ASSD hiệu chỉnh của Tzeng, Huang & Shih (2013, *Management Science*) trên tích phân bậc hai $F^{(2)}(t)$.
   - Giải bài toán tối ưu danh mục *Enhanced Indexation* theo Cesarone & Puerto (2025, *EJOR*) & Luo, Chen & Jaillet (2025, *MS*) với giới hạn tỷ trọng đơn lẻ 20% và phí giao dịch + thuế bán HOSE `0,35%`.
4. **Kho 376 Bài báo Q1 Đã Xác minh DOI (`q1_200plus_corpus.csv` & `docs/data/q1_corpus.json`):** Phân cụm theo 6 nhánh nghiên cứu cùng 24 tạp chí tài chính & ML hàng đầu thế giới.

## Chạy Server Local (Tùy chọn)
```bash
cd web_demo
python server.py
# Mở http://127.0.0.1:8505/
```
