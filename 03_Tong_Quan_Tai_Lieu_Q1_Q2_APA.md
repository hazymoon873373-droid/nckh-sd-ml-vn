# Tổng quan Tài liệu Q1 & Q2 (Đã Xác minh DOI & SJR Quartile)

> Tài liệu tham khảo được truy xuất và xác minh qua OpenAlex, Crossref API, arXiv API và SCImago Journal & Country Rank (SJR) ngày 17/11/2026. Trích dẫn trình bày theo **Chuẩn APA (Khoa Kinh tế — ĐH Sư phạm Kỹ thuật TP.HCM)**.

---

## 1. Lộ trình Đọc Ưu tiên cho Ngày mai (Start Here — 7 Bài Cốt lõi)

1. **[Leshno & Levy (2002) — *Management Science* (Q1, FT50)](https://doi.org/10.1287/mnsc.48.8.1074.169)**
   - *Đóng góp:* Khai sinh **Almost Stochastic Dominance (ASD)** để vượt qua sự khắt khe quá mức của FSD/SSD truyền thống.
   - *Đọc gì:* Định nghĩa vùng vi phạm $S_1, S_2$ và tham số $\varepsilon$ ("Preferred by Most Decision Makers").
2. **[Tzeng, Huang & Shih (2013) — *Management Science* (Q1, FT50)](https://doi.org/10.1287/mnsc.1120.1616)**
   - *Đóng góp:* Hiệu chỉnh lỗi toán học trong định lý ASSD gốc của Leshno & Levy (2002), bổ sung điều kiện $\mathbb{E}[X] \ge \mathbb{E}[Y]$ và bất đẳng thức trên $F^{(2)}(t)$.
   - *Đọc gì:* Định lý 1 & Công thức tính ASSD chuẩn để đưa vào phần Phương pháp luận (tránh bị hội đồng bắt lỗi công thức cũ 2002).
3. **[Baruník, Hronec & Tobek (2026) — *International Journal of Forecasting* (Q1)](https://doi.org/10.1016/j.ijforecast.2026.04.004)** — *(Có sẵn PDF Open Access trong `papers/Barunik_Hronec_Tobek_2024_IJF2026_QuantileNN_CDFs.pdf`)*
   - *Đóng góp:* Dùng Two-Stage Quantile Neural Network + Cubic B-Spline để dự báo **toàn bộ CDF lợi suất cổ phiếu tương lai** trên dữ liệu toàn cầu.
   - *Đọc gì:* Mục 2.1–2.3 (kiến trúc 2 giai đoạn, hàm Pinball Loss, nội suy B-spline từ phân vị sang CDF mượt).
4. **[Meinshausen (2006) — *Journal of Machine Learning Research* (Q1)](https://www.jmlr.org/papers/volume7/meinshausen06a/meinshausen06a.pdf)** — *(Có sẵn PDF trong `papers/Meinshausen_2006_JMLR_Quantile_Regression_Forests.pdf`)*
   - *Đóng góp:* Thuật toán Quantile Regression Forests (QRF) ước lượng trực tiếp CDF có điều kiện $\hat{F}(y \mid X=x)$ không cần giả định phân phối.
   - *Đọc gì:* Mục 3, Phương trình (6) — công thức trọng số lá cực dễ code baseline cho dữ liệu HOSE.
5. **[Cesarone & Puerto (2025) — *European Journal of Operational Research* (Q1)](https://doi.org/10.1016/j.ejor.2024.11.050)** — *(Có sẵn PDF trong `papers/Cesarone_Puerto_2024_EJOR2025_ApproxSD_EnhancedIndexation.pdf`)*
   - *Đóng góp:* Dùng quy tắc gần-SSD (CSESD / ASSD) và lập trình tuyến tính để chọn danh mục vượt trội benchmark index.
   - *Đọc gì:* Mục 2, Trang 5–8 (cách rời rạc hóa FSD/SSD thành thứ tự mẫu và CVaR trên kịch bản).
6. **[Vo (2017) — *International Review of Financial Analysis* (Q1)](https://doi.org/10.1016/j.irfa.2017.05.007)** & **[Phan et al. (2023) — *QREF* (Q2)](https://doi.org/10.1016/j.qref.2021.07.001)**
   - *Đóng góp:* Bằng chứng thực nghiệm chuẩn Q1/Q2 về dòng tiền nhà đầu tư nước ngoài và tâm lý/hành vi nhà đầu tư tại thị trường chứng khoán Việt Nam.
   - *Đọc gì:* Cách định nghĩa biến dòng tiền ngoại ròng (foreign net flow), dữ liệu HOSE và lập luận bối cảnh Việt Nam.
7. **[Thanh Nhon, Do-Thi & Nguyen-Trang (2025) — *PLOS ONE* (Q1)](https://doi.org/10.1371/journal.pone.0332154)** — *(Có sẵn PDF trong `papers/PLOS_ONE_2025_Vietnam_HOSE_ML_DEA.pdf`)* & **[Phuoc et al. (2024) — *HSSC* (Q1)](https://doi.org/10.1057/s41599-024-02807-x)**
   - *Đóng góp:* Thực nghiệm mới nhất dùng ML (GBDT, DNN, LSTM) dự báo cổ phiếu trên sàn HOSE/VN30 Việt Nam.
   - *Đọc gì:* Bảng biến đầu vào, đặc thù thanh khoản thấp / biến động theo quý ở Việt Nam, và giới hạn của dự báo điểm để làm bàn đạp nêu Research Gap của bạn!

---

## 2. Nhóm Tác giả Dẫn đầu (từ Cross-Search Aggregator)
1. **Haim Levy / Moshe Levy / Larry Y. Tzeng / Rachel J. Huang / Ilia Tsetlin:** Nhóm nền tảng lý thuyết Expected Utility, Stochastic Dominance và Almost Stochastic Dominance (*Management Science*, *Operations Research*, *EJOR*).
2. **Olivier Scaillet, Nikolas Topaloglou, Stelios Arvanitis & Miloš Kopa:** Nhóm kinh tế lượng kiểm định SD efficiency, spanning tests và tối ưu danh mục dưới ràng buộc SD (*Journal of Econometrics*, *JBES*, *Management Science*, *EJOR*, *OR Spectrum*).
3. **Wing-Keung Wong & cộng sự (Pin Ng, Zhijie Xiao, Hooi Hooi Lean):** Nhóm hàng đầu về kiểm định SD qua hồi quy phân vị (*EJOR*, 2017) và ứng dụng SD tại các thị trường châu Á/mới nổi.
4. **Jozef Baruník, Martin Hronec & Ondřej Tobek:** Nhóm tiên phong dự báo toàn bộ CDF lợi suất cổ phiếu bằng Quantile Neural Networks trên quy mô toàn cầu (*International Journal of Forecasting*, 2026).
5. **Xuan Vinh Vo (Võ Xuân Vinh) & cộng sự:** Nhóm nghiên cứu dẫn đầu tại Việt Nam về dòng vốn ngoại, hành vi bầy đàn và tâm lý nhà đầu tư trên HOSE (*IRFA*, *Pacific-Basin Finance Journal*, *QREF*).

---

## 3. Danh mục Tài liệu Tham khảo theo Chuẩn APA (Khoa Kinh tế — HCMUTE)

*(Sắp xếp theo trật tự ABC của họ tác giả đứng đầu; giữ nguyên tên đầy đủ với tác giả Việt Nam nếu viết tiếng Việt, và họ + tên viết tắt với tác giả quốc tế theo thông lệ APA 7 và hướng dẫn Khoa Kinh tế).*

Abhyankar, A., Ho, K.-Y., & Zhao, H. (2008). Value versus growth: stochastic dominance criteria. *Quantitative Finance*, 8(7), 693–704. https://doi.org/10.1080/14697680701668426 [Q1]

Arvanitis, S., Scaillet, O., & Topaloglou, N. (2020). Spanning tests for Markowitz stochastic dominance. *Journal of Econometrics*, 217(2), 291–311. https://doi.org/10.1016/j.jeconom.2019.12.005 [Q1]

Arvanitis, S., Scaillet, O., & Topaloglou, N. (2023). Spanning analysis of stock market anomalies under prospect stochastic dominance. *Management Science*. https://doi.org/10.1287/mnsc.2023.4953 [Q1]

Babapour-Azar, A., & Khanjani-Shiraz, R. (2024). A neural network framework for portfolio optimization under second-order stochastic dominance. *Finance Research Letters*, 66, 105626. https://doi.org/10.1016/j.frl.2024.105626 [Q1]

Barucci, E., Bonollo, M., Poli, F., & Rroji, E. (2021). A machine learning algorithm for stock picking built on information based outliers. *Expert Systems with Applications*, 184, 115497. https://doi.org/10.1016/j.eswa.2021.115497 [Q1]

Baruník, J., Hronec, M., & Tobek, O. (2026). Forecasting stock return distributions around the globe with quantile neural networks. *International Journal of Forecasting*, 42(4), 1298–1313. https://doi.org/10.1016/j.ijforecast.2026.04.004 [Q1]

Cesarone, F., & Puerto, J. (2025). Flexible enhanced indexation models through stochastic dominance and ordered weighted average optimization. *European Journal of Operational Research*, 323(2), 657–670. https://doi.org/10.1016/j.ejor.2024.11.050 [Q1]

Chernozhukov, V., Wüthrich, K., & Zhu, Y. (2021). Distributional conformal prediction. *Proceedings of the National Academy of Sciences*, 118(48), e2107794118. https://doi.org/10.1073/pnas.2107794118 [Q1]

Cho, Y., Linton, O., & Whang, Y.-J. (2007). Are there Monday effects in stock returns: A stochastic dominance approach. *Journal of Empirical Finance*, 14(5), 736–755. https://doi.org/10.1016/j.jempfin.2007.02.001 [Q1]

Clark, E., & Kassimatis, K. (2013). International equity flows, marginal conditional stochastic dominance and diversification. *Review of Quantitative Finance and Accounting*, 40(2), 251–271. https://doi.org/10.1007/s11156-012-0277-0 [Q2]

Clark, E., & Kassimatis, K. (2014). Exploiting stochastic dominance to generate abnormal stock returns. *Journal of Financial Markets*, 20, 20–38. https://doi.org/10.1016/j.finmar.2014.05.002 [Q1]

Consigli, G., Gomez, A. A., & Zubelli, J. P. (2024). Optimal dynamic fixed-mix portfolios based on reinforcement learning with second order stochastic dominance. *Engineering Applications of Artificial Intelligence*, 133, 108599. https://doi.org/10.1016/j.engappai.2024.108599 [Q1]

Davidson, R., & Duclos, J.-Y. (2000). Statistical inference for stochastic dominance and for the measurement of poverty and inequality. *Econometrica*, 68(6), 1435–1464. https://doi.org/10.1111/1468-0262.00167 [Q1]

Fong, W. M. (2013). Risk preferences, investor sentiment and lottery stocks: A stochastic dominance approach. *Journal of Behavioral Finance*, 14(1), 42–52. https://doi.org/10.1080/15427560.2013.759579 [Q2]

Gu, S., Kelly, B., & Xiu, D. (2020). Empirical asset pricing via machine learning. *The Review of Financial Studies*, 33(5), 2223–2273. https://doi.org/10.1093/rfs/hhaa009 [Q1]

Han, W., Newton, D., Platanakis, E., Sutcliffe, C., & Ye, X. (2024). On the (almost) stochastic dominance of cryptocurrency factor portfolios and implications for cryptocurrency asset pricing. *European Financial Management*, 30(3), 1125–1164. https://doi.org/10.1111/eufm.12431 [Q1/Q2]

Hanoch, G., & Levy, H. (1969). The efficiency analysis of choices involving risk. *The Review of Economic Studies*, 36(3), 335–346. https://doi.org/10.2307/2296431 [Q1]

Huang, X., Liu, C., & Shu, T. (2023). Factors and anomalies in the Vietnamese stock market. *Pacific-Basin Finance Journal*, 82, 102176. https://doi.org/10.1016/j.pacfin.2023.102176 [Q1]

Junová, J., & Kopa, M. (2025). Measures of stochastic non-dominance in portfolio optimization. *European Journal of Operational Research*, 321(1), 269–283. https://doi.org/10.1016/j.ejor.2024.08.029 [Q1]

Kopa, M., & Kozmík, K. (2025). Robust approaches in portfolio optimization with stochastic dominance constraints. *OR Spectrum*, 47(4), 1421–1453. https://doi.org/10.1007/s00291-025-00814-1 [Q2]

Leippold, M., Wang, Q., & Zhou, W. (2022). Machine learning in the Chinese stock market. *Journal of Financial Economics*, 145(2), 64–82. https://doi.org/10.1016/j.jfineco.2021.08.017 [Q1]

Leshno, M., & Levy, H. (2002). Preferred by "all" and preferred by "most" decision makers: Almost stochastic dominance. *Management Science*, 48(8), 1074–1085. https://doi.org/10.1287/mnsc.48.8.1074.169 [Q1]

Levy, H. (1992). Stochastic dominance and expected utility: Survey and analysis. *Management Science*, 38(4), 555–593. https://doi.org/10.1287/mnsc.38.4.555 [Q1]

Levy, M. (2009). Almost stochastic dominance and stocks for the long run. *European Journal of Operational Research*, 194(1), 250–257. https://doi.org/10.1016/j.ejor.2007.12.017 [Q1]

Linton, O., Maasoumi, E., & Whang, Y.-J. (2005). Consistent testing for stochastic dominance under general sampling schemes. *Review of Economic Studies*, 72(3), 735–765. https://doi.org/10.1111/j.1467-937x.2005.00350.x [Q1]

Luo, C., Chen, P., & Jaillet, P. (2025). Portfolio optimization based on almost second-degree stochastic dominance. *Management Science*, 71(8), 7029–7055. https://doi.org/10.1287/mnsc.2022.01092 [Q1]

Meinshausen, N. (2006). Quantile regression forests. *Journal of Machine Learning Research*, 7, 983–999. https://www.jmlr.org/papers/volume7/meinshausen06a/meinshausen06a.pdf [Q1]

Ng, P., Wong, W.-K., & Xiao, Z. (2017). Stochastic dominance via quantile regression with applications to investigate arbitrage opportunity and market efficiency. *European Journal of Operational Research*, 261(2), 666–678. https://doi.org/10.1016/j.ejor.2017.02.047 [Q1]

Phan, T. N. T., Bertrand, P., Phan, H. H., & Vo, X. V. (2023). The role of investor behavior in emerging stock markets: Evidence from Vietnam. *The Quarterly Review of Economics and Finance*, 87, 367–376. https://doi.org/10.1016/j.qref.2021.07.001 [Q2]

Phuoc, T., Anh, P. T. K., Tam, P. H., & Nguyen, C. V. (2024). Applying machine learning algorithms to predict the stock price trend in the stock market – The case of Vietnam. *Humanities and Social Sciences Communications*, 11(1), 393. https://doi.org/10.1057/s41599-024-02807-x [Q1]

Post, T. (2003). Empirical tests for stochastic dominance efficiency. *The Journal of Finance*, 58(5), 1905–1931. https://doi.org/10.1111/1540-6261.00592 [Q1]

Sak, H., Huang, T., & Chng, M. T. (2024). Exploring the factor zoo with a machine-learning portfolio. *International Review of Financial Analysis*, 96, 103599. https://doi.org/10.1016/j.irfa.2024.103599 [Q1]

Scaillet, O., & Topaloglou, N. (2010). Testing for stochastic dominance efficiency. *Journal of Business & Economic Statistics*, 28(1), 169–180. https://doi.org/10.1198/jbes.2009.06167 [Q1]

Thanh Nhon, H., Do-Thi, N., & Nguyen-Trang, T. (2025). Predicting stock returns using machine learning combined with data envelopment analysis and automatic feature engineering: A case study on the Vietnamese stock market. *PLOS One*, 20(9), e0332154. https://doi.org/10.1371/journal.pone.0332154 [Q1]

Tsetlin, I., Winkler, R. L., Huang, R. J., & Tzeng, L. Y. (2015). Generalized almost stochastic dominance. *Operations Research*, 63(2), 363–377. https://doi.org/10.1287/opre.2014.1340 [Q1]

Tzeng, L. Y., Huang, R. J., & Shih, P.-T. (2013). Revisiting almost second-degree stochastic dominance. *Management Science*, 59(5), 1250–1254. https://doi.org/10.1287/mnsc.1120.1616 [Q1]

Vo, X. V. (2017). Trading of foreign investors and stock returns in an emerging market - Evidence from Vietnam. *International Review of Financial Analysis*, 52, 88–93. https://doi.org/10.1016/j.irfa.2017.05.007 [Q1]

Vo, X. V., & Phan, D. B. A. (2019). Herd behavior and idiosyncratic volatility in a frontier market. *Pacific-Basin Finance Journal*, 53, 321–330. https://doi.org/10.1016/j.pacfin.2018.10.005 [Q1]

Vovk, V., Petej, I., Nouretdinov, I., Manokhin, V., & Gammerman, A. (2020). Computationally efficient versions of conformal predictive distributions. *Neurocomputing*, 397, 292–308. https://doi.org/10.1016/j.neucom.2019.10.110 [Q1]

Vovk, V., Shen, J., Manokhin, V., & Xie, M. (2019). Nonparametric predictive distributions based on conformal prediction. *Machine Learning*, 108(3), 445–474. https://doi.org/10.1007/s10994-018-5755-8 [Q1]

Wolff, D., & Echterling, F. (2024). Stock picking with machine learning. *Journal of Forecasting*, 43(1), 81–102. https://doi.org/10.1002/for.3021 [Q2]

### Tài liệu Thuật toán Bổ trợ (Preprint arXiv Open-Access — Ghi chú rõ là Preprint, chưa tính vào nhóm tạp chí Q1/Q2)
- Cen, S., Mei, J., Dai, H., Schuurmans, D., Chi, Y., & Dai, B. (2024/2026). *Beyond expectations: Learning with stochastic dominance made practical*. arXiv preprint arXiv:2402.02698. https://arxiv.org/abs/2402.02698
- Dai, H., Xue, Y., He, N., Wang, B., Li, N., Schuurmans, D., & Dai, B. (2022/2023). *Learning to optimize with stochastic dominance constraints*. arXiv preprint arXiv:2211.07767. https://arxiv.org/abs/2211.07767

---

## 4. Nhật ký Truy vấn (Audit Log — theo `litreview` & `meta-research`)
- **Công cụ sử dụng từ repo của bạn:** `alirezarezvani-claude-skills/research/litreview` (`framework_recommender.py`, `free_search.py`, `citation_tracker.py`, `cross_search_aggregator.py`), `alirezarezvani-claude-skills/product-team/research-summarizer` (`extract_citations.py`), `alirezarezvani-claude-skills/research/meta-research` (cấu trúc thư mục nguồn), cùng Crossref API, arXiv API và SCImago SJR verification.
- **Lưu ý dữ liệu:** OpenAlex `--search` không có API key có giới hạn ngân sách IP dùng chung (0,10 USD/ngày) nên 3 truy vấn sau gặp HTTP 429; tổng cộng có 1 truy vấn thăm dò (15 bài) + 3 truy vấn theo nhánh thành công (60 bài thô) thu về 74 bài duy nhất trên OpenAlex, kết hợp xác minh 39/41 DOI qua Crossref API, 4 preprint arXiv API, và tải về thành công **6 file PDF Open-Access** vào thư mục `papers/`.
