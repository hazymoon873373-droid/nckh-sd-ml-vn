# Cơ sở Toán học: Utility Theory $\to$ CDF $\to$ FSD, SSD & Almost Stochastic Dominance

> Tài liệu này tổng hợp và chứng minh trực quan phần ghi chú viết tay: *Utility theory $\to$ không cần biết chính xác hàm utility $u(x)$ $\to$ chuyển sang so sánh qua CDF $\to$ FSD, SSD, Almost SD $\to$ cách hiểu & diễn giải*.

---

## 1. Tại sao phải đi từ Lý thuyết Hữu dụng Kỳ vọng (Expected Utility) sang Stochastic Dominance?

Trong lý thuyết tài chính cổ điển (von Neumann & Morgenstern), nhà đầu tư chọn cổ phiếu $A$ thay vì cổ phiếu $B$ nếu hữu dụng kỳ vọng lớn hơn:

\[
\mathbb{E}[u(R_A)] \ge \mathbb{E}[u(R_B)]
\]

**Vấn đề thực tế (điểm nghẽn trong ghi chú của bạn):**
Để tính $\mathbb{E}[u(R)]$, bạn **bắt buộc phải giả định một dạng hàm $u(x)$ cụ thể** (ví dụ hàm mũ CARA $u(x) = -e^{-\alpha x}$ hoặc logarit $u(x) = \ln(x)$). Nhưng mỗi nhà đầu tư trên sàn HOSE có một hàm $u(x)$ khác nhau và không thể quan sát được!

**Bước đột phá của Hadar & Russell (1969) và Hanoch & Levy (1969):**
Nhờ tích phân từng phần (integration by parts), điều kiện $\mathbb{E}[u(R_A)] \ge \mathbb{E}[u(R_B)]$ cho **cả một họ hàm hữu dụng rộng** có thể chuyển hoàn toàn thành điều kiện hình học trên **Hàm phân phối tích lũy (CDF)**:

\[
F_A(r) = \mathbb{P}(R_A \le r), \qquad F_B(r) = \mathbb{P}(R_B \le r)
\]

Nói cách khác: **Không cần biết công thức chính xác của $u(x)$**, chỉ cần biết quy luật tâm lý chung của nhà đầu tư (thích lời hơn lỗ, hay sợ rủi ro), ta kiểm tra trực tiếp trên **CDF của lợi suất**!

---

## 2. Công thức Toán học Chuẩn hóa (FSD, SSD, Almost SD)

Giả sử hai cổ phiếu/danh mục $A$ và $B$ có lợi suất ngẫu nhiên thuộc miền bị chặn $[a, b]$, với CDF lần lượt là $F_A(x)$ và $F_B(x)$.

### 2.1. First-Order Stochastic Dominance (FSD — Trội bậc 1)

**Lớp nhà đầu tư $\mathcal{U}_1$:** Chỉ cần "ham lời" (non-satiation: $u'(x) \ge 0$, càng nhiều tiền càng thích).

**Công thức CDF:** $A$ trội hơn $B$ theo FSD ($A \succeq_{\text{FSD}} B$) khi và chỉ khi:

\[
F_A(x) \le F_B(x) \quad \forall x \in [a, b]
\]

*(và dấu $<$ xảy ra tại ít nhất một điểm $x$).*

**Cách hiểu & diễn giải bằng trực giác:**
- $F_A(x) = \mathbb{P}(R_A \le x)$ là **xác suất cổ phiếu $A$ đạt lợi suất thấp hơn hoặc bằng ngưỡng $x$**.
- Vì vậy $F_A(x) \le F_B(x)$ nghĩa là ở **mọi mức lợi suất $x$**, xác suất $A$ rơi xuống dưới $x$ luôn nhỏ hơn hoặc bằng $B$ (tức xác suất vượt ngưỡng $1 - F_A(x) \ge 1 - F_B(x)$ luôn cao hơn).
- Trên đồ thị CDF: đường $F_A(x)$ **luôn nằm hoàn toàn bên phải / phía dưới** đường $F_B(x)$ và không bao giờ cắt nhau.

### 2.2. Second-Order Stochastic Dominance (SSD — Trội bậc 2)

**Lớp nhà đầu tư $\mathcal{U}_2$:** Vừa ham lời ($u'(x) \ge 0$) vừa **ngại rủi ro** (risk-averse: $u''(x) \le 0$).

Đặt hàm tích phân bậc 2 (Integrated CDF):

\[
F_A^{(2)}(t) = \int_{a}^{t} F_A(x)\,dx, \qquad F_B^{(2)}(t) = \int_{a}^{t} F_B(x)\,dx
\]

**Công thức CDF:** $A$ trội hơn $B$ theo SSD ($A \succeq_{\text{SSD}} B$) khi và chỉ khi:

\[
\int_{a}^{t} \bigl[F_A(x) - F_B(x)\bigr]\,dx \le 0 \quad \forall t \in [a, b]
\]

*(tương đương $F_A^{(2)}(t) \le F_B^{(2)}(t)$ với mọi $t$, và nghiêm ngặt tại ít nhất một điểm).*

**Liên hệ cực hay với Risk Management (theo Cesarone & Puerto, 2025, *EJOR*):**
Khi dữ liệu rời rạc gồm $T$ kịch bản đồng xác suất $\pi_t = 1/T$, nếu sắp xếp lợi suất tăng dần $R_{(1)} \le R_{(2)} \le \dots \le R_{(T)}$, điều kiện SSD rút gọn thành:

\[
\sum_{j=1}^{k} R_{A,(j)} \ge \sum_{j=1}^{k} R_{B,(j)} \quad \forall k = 1, \dots, T
\]

Điều này tương đương với việc $A$ có **$\operatorname{CVaR}_{\beta}(A) \le \operatorname{CVaR}_{\beta}(B)$ ở mọi mức tail $\beta \in (0{,}1]$**!
- **Diễn giải trực quan:** Cho phép hai đường CDF $F_A$ và $F_B$ được phép cắt nhau một chút ở vùng lãi cao, miễn là **diện tích thua lỗ/bất lợi tích lũy từ đuôi trái trở lên** của $A$ luôn nhỏ hơn hoặc bằng $B$. Nhà đầu tư ngại rủi ro chấp nhận cổ phiếu $A$ có thể kém bùng nổ cực đại hơn $B$ một chút, miễn là độ an toàn đuôi dưới (downside protection) đủ tốt để bù đắp.

### 2.3. Almost Stochastic Dominance (ASD — Trội ngẫu nhiên gần đúng)

**Vì sao ngoài đời FSD và SSD hay "bất lực"? (Hạn chế thực nghiệm):**
Chỉ cần cổ phiếu $B$ có **1 phiên tăng trần đột biến** hoặc **1 xác suất cực nhỏ $0{,}1\%$ ăn bằng lần**, hoặc cổ phiếu $A$ thua $B$ đúng $0{,}01\%$ ở cực trị đuôi trái, thì FSD và SSD truyền thống đều kết luận **"Không so sánh được"** — vì tồn tại 1 hàm hữu dụng quái dị (pathological utility) thích cực trị đó!

**Giải pháp của Leshno & Levy (2002, *Management Science*) và bản hiệu chỉnh chuẩn của Tzeng, Huang & Shih (2013, *Management Science*):**
Loại bỏ các hàm hữu dụng cực đoan, chỉ xét **"đa số" (Most) nhà đầu tư hợp lý** bằng cách giới hạn tỷ lệ diện tích vùng vi phạm!

#### (a) Almost First-Order Stochastic Dominance ($\varepsilon$-AFSD)
Đặt tập vùng vi phạm FSD (nơi $F_A(x) > F_B(x)$):

\[
S_1 = \bigl\{x \in [a, b] : F_A(x) > F_B(x)\bigr\}
\]

$A$ trội hơn $B$ theo $\varepsilon_1$-AFSD ($\varepsilon_1 \in (0, 0{,}5)$) khi và chỉ khi:

\[
\int_{S_1} \bigl[F_A(x) - F_B(x)\bigr]\,dx \le \varepsilon_1 \int_{a}^{b} \bigl|F_A(x) - F_B(x)\bigr|\,dx
\]

Định nghĩa **Chỉ số vi phạm FSD tối thiểu (Minimum Violation Ratio)**:

\[
\varepsilon_1^{*}(A \succ B) = \frac{\int_{S_1} \bigl[F_A(x) - F_B(x)\bigr]\,dx}{\int_{a}^{b} \bigl|F_A(x) - F_B(x)\bigr|\,dx}
\]

- Nếu $\varepsilon_1^{*} = 0$: $A$ **FSD hoàn hảo** $B$.
- Nếu $\varepsilon_1^{*} \le \varepsilon_1$ (thường ngưỡng kinh nghiệm trong tài chính là $\varepsilon \approx 0{,}059$ theo thực nghiệm của Levy et al.): **đa số nhà đầu tư thích $A$ hơn $B$**!

#### (b) Almost Second-Order Stochastic Dominance ($\varepsilon$-ASSD — Công thức chuẩn Tzeng et al., 2013)
Đặt tập vùng vi phạm SSD (nơi tích phân bậc 2 của $A$ lớn hơn $B$):

\[
S_2 = \bigl\{t \in [a, b] : F_A^{(2)}(t) > F_B^{(2)}(t)\bigr\}
\]

Theo **Tzeng, Huang & Shih (2013)** (sửa lỗi điều kiện cần và đủ của Leshno & Levy, 2002), $A$ trội hơn $B$ theo $\varepsilon_2$-ASSD khi thỏa đồng thời:

1. Lợi suất kỳ vọng không thấp hơn: $\mathbb{E}[R_A] \ge \mathbb{E}[R_B]$
2. Tỷ lệ vi phạm tích phân bậc 2 bị chặn:

\[
\int_{S_2} \bigl[F_A^{(2)}(t) - F_B^{(2)}(t)\bigr]\,dt \le \varepsilon_2 \int_{a}^{b} \bigl|F_A^{(2)}(t) - F_B^{(2)}(t)\bigr|\,dt
\]

*(Hoặc viết theo dạng tỷ số $\tau = \frac{1}{\varepsilon_2} - 1$: phần diện tích thắng $\int_{\bar{S}_2}[F_B^{(2)} - F_A^{(2)}]\,dt \ge \tau \int_{S_2}[F_A^{(2)} - F_B^{(2)}]\,dt$).*

---

## 3. Ánh xạ trực tiếp sang Bài toán của Bạn ("Yêu / Không yêu — Thích / Không thích")

| Khái niệm trong đề tài | Nghĩa toán học trên CDF (sau khi hiệu chỉnh ASSD) | Giá trị sử dụng cho ML & Dòng tiền |
|---|---|---|
| **Yêu / Thích ($y = 1$)** | $\varepsilon_2^{*}(i \succ \text{VNIndex}) \le \varepsilon_{\text{ngưỡng}}$ và $\mathbb{E}[\hat{R}_i] \ge \mathbb{E}[\hat{R}_{\text{VNIndex}}]$ | Cổ phiếu có phân phối lợi suất tương lai áp đảo chỉ số chuẩn cho đa số nhà đầu tư ngại rủi ro $\to$ **hút dòng tiền thông minh** |
| **Trung tính / Phân hóa** | Không cổ phiếu nào chi phối cổ phiếu nào với $\varepsilon$ nhỏ | Chờ tín hiệu bổ sung từ dòng tiền bảng điện tử |
| **Không yêu / Không thích ($y = 0$)** | $\varepsilon_2^{*}(\text{VNIndex} \succ i) \le \varepsilon_{\text{ngưỡng}}$ (bị benchmark chi phối) | Phân phối bất lợi ở cả trung bình lẫn rủi ro đuôi $\to$ **nguy cơ dòng tiền rút ra** |
| **Điểm ưa thích liên tục (Preference Score)** | $S_{\text{ASD}}(i) = 1 - 2\,\varepsilon_2^{*}(i \succ \text{Bench})$ hoặc Net Dominance Score trên tập $\mathcal{N}$ cổ phiếu | Dùng trực tiếp để xếp hạng (ranking) Top-K cổ phiếu thu hút dòng tiền! |

## 4. Bẫy Toán học cần tránh khi viết NCKH
1. **Dùng công thức ASSD 2002 mà quên sửa của Tzeng, Huang & Shih (2013):** Nếu trích Leshno & Levy (2002) cho ASSD mà không kiểm tra điều kiện $\mathbb{E}[X] \ge \mathbb{E}[Y]$ và công thức $F^{(2)}$, hội đồng phản biện tài chính định lượng sẽ bắt bẻ ngay.
2. **Nhầm lẫn "diện tích giữa hai đường $F_A, F_B$" với "diện tích giữa $F_A^{(2)}, F_B^{(2)}$":** AFSD tích phân trực tiếp trên $F(x)$; còn ASSD (bản Tzeng et al., 2013) tích phân trên $F^{(2)}(t)$!
3. **Không chuẩn hóa miền $[a, b]$:** Vì cổ phiếu HOSE có biên độ ngày $\pm 7\%$ và lợi suất tháng phân tán khác nhau, khi tính tích phân số (numerical integration) phải dùng chung lưới điểm $x_1 < x_2 < \dots < x_M$ phủ trọn $[\min(R_A, R_B), \max(R_A, R_B)]$.
