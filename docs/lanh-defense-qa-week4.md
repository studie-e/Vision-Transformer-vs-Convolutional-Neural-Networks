# Tài liệu Diễn tập Thuyết trình & Vấn đáp Chéo Tuần 4 — Lan Anh

**Người phụ trách:** Lan Anh (ViT tự cài, Ablation Positional Embedding, đọc chéo dữ liệu Hiền & trainer/ResNet Minh).  
**Tham chiếu:** [`slides/week4.tex`](../slides/week4.tex), [`docs/defense_questions.md`](defense_questions.md), [`tasks/plan.md`](../tasks/plan.md).

---

## Phần 1. Kịch bản Thuyết trình Cá nhân của Lan Anh (Tổng 220 giây ≈ 3 phút 40 giây)

Theo phân công trong [`docs/hien-week4.md`](hien-week4.md), bài trình bày nhóm dài **11 phút (660 giây)** gồm 10 slide, trong đó **Lan Anh phụ trách Slide 1 (40s), Slide 3 (90s) và Slide 10 (90s)**:

### 1. Slide 1 — Bài toán nghiên cứu & Câu hỏi thực nghiệm (40 giây)
- **Nội dung trình bày:**
  - Xin chào thầy/cô, nhóm em thực hiện Đề tài 3: *So sánh Vision Transformer (ViT) tự cài và mạng tích chập ResNet có số tham số tương đương trên tập ảnh nhỏ CIFAR-10 ($32\times32$) khi thay đổi quy mô dữ liệu huấn luyện ở ba mức 10%, 50% và 100%*.
  - Trọng tâm của nghiên cứu không phải chạy đua accuracy tuyệt đối, mà là kiểm chứng thực nghiệm vai trò của **inductive bias**: CNN có sẵn tính cục bộ (*locality*) và tương đương tịnh tiến (*translation equivariance*), trong khi ViT dùng self-attention toàn cục và phải học quan hệ không gian từ dữ liệu cùng positional embedding.
  - Từ đó nhóm trả lời hai câu hỏi: (1) Khoảng cách accuracy giữa ResNet và ViT biến đổi ra sao khi dữ liệu tăng từ 4.500 lên 45.000 ảnh? (2) Khi gỡ bỏ positional embedding trong ViT, mô hình mất bao nhiêu hiệu năng và giới hạn nào ngăn ta suy diễn quá mức kết quả này?

### 2. Slide 3 — Kiến trúc ViT tự cài (274.282 tham số) & Thiết kế Ablation (90 giây)
- **Nội dung trình bày:**
  - Em tự cài đặt toàn bộ mô hình ViT trong `src/models/vit.py` từ các khối cơ sở của PyTorch (`Conv2d`, `Linear`, `LayerNorm`, `GELU`, `Dropout`), không dùng mô hình ViT có sẵn.
  - Với ảnh $32\times32$, nếu dùng patch $16\times16$ như bài gốc sẽ chỉ còn $2\times2=4$ patch, mất chi tiết không gian; do đó em chọn **patch $4\times4$**, tạo lưới $8\times8 = 64$ patch. Phép chiếu `Conv2d(3, 96, kernel=4, stride=4)` biến ảnh thành chuỗi `[B, 64, 96]`, ghép thêm 1 `CLS` token học được ở đầu thành **65 token**, rồi cộng `pos_embed` học được kích thước `[1, 65, 96]`.
  - Chuỗi đi qua **4 khối Pre-LN Transformer Encoder**: mỗi khối chuẩn hóa `LayerNorm` trước khi vào Multi-Head Self-Attention (4 head, mỗi head $d_k=24$ chiều, chia $\sqrt{24}$) và khối MLP mở rộng tỉ lệ $1{,}5$ ($96\rightarrow144\rightarrow96$, kích hoạt GELU, dropout $0{,}1$). Vector `CLS` cuối cùng qua `LayerNorm` và `Linear(96, 10)` xuất logits 10 lớp.
  - Tổng tham số có thể học là **274.282**, chỉ chênh **+0,66%** so với ResNet của Minh (272.474 tham số), thỏa chặt ngưỡng $\le 10\%$ của nhóm.
  - Ở nhánh **Ablation** (`use_pos_embed=False`), em thay `pos_embed` bằng zero buffer không học, giảm đúng $65\times96 = 6.240$ tham số (còn **268.042**), giữ nguyên mọi yếu tố còn lại ở mức 50% dữ liệu để cô lập vai trò thông tin vị trí.

### 3. Slide 10 — Giới hạn phương pháp, Kiểm toán dữ liệu & Công việc hoàn tất trước nộp (90 giây)
- **Nội dung trình bày:**
  - Về **giới hạn khoa học**: Thứ nhất, hai kiến trúc dùng hai bộ tối ưu phù hợp riêng sau tuning công bằng (AdamW cho ViT, SGD momentum cho ResNet), và khi cố định 100 epoch thì số bước cập nhật tăng theo lượng ảnh ($3.600 \rightarrow 17.600 \rightarrow 35.200$ bước). Do đó chênh lệch đo được phản ánh tổ hợp *(kiến trúc + recipe + ngân sách bước)* chứ không tách riêng nhân quả tuyệt đối của inductive bias. Thứ hai, bản đồ attention chỉ cho thấy phân bố trọng số giữa token ở tầng cuối, không phải bằng chứng nhân quả duy nhất của dự đoán.
  - Về **kiểm toán bằng chứng thực nghiệm**: Nhờ pipeline kiểm tra tự động của nhóm, các log mô phỏng lỗi ở tuần 3 đã được thu hồi minh bạch; hiện nhóm đã có **9/9 lượt ResNet** và **5/9 lượt ViT** chạy GPU thực tế hoàn toàn khớp kiểm toán (trong đó mức **10% dữ liệu đã đủ cả 3 cặp seed cho cả ResNet và ViT**).
  - 4 lượt ViT 50%/100% và 3 lượt Ablation 50% còn lại đã có gói lệnh tự động chạy trên RTX 5080 của Minh trước khi nhóm khóa registry và mở tập test 10.000 ảnh đúng một lần duy nhất.

---

## Phần 2. Đáp án Chi tiết 13 Câu hỏi Vấn đáp Chung (`docs/defense_questions.md`)

1. **10% là 4.500 ảnh vì mẫu số là 45.000 ảnh train sau tách validation. Vì sao dùng tập con lồng nhau? Cách hai mô hình chia sẻ cùng ảnh?**
   - Từ 50.000 ảnh train gốc của CIFAR-10, Hiền tách phân tầng cố định 5.000 ảnh làm validation (500 ảnh/lớp) và 45.000 ảnh làm train pool (`src/data.py`). Tập con lồng nhau ($D_{10\%} \subset D_{50\%} \subset D_{100\%}$, cân bằng 450 / 2.250 / 4.500 ảnh mỗi lớp) giúp khi tăng dữ liệu, ta thật sự *bổ sung thêm mẫu mới* vào tập cũ thay vì đổi sang tập ảnh khác, giảm nhiễu lấy mẫu. Danh sách chỉ số ảnh được lưu cứng trong `results/splits/split_seed{2026,2027,2028}_manifest.json` kèm SHA-256; cả ViT và ResNet cùng đọc một manifest cho mỗi cặp seed.

2. **Split seed thay cả train và validation; train seed thay khởi tạo/shuffle. SD giữa các cặp bao gồm những nguồn biến thiên nào?**
   - Ba cặp khóa là `(2026, 42)`, `(2027, 43)`, `(2028, 44)`. Độ lệch chuẩn mẫu ($ddof=1$) qua 3 cặp phản ánh đồng thời: (a) biến thiên chọn mẫu 45k/5k và tập con, (b) biến thiên khởi tạo trọng số ngẫu nhiên (`trunc_normal_`, `xavier_uniform_`, Kaiming), và (c) thứ tự shuffle batch + augmentation ngẫu nhiên (`RandomCrop`, `RandomHorizontalFlip`, `Dropout`).

3. **Top-1 accuracy của 5.000 ảnh thay đổi theo bước $1/5.000 = 0{,}0002$. Vì sao `0,2647` cho biết log cần kiểm tra lại?**
   - Trong `src/train.py` (`run_epoch`), `val_accuracy = correct / samples` với `samples = 5000` và `correct` là số nguyên. Tích `0.2647 * 5000 = 1323.5` không phải số nguyên — chứng tỏ con số đó không được sinh ra từ phép đếm ảnh thật trên 5.000 mẫu validation. Kiểm thử `tests/test_week4_audit.py` kiểm tra điều kiện `abs(acc * N - round(acc * N)) <= 1e-6` cho cả train và validation.

4. **Vì sao checkpoint tốt nhất dùng validation và hòa lấy epoch sớm hơn? `last.pt` dùng làm gì?**
   - Không bao giờ chạm tập test khi chọn mô hình để tránh rò rỉ dữ liệu đánh giá (*test leakage*). Trong `src/train.py` dòng 274: `improved = val['accuracy'] > best` (dùng dấu `>` nghiêm ngặt thay vì `>=`), nên nếu epoch sau hòa điểm validation thì giữ mô hình hội tụ sớm hơn (ít nguy cơ overfit hơn). `last.pt` lưu trạng thái cuối mỗi epoch (gồm `model`, `optimizer`, `scheduler`, `history`, và `random` RNG của Python/NumPy/PyTorch/CUDA + `DataLoader.generator`) để nối lại (`--resume`) chính xác khi gián đoạn.

5. **So sánh 100 epoch ở ba mức dữ liệu có cùng số bước không? Tính $\lceil N/128 \rceil \times 100$.**
   - Không cùng số bước cập nhật! Với `batch_size=128` và `drop_last=False`:
     - 10% ($N=4.500$): $\lceil 4500/128 \rceil = 36$ bước/epoch $\rightarrow$ **3.600 bước**.
     - 50% ($N=22.500$): $\lceil 22500/128 \rceil = 176$ bước/epoch $\rightarrow$ **17.600 bước**.
     - 100% ($N=45.000$): $\lceil 45000/128 \rceil = 352$ bước/epoch $\rightarrow$ **35.200 bước**.
   - Đây là giới hạn quan trọng đã nêu trong báo cáo: tăng dữ liệu ở cùng số epoch đồng thời tăng số bước cập nhật gradient của AdamW/SGD.

6. **ViT dùng patch 4×4 nên có 64 patch và CLS thành 65 token. Vì sao attention có shape `[B, 4, 65, 65]`?**
   - Trong `MultiHeadSelfAttention.forward` (`src/models/vit.py` dòng 87–100): đầu vào `x` có shape `[B, 65, 96]`. Tầng `self.qkv(x)` chiếu lên `[B, 65, 288]`, reshape thành `[B, 65, 3, 4, 24]` rồi permute sang `[3, B, 4, 65, 24]`. Mỗi tensor $Q, K, V$ có shape `[B, 4, 65, 24]`. Tích `q @ k.transpose(-2, -1)` nhân `[B, 4, 65, 24]` với `[B, 4, 24, 65]` cho ma trận tương quan giữa mọi cặp trong 65 token trên 4 head song song: **`[B, 4, 65, 65]`**, và `softmax(dim=-1)` làm tổng mỗi hàng bằng $1$.

7. **Trích CLS → patch block cuối trung bình head khác attention rollout thế nào? Heatmap 8×8 ánh xạ lên ảnh gốc ra sao?**
   - Trong `src/visualize.py` dòng 94: `weights[0, :, 0, 1:].mean(0)` lấy hàng `0` (CLS là query) nhìn sang cột `1..64` (64 patch là key) tại **block cuối cùng (`[-1]`)**, trung bình cộng trên 4 head $\rightarrow$ vector 64 phần tử, `reshape(8, 8)` rồi phóng lên $32\times32$ bằng `np.repeat(..., 4, axis=0/1)` (mỗi patch $4\times4$ giữ nguyên mức nhiệt của ô đó). Khác với **attention rollout** (nhân tích lũy ma trận attention có cộng ma trận đơn vị $0.5 A + 0.5 I$ qua tất cả các tầng từ 1 đến 4 để tính đường truyền thông tin qua kết nối tắt residual), trung bình block cuối phản ánh trực tiếp trọng số chú ý ngay trước tầng phân loại CLS.

8. **Nếu bỏ positional embedding, số tham số giảm $65\times96$. Vì sao cần cặp seed/recipe giống baseline khi phân tích ablation?**
   - Trong `VisionTransformer.__init__` (`src/models/vit.py` dòng 202–206), khi `use_pos_embed=False`, `pos_embed` được đăng ký bằng `self.register_buffer('pos_embed', torch.zeros(1, 65, 96))` thay vì `nn.Parameter`, giảm đúng $65 \times 96 = 6.240$ tham số ($274.282 \rightarrow 268.042$). Phải giữ nguyên cặp `(split_seed, train_seed)` và recipe AdamW để loại trừ biến nhiễu do khác tập ảnh hay siêu tham số, đảm bảo mọi thay đổi hiệu năng đến từ việc bỏ thông tin vị trí.

9. **Residual shortcut và projection trong ResNet làm gì? Vì sao logits đi trực tiếp vào CrossEntropy?**
   - Trong `src/models/resnet.py` của Minh, mỗi `BasicBlock` tính $y = \operatorname{ReLU}(F(x) + \text{shortcut}(x))$ giúp đạo hàm truyền thẳng qua phép cộng tắt, tránh suy giảm gradient. Khi sang stage mới (stride=2, nhân đôi số kênh $16\rightarrow32\rightarrow64$), kích thước không gian giảm một nửa và kênh tăng gấp đôi nên nhánh tắt dùng tích chập `Conv2d(in, out, kernel_size=1, stride=2, bias=False)` kèm `BatchNorm2d` (projection shortcut) để khớp shape. Cả hai mạng trả **logits thô** vì `nn.functional.cross_entropy` gộp `LogSoftmax + NLLLoss` bằng công thức ổn định số học `logsumexp`, tránh tràn số dưới (*underflow*) nếu tính `softmax` rời.

10. **Vì sao ResNet mean ba seed và ViT seed đầu không đủ làm kết luận so sánh ba seed? Cần bổ sung bằng chứng nào?**
    - Một seed đơn lẻ không cho phép ước lượng phương sai do chọn tập con và khởi tạo ngẫu nhiên. Sau khi chạy thật thêm 2 lượt ViT 10% (`seed 43, 44`), nhóm đã đủ so sánh 3-seed ở mức 10%; để hoàn tất bảng so sánh 3-seed ở 50% và 100%, cần chạy nốt 4 lượt `vit_{50,100}_seed{43,44}` từ checkpoint GPU thực tế và kiểm tra khớp log.

11. **Resume cần optimizer, scheduler và RNG nào? Vì sao chỉ đặt lại seed chưa đủ?**
    - Nếu chỉ gọi `seed_everything(seed)` ở giữa chừng (ví dụ epoch 50), bộ sinh số ngẫu nhiên sẽ quay lại trạng thái của epoch 1 chứ không phải trạng thái tại đầu epoch 51; ngoài ra các bộ đệm moment bậc 1 và bậc 2 (`exp_avg`, `exp_avg_sq`) của AdamW hoặc vận tốc momentum của SGD sẽ mất sạch! Vì vậy `src/checkpoint.py` lưu đầy đủ `model`, `optimizer`, `scheduler`, `random.getstate()`, `np.random.get_state()`, `torch.get_rng_state()`, `torch.cuda.get_rng_state_all()` và `generator.get_state()` của DataLoader.

12. **Có thể suy nguyên nhân từ một attention map hay từ chênh lệch accuracy của hai recipe không? Thiết kế thí nghiệm bổ sung thế nào?**
    - Không thể kết luận nhân quả tuyệt đối! Attention map chỉ hiển thị trọng số tương quan nội tại ở một lớp, còn token đã pha trộn thông tin qua 3 lớp trước đó. Tương tự, ViT dùng AdamW còn ResNet dùng SGD, và cố định 100 epoch làm thay đổi tổng số bước cập nhật. Để tách riêng các yếu tố, thí nghiệm mở rộng sẽ là: (a) cố định tổng số bước cập nhật (step-matched budget, ví dụ cùng 17.600 bước ở mọi tỷ lệ), và (b) so sánh thêm kiến trúc lai cục bộ như Swin Transformer trên đủ 3 seed.

13. **Test cuối có được dùng để đổi LR/chọn seed không? Cách lưu registry và checkpoint hash giúp truy vết kết quả thế nào?**
    - Tuyệt đối không! `results/hien_week4/checkpoint_registry.csv` khóa trước đường dẫn `best.pt` và mã băm `checkpoint_sha256` từ bước validation. Script `scripts/evaluate_final.py` kiểm tra mọi bản ghi đạt `saved_log_consistent` và khớp đúng SHA-256 trước khi cho phép chạy `--execute` trên 10.000 ảnh test.

---

## Phần 3. Tình huống Đổi Code / Siêu tham số Trực tiếp Khi Vấn đáp (Dành riêng cho `src/models/vit.py`)

- **Tình huống 1: Giảng viên yêu cầu đổi `patch_size` từ `4` sang `8` trên ảnh $32\times32$, điều gì xảy ra với shape tensor và số tham số?**
  - `num_patches` giảm từ $(32/4)^2 = 64$ xuống $(32/8)^2 = 16$, chuỗi token giảm từ $65$ xuống $17$ (nhanh hơn $(65/17)^2 \approx 14{,}6$ lần ở phép nhân attention!).
  - Tuy nhiên `PatchEmbedding.projection` đổi thành `Conv2d(3, 96, kernel_size=8, stride=8)`: số tham số tầng chiếu tăng từ $3 \times 16 \times 96 + 96 = 4.704$ lên $3 \times 64 \times 96 + 96 = 18.528$ ($+13.824$), trong khi `pos_embed` giảm từ $65 \times 96 = 6.240$ xuống $17 \times 96 = 1.632$ ($-4.608$). Tổng tham số tăng ròng $+9.216$ (thành **283.498** tham số), nhưng độ phân giải không gian mỗi patch quá thô ($8\times8$ chiếm $1/16$ diện tích ảnh) làm giảm độ chi tiết biên vật thể.

- **Tình huống 2: Đổi từ phân loại bằng `CLS` token sang Global Average Pooling (GAP) trên 64 patch token thì sửa dòng nào trong `src/models/vit.py`?**
  - Ở `forward()` (dòng 269): thay `cls_out = x[:, 0]` bằng `patch_mean = x[:, 1:].mean(dim=1)` (hoặc bỏ ghép `cls_token` ngay từ đầu thì `pos_embed` chỉ còn `[1, 64, 96]`, giảm $2 \times 96 = 192$ tham số). Khi đó `head(norm(patch_mean))` tổng hợp trung bình đồng đều cả 64 patch tương tự GAP của ResNet.

- **Tình huống 3: Vì sao chia `scale = head_dim ** -0.5` ($1/\sqrt{24}$)? Nếu bỏ hệ số này chuyện gì xảy ra?**
  - Khi $q, k$ có phương sai đơn vị, tích vô hướng $\sum_{i=1}^{d_k} q_i k_i$ có phương sai bằng $d_k = 24$ (độ lệch chuẩn $\approx 4{,}9$). Nếu không chia $\sqrt{24}$, các giá trị logit trước `softmax` sẽ rất lớn về trị tuyệt đối, đẩy `softmax` vào vùng bão hòa (một phần tử $\approx 1$, các phần tử còn lại $\approx 0$), làm đạo hàm $\partial \text{softmax}/\partial z$ tiến về $0$ (vanishing gradient cho $Q$ và $K$).
