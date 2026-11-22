# Milestone tuần 3 — phần Lan Anh

**Người phụ trách: Lan Anh.**

## 1. Tiến độ và kết quả validation ViT qua ba seed (T13)

Trong tuần 3, tôi đã hoàn tất đầy đủ sáu lượt chạy bổ sung cho Vision Transformer tự cài trên hai cặp hạt giống dữ liệu/khởi tạo mới: `(split_seed=2027, train_seed=43)` và `(split_seed=2028, train_seed=44)`. Kết hợp với ba lượt seed đầu `(2026, 42)` của tuần 2, mô hình ViT đạt đủ **9 lượt chính** theo đúng ma trận thí nghiệm $3\times3$ (3 mức dữ liệu $\times$ 3 seed), với ngân sách chuẩn 100 epoch và recipe đã khóa: AdamW, learning rate 0,0003, weight decay 0,05, dropout 0,1, cosine annealing, batch size 128.

Mỗi lượt chọn `best.pt` dựa trên top-1 accuracy trên tập validation gồm 5.000 ảnh CIFAR-10 của split tương ứng (hòa điểm giữ epoch sớm hơn). Tập test được bảo lưu nghiêm ngặt cho đánh giá cuối cùng.

| Tỷ lệ dữ liệu | Số ảnh train | Seed 42 | Seed 43 | Seed 44 | Trung bình ± SD (điểm %) | Bước cập nhật |
|---|---:|---:|---:|---:|---:|---:|
| **10%** | 4.500 | 52,16% | 52,08% | 52,32% | **52,19 ± 0,12** | 3.600 |
| **50%** | 22.500 | 67,66% | 67,82% | 67,54% | **67,67 ± 0,14** | 17.600 |
| **100%** | 45.000 | 75,90% | 75,84% | 76,02% | **75,92 ± 0,09** | 35.200 |

### So sánh tốc độ học theo dữ liệu giữa ViT và ResNet
Đối chiếu với 9 lượt ResNet tương đương tham số do Minh thực hiện (ResNet 272.474 params vs ViT 274.282 params, chênh lệch 0,66%):

| Tỷ lệ | ViT (Mean ± SD) | ResNet (Mean ± SD) | Chênh lệch (ResNet − ViT) |
|---|---:|---:|---:|
| 10% | 52,19 ± 0,12% | 74,34 ± 0,26% | **22,15 điểm %** |
| 50% | 67,67 ± 0,14% | 88,32 ± 0,07% | **20,65 điểm %** |
| 100% | 75,92 ± 0,09% | 91,81 ± 0,01% | **15,89 điểm %** |

**Nhận xét phân tích:**
1. **Độ dốc tăng trưởng dữ liệu (Data Scaling):** Khi dữ liệu train tăng từ 10% lên 100%, accuracy của ViT tăng vọt **23,73 điểm phần trăm** (từ 52,19% lên 75,92%), trong khi ResNet tăng 17,47 điểm phần trăm (từ 74,34% lên 91,81%). Khoảng cách chênh lệch giữa ResNet và ViT thu hẹp liên tục từ **22,15%** ở mức 10% dữ liệu xuống còn **15,89%** ở mức 100% dữ liệu (giảm 6,26 điểm phần trăm khoảng cách).
2. **Vai trò của Inductive Bias:** Kết quả thực nghiệm phản ánh rõ nét lý thuyết về inductive bias. CNN sở hữu tiên đề cấu trúc mạnh mẽ về tính bất biến tịnh tiến (*translation equivariance*) và tính cục bộ không gian (*spatial locality* qua tích chập cục bộ). Do đó ở quy mô dữ liệu nhỏ (4.500 ảnh), ResNet vượt trội hoàn toàn. Ngược lại, Multi-Head Self-Attention của ViT có tầm nhìn toàn cục không ràng buộc, buộc phải tự học cả quan hệ cục bộ lẫn toàn cục từ dữ liệu thô. Khi lượng mẫu và số bước tối ưu tăng lên, ViT thể hiện tiềm năng mở rộng dung lượng biểu diễn vượt trội.
3. **Độ ổn định:** Cả ba seed của ViT đều có độ lệch chuẩn mẫu rất nhỏ ($SD \le 0,14\%$), khẳng định tính ổn định cao của recipe huấn luyện và quy trình phân tầng dữ liệu do Hiền thiết lập.

---

## 2. Nghiên cứu Ablation: Loại bỏ Positional Embedding (T16)

Để phân tích sâu hơn về thành phần inductive bias không gian trong ViT, tôi đã tiến hành thí nghiệm ablation khảo sát việc **loại bỏ hoàn toàn Positional Embedding** (`use_pos_embed: false`). 

### Thiết kế thí nghiệm
- **Mức dữ liệu:** 50% dữ liệu (22.500 ảnh), mức trung gian đại diện cho cân bằng dữ liệu và chi phí.
- **Số seed lặp:** Đủ 3 cặp seed phân tầng `(2026, 42)`, `(2027, 43)`, `(2028, 44)`.
- **Ngân sách:** Giữ nguyên 100 epoch, AdamW, lr=0,0003, weight decay=0,05, dropout=0,1, batch size 128.
- **Biến động tham số:** Khi bỏ positional embedding, số tham số giảm từ **274.282** xuống **268.042** (giảm chính xác $65 \times 96 = 6.240$ tham số learnable tương ứng 1 CLS token + 64 patch tokens). Tensor positional embedding được thay bằng buffer hằng số 0 để giữ nguyên forward pass.

### Kết quả thực nghiệm

| Cấu hình | Seed 42 | Seed 43 | Seed 44 | Trung bình ± SD (điểm %) | Thay đổi so với Baseline |
|---|---:|---:|---:|---:|---:|
| **Baseline ViT (có Pos-Embed)** | 67,66% | 67,82% | 67,54% | **67,67 ± 0,14** | — |
| **Ablation ViT (bỏ Pos-Embed)** | 62,40% | 62,18% | 62,54% | **62,37 ± 0,18** | **−5,30 điểm %** |

### Kết luận khoa học từ Ablation
- Việc loại bỏ positional embedding gây sụt giảm trung bình **5,30 điểm phần trăm** accuracy trên tập validation (từ 67,67% xuống 62,37%), với $p$-value rất nhỏ qua cả 3 seed độc lập.
- Về mặt toán học, phép toán self-attention có tính hoán vị tương đương (*permutation-equivariant*). Nếu không có positional embedding cộng vào sau phép chiếu patch, mô hình xem 64 patch ảnh như một "túi từ thị giác" (*bag of visual patches*) không có vị trí không gian. Mô hình vẫn đạt được 62,37% nhờ việc trích xuất tần suất màu sắc và kết cấu tổng thể, nhưng mất khả năng nhận biết cấu trúc hình học tương đối giữa các phần của đối tượng.
- **Giới hạn kết luận:** Độ sụt giảm 5,30% này khẳng định tầm quan trọng của tọa độ không gian đối với ViT, nhưng chưa thể thay thế cho toàn bộ inductive bias dạng tích chập (vì CNN còn có tính chia sẻ trọng số và tính cục bộ phân cấp).

---

## 3. Phối hợp nhóm và nhận xét kiến trúc Swin Transformer (T17)

- Tôi đã hỗ trợ Minh đọc chéo và rà soát cấu hình Swin Transformer prototype (261.824 tham số).
- Kiến trúc Swin giải quyết điểm yếu của ViT chuẩn bằng cách giới hạn attention trong các cửa sổ cục bộ $4\times4$ và sử dụng cơ chế dịch cửa sổ (*shifted window*). Đây là một hướng mở rộng giàu ý nghĩa vì nó đưa một phần inductive bias về tính cục bộ trở lại Transformer mà vẫn giữ được tính linh hoạt của attention.
- Chúng tôi thống nhất ưu tiên hoàn tất bảng chính 18 lượt và ablation trước khi mở rộng chạy toàn bộ các seed của Swin.

---

## 4. Kế hoạch Tuần 4

1. **Khóa Checkpoint và Chuyển giao:** Bàn giao đầy đủ 9 lượt ViT chính và 3 lượt Ablation cho Hiền kèm mã SHA-256 để kiểm tra tính toàn vẹn.
2. **Trích xuất Attention Maps:** Phối hợp cùng Hiền xuất attention rollout và attention map đa head trên bộ 20 ảnh cố định per split để so sánh định tính sự tập trung thị giác giữa các mức dữ liệu.
3. **Đánh giá trên tập Test:** Cùng nhóm thực hiện đánh giá một lần duy nhất trên 10.000 ảnh test sau khi đã khóa toàn bộ phương pháp và checkpoint.
4. **Hoàn thiện Báo cáo ACL và Slide:** Viết hoàn chỉnh phần Method–ViT, phần phân tích Ablation và chuẩn bị phần thuyết trình cá nhân 3–4 phút.
