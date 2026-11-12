# Proposal — Đề tài 3: So sánh Vision Transformer và CNN trên CIFAR-10

> **Tệp này là bản Markdown để ghép nội dung nội bộ.**
> Bản nộp sẽ được biên soạn theo template ACL (LaTeX, hai cột, A4, không ẩn danh).
> Tệp LaTeX: `reports/proposal.tex` (source hiện có; giữ trống thông tin thành viên theo yêu cầu, điền trước khi nộp).

---

## Thông tin nhóm

| Họ và tên | MSSV | Vai trò |
|---|---|---|
| [Lan Anh] | [MSSV Lan Anh] | Cài đặt và huấn luyện ViT; ablation positional embedding |
| [Hiền] | [MSSV Hiền] | Dữ liệu, đánh giá, attention map, phân tích và ghép báo cáo |
| [Minh] | [MSSV Minh] | ResNet, trainer/checkpoint/môi trường; thử Swin |

---

## 1. Bài toán và động lực

**Bài toán:** So sánh Vision Transformer (ViT) tự cài đặt bằng PyTorch với ResNet có số tham số tương đương khi thay đổi lượng dữ liệu huấn luyện (10%, 50%, 100%) trên CIFAR-10. Phân tích sự thay đổi chênh lệch accuracy theo lượng dữ liệu và bàn về vai trò của inductive bias trong hai kiến trúc.

**Động lực:** CNN (ResNet) có inductive bias mạnh về tính địa phương và tương đương dịch chuyển ở lớp tích chập nhờ tích chập và chia sẻ trọng số, trong khi ViT dựa trên self-attention toàn cục và cần nhiều dữ liệu hơn để học các đặc trưng hiệu quả. Câu hỏi về điểm chuyển tiếp hiệu quả giữa hai kiến trúc theo lượng dữ liệu còn chưa được khảo sát đầy đủ trên bộ dữ liệu nhỏ.

**Câu hỏi nghiên cứu:** Khi tăng lượng dữ liệu huấn luyện từ 10% lên 100%, chênh lệch accuracy giữa ViT tự cài và ResNet tương đương tham số thay đổi thế nào? Nhận xét nào phù hợp với inductive bias của hai kiến trúc, và giới hạn nào khiến nhóm chưa thể kết luận rộng hơn?

---

## 2. Đầu vào và đầu ra

| Hạng mục | Mô tả |
|---|---|
| **Đầu vào** | Ảnh màu RGB 32×32, tập CIFAR-10 |
| **Đầu ra dự đoán** | Nhãn lớp (0–9) và phân phối xác suất 10 lớp |
| **Kết quả nghiên cứu** | Bảng so sánh accuracy theo lượng dữ liệu; attention map ViT; phân tích lỗi |

---

## 3. Dữ liệu

**Bộ dữ liệu:** CIFAR-10 \citep{krizhevsky2009learning}, ảnh RGB 32×32, 10 lớp, 60.000 ảnh.

**Phân chia:**
- Tách phân tầng 45.000 train / 5.000 validation từ tập train gốc 50.000 ảnh.
- Giữ nguyên 10.000 ảnh test chính thức; chỉ dùng đánh giá cuối sau khi khóa toàn bộ phương pháp.

**Tập con huấn luyện** (tính trên 45.000 ảnh train):
| Tỷ lệ | Số ảnh | Ảnh/lớp |
|---|---:|---:|
| 10% | 4.500 | 450 |
| 50% | 22.500 | 2.250 |
| 100% | 45.000 | 4.500 |

Tập con lồng nhau: 10% ⊂ 50% ⊂ 100%. Hai mô hình dùng **cùng danh sách ảnh** từ manifest được lưu theo seed. Ba seed được dùng để ước tính độ biến thiên kết quả. Augmentation (random crop, random horizontal flip) chỉ áp dụng trên train; val/test chỉ chuẩn hóa.

**Chỉ số chính:** Top-1 accuracy trên tập test, báo cáo mean ± std của 3 seed.

---

## 4. Mô hình và baseline dự kiến

### 4.1 ViT tự cài đặt — Lan Anh

ViT nhỏ tự cài đặt bằng PyTorch \citep{dosovitskiy2020image}: patch embedding (patch 4×4), positional embedding học được, Transformer encoder (multi-head self-attention + MLP), classification head. Không dùng mô hình ViT có sẵn hoàn chỉnh thay phần tự cài. Số tham số trainable sẽ được đếm và công bố; mục tiêu chênh lệch với ResNet không quá 10%.

### 4.2 ResNet tương đương tham số — Minh

ResNet cho ảnh nhỏ (CIFAR-style) \citep{he2015deep}: stem tích chập 3×3, ba stage basic block, global average pooling, lớp tuyến tính 10 lớp. Bản thử nghiệm: width 16, 3 block/stage, projection shortcut, **272.474 tham số trainable**. Độ rộng sẽ được điều chỉnh sau khi có số tham số ViT để đảm bảo tương đương. Khởi tạo từ đầu, không dùng pretrained.

### 4.3 Quy ước so sánh

Cả hai mô hình dùng: cùng bộ dữ liệu/tập con/seed, cùng tiền xử lý, cùng metric (top-1 accuracy); recipe từng mô hình (optimizer, lr, epoch) được chọn trên validation với ngân sách thử tương đương và khóa trước loạt thí nghiệm chính. Test không được dùng để chọn cấu hình hoặc seed.

---

## 5. Kế hoạch thí nghiệm

| Mức | Thí nghiệm | Số lượt | Phụ trách |
|---|---|---:|---|
| P0 — bắt buộc | ViT × 3 tỷ lệ × seed đầu tiên | 3 | Lan Anh |
| P0 — bắt buộc | ResNet × 3 tỷ lệ × seed đầu tiên | 3 | Minh |
| P1 — tin cậy | +2 seed cho cả 6 cấu hình | 12 | Lan Anh 6; Minh 6 |
| P2 — ablation | ViT bỏ positional embedding, 50% × 3 seed | 3 | Lan Anh; Hiền tổng hợp |
| P2 — mở rộng | Swin nhỏ, 10%/100% × 3 seed | 6 | Minh; Lan Anh hỗ trợ |

Tổng dự kiến 27 lượt chính. Phần mở rộng chỉ bắt đầu sau khi hoàn thành bảng ViT–ResNet (P0+P1).

---

## 6. Đánh giá và phân tích

- **Bảng chính:** accuracy mean ± std theo 3 seed cho từng (model, ratio); kèm số tham số, thời gian train và số bước cập nhật.
- **Biểu đồ data scaling:** accuracy theo 10%/50%/100%, so sánh hai mô hình.
- **Learning curve:** train/validation accuracy theo epoch để kiểm tra hội tụ và overfitting.
- **Attention map** (ViT): attention từng head/layer chồng lên ảnh gốc; tập ảnh cố định gồm mẫu đúng và sai.
- **Confusion matrix và phân tích lỗi:** nhóm ảnh theo (ViT đúng/sai) × (CNN đúng/sai); không chọn ví dụ đẹp theo kết quả.
- **Ablation positional embedding:** so sánh ViT có/không có positional embedding ở 50% dữ liệu.

---

## 7. Phân công

| Người | Phần kỹ thuật | Phần phân tích/báo cáo |
|---|---|---|
| **Lan Anh** | Tự cài ViT; train ViT và ablation | Method–ViT, hình kiến trúc, phần ablation |
| **Minh** | ResNet, trainer/checkpoint/môi trường; thử Swin | Method–CNN/Swin, thiết lập train, README |
| **Hiền** | Dữ liệu/split, script đánh giá/tổng hợp, attention map | Dữ liệu/đánh giá, Analysis, ghép báo cáo ACL |

Mỗi người tự viết phần mình; Hiền ghép và kiểm tra format. Hiền không tự tạo số liệu khi chưa có kết quả train.

---

## Tài liệu tham khảo

Xem `reports/references.bib`. Các bài chính:
- \citet{dosovitskiy2020image} — ViT
- \citet{he2015deep} — ResNet
- \citet{liu2021swin} — Swin Transformer
- \citet{krizhevsky2009learning} — CIFAR-10

---

## Khai báo AI

Proposal này được soạn thảo với sự hỗ trợ của AI (Antigravity/Gemini, Codex). Nội dung kỹ thuật dựa trên tài liệu gốc của nhóm (plan.md, todo.md, proposal_minh.md). Nhóm đọc, kiểm tra và chịu trách nhiệm toàn bộ nội dung.

