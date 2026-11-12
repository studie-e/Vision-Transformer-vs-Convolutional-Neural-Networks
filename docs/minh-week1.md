# Bàn giao tuần 1 — Minh

## Phạm vi và trạng thái

Ngày thực hiện: **12/11/2026**, theo lịch Asia/Saigon. Repo ban đầu chỉ có tài liệu, trên nhánh `Minh`. Lan Anh và Hiền xác nhận chưa có ViT/split chính. Minh xác nhận giảng viên không cần lời mời GitHub; không gửi lời mời hay tin nhắn ra ngoài.

Đã chuẩn bị ResNet ảnh nhỏ, trainer dùng chung, checkpoint/resume, log TensorBoard/CSV, cấu hình pilot, công cụ đếm/khớp tham số, README và kiểm thử. Bằng chứng chạy thực tế được ghi trong [kết quả pilot](../results/pilot.md); phần nào còn chờ được giữ chưa hoàn thành trong [todo](../tasks/todo.md).

## Bản đồ mã để Minh đọc và vấn đáp

| File | Trách nhiệm | Minh cần giải thích được |
|---|---|---|
| `src/models/resnet.py` | Stem 3×3; 3 stage residual; global average pooling; head 10 lớp | Vì sao có shortcut, vì sao đổi chiều cần projection; width ảnh hưởng tham số |
| `src/train.py` | Factory, seed, train/validation, optimizer/lr, log, chọn best checkpoint | CrossEntropy nhận logits; train khác eval; vì sao val loss tính trọng số theo số ảnh |
| `src/checkpoint.py` | Lưu state_dict và RNG; ghi file tạm rồi thay thế | Chỉ lưu trọng số chưa đủ để tiếp tục train; last khác best; resume tại epoch boundary |
| `src/pilot_data.py` | Adapter dữ liệu chẩn đoán, dùng tập train CIFAR | Vì sao đây chưa phải split chính; không lấy test làm validation; seed ảnh hưởng tập con |
| `src/inspect_model.py` | Đếm tham số, thử các width 4–128 để khớp mục tiêu | Công thức chênh lệch, vì sao chưa có ViT thì chưa chốt được width cuối |
| `tests/test_week1.py` | Kiểm tra gradient, học batch nhỏ, validation, resume, chống ghi đè, split pilot | Kiểm thử nào chứng minh resume giữ optimizer/scheduler/RNG, phạm vi CPU/GPU của bằng chứng |

## ResNet bản đầu

- Mô hình gốc tham khảo: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385), phần kiến trúc và CIFAR.
- Bản đang cài: stem 3×3 stride 1; ba stage kênh 16/32/64, mỗi stage 3 basic block, mỗi block 2 conv 3×3; downsample ở đầu stage 2/3; average pooling và linear 10 lớp.
- Shortcut dùng identity khi cùng shape; dùng conv 1×1 + BatchNorm khi đổi shape (projection/option B). Bài gốc dùng option A cho thí nghiệm CIFAR; cần khai báo khác biệt này.
- Tên đầy đủ khi báo cáo: **CIFAR-style ResNet, 3×3 basic blocks, width 16, projection shortcuts**. Số tham số trainable hiện tại **272.474**. Đây chưa phải width cuối khớp ViT.
- Mã được xây bằng PyTorch, có AI hỗ trợ; không dùng trọng số pretrained.

Ghi chú này hỗ trợ Minh đọc bài, không thay thế việc Minh tự đọc và trình bày lại. Mục T02 chưa đánh dấu hoàn thành thay cho người học.

## Quy ước bàn giao với Lan Anh và Hiền

1. Lan Anh gửi `build_model(config)` tạo ViT nhận batch RGB và trả logits 10 lớp; không softmax trong forward dùng để train. Cách xuất attention phải dùng được ngoài trainer.
2. Hiền gửi `build_loaders(config)` trả train/val DataLoader, batch gồm tensor ảnh và nhãn kiểu long. Phiên bản tuần 1 dùng `num_workers=0`, generator được seed và có manifest tập con cố định.
3. Minh gửi trainer và README. Config đầy đủ phải ghi model/kwargs, data factory, seed, epoch, optimizer, lr, momentum/weight decay; không đổi tổng epoch khi resume.
4. Chạy lại pilot hai mô hình, đếm tham số thật của ViT và khớp width ResNet trước tuần 2. Không tự lấy một ViT thư viện để thay phần tự cài của Lan Anh.

## Resume và cách kiểm tra

Theo [hướng dẫn PyTorch về checkpoint](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html), lưu trạng thái mô hình và optimizer; repo còn lưu scheduler và RNG để tiếp tục đúng thứ tự ngẫu nhiên. [Tài liệu tái lập PyTorch 2.11](https://docs.pytorch.org/docs/2.11/notes/randomness.html) là nguồn cho seed/generator và giới hạn tái lập.

`last.pt` chứa trạng thái cuối epoch. `best.pt` được chọn bằng validation accuracy, không bằng test. File CSV dựng từ lịch sử đã lưu trong checkpoint khi resume để tránh trùng epoch. TensorBoard tiếp tục cùng run và loại bước cũ vượt ranh giới checkpoint.

Kiểm thử CPU so sánh cùng seed: chạy liền 3 epoch và chạy 1 epoch rồi khởi tạo lại mô hình/optimizer, tải checkpoint, chạy tiếp 2 epoch. Có dropout để kiểm tra việc phục hồi RNG và DataLoader shuffle. Bằng chứng chạy GPU thực tế ghi riêng trong kết quả pilot, không suy ra từ kiểm thử CPU.

## Log, sao lưu và phần gửi Hiền

Mở log bằng `python -m tensorboard.main --logdir runs --host localhost`. [API SummaryWriter PyTorch 2.11](https://docs.pytorch.org/docs/2.11/tensorboard.html). Lệnh này chỉ mở log cục bộ; không tạo link chia sẻ ngoài. Sao lưu `runs/<run-id>/` ra nơi nhóm thống nhất, gồm config/metadata/metrics/TensorBoard/checkpoint. Repo chỉ lưu tài liệu, cấu hình, mã và bản tóm tắt nhỏ.

Đoạn nội dung dùng cho proposal nằm ở [proposal_minh.md](../reports/proposal_minh.md); mục BibTeX nằm ở [references_minh.bib](../reports/references_minh.bib). Hiền ghép vào template ACL, không nộp file Markdown này thay proposal PDF.

## Minh cần tự thực hiện sau khi nhận bàn giao

- Đọc bài ResNet và giải thích code cùng hai bạn; AI chưa thể xác nhận thay Minh đã hiểu bài.
- Nhận ViT/split chính rồi tích hợp, khớp tham số và khóa recipe với cả nhóm.
- Xác nhận nơi sao lưu/chia sẻ log; không cần mời GitHub theo chỉ dẫn mới của giảng viên.
- Gửi phần proposal cho Hiền, cùng kiểm tra PDF cuối và hạn nộp thực tế.
- Khi viết báo cáo, khai báo AI đã hỗ trợ cài ResNet/trainer/checkpoint/kiểm thử/tài liệu; Minh kiểm tra, điều chỉnh và chịu trách nhiệm giải thích.
