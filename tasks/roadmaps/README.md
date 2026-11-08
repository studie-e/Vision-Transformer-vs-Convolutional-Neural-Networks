# Lộ trình chung — 4 tuần, 3 người

Phân công giữ nguyên bảng gốc, chỉ đổi chỗ Minh và Hiền:

| Người | Trách nhiệm xuyên suốt |
|---|---|
| **Lan Anh** | Tự cài ViT, trực tiếp train ViT ở ba tỷ lệ và nhiều seed; tự làm/train ablation; viết Method–ViT và slide |
| **Minh** | ResNet, trainer, môi trường RTX 5080, trực tiếp train ResNet và Swin; README, tái lập, demo và slide |
| **Hiền** | Dữ liệu/tập con, đánh giá, attention, bảng/hình, phân tích; tổng hợp proposal, milestone, báo cáo ACL và slide |

Lộ trình riêng: [Lan Anh](lan-anh.md), [Minh](minh.md), [Hiền](hien.md). Yêu cầu môn học: [plan.md](../plan.md). Trạng thái nghiệm thu cập nhật tập trung tại [todo.md](../todo.md); bảng ngày dưới đây là lịch thực hiện, không phải bộ trạng thái thứ hai.

## Quy ước thời gian

Ngày 1–28 là ngày tương đối trong 4 tuần đầu của kế hoạch. Chưa gắn ngày lịch vì nhóm chưa cung cấp ngày bắt đầu. Đối chiếu hạn giảng viên trước khi triển khai: proposal cuối tuần 1, milestone cuối tuần 3, bản cuối tuần 6. Nhóm hoàn thiện nội bộ ngày 28.

Mỗi ngày là một phiên công việc chính; có thể ghép các ngày nếu lịch học yêu cầu, nhưng giữ thứ tự phụ thuộc và mốc bàn giao. Ngày 7/14/21/28 dành cho kiểm tra, hạn nộp và dự phòng. Thời gian GPU tách khỏi thời gian làm việc cá nhân; chưa cam kết số giờ khi chưa có pilot.

## Các mốc phối hợp

| Mốc | Lan Anh | Minh | Hiền | Điều kiện chuyển bước |
|---|---|---|---|---|
| **Ngày 1** | Thiết kế ViT, đọc bài | Kiểm tra GPU/môi trường, thiết kế ResNet/trainer | Chốt dữ liệu, khung ACL | Thống nhất ảnh đầu vào và logits 10 lớp |
| **Ngày 3** | Các thành phần ViT ban đầu | Trainer bản đầu, ResNet ban đầu | Bàn giao split/tập con và metric | Dữ liệu dùng chung, không giao nhau |
| **Ngày 5** | ViT forward/backward, số tham số | ResNet/trainer tích hợp, đối chiếu tham số | Kiểm tra dữ liệu và mẫu log/bảng | Hai mô hình sẵn sàng chạy pilot |
| **Ngày 6–7** | Pilot ViT; đoạn proposal | Pilot ResNet, ngân sách GPU; đoạn proposal | Ghép và nộp proposal ≤2 trang | Repo/log cho giảng viên truy cập từ tuần 1 |
| **Ngày 9** | Khóa ViT/recipe | Khóa ResNet/recipe, lịch GPU | Xác nhận split/metric/giao thức | Bắt đầu loạt chính |
| **Ngày 13–14** | Đủ 3 lượt ViT seed đầu | Đủ 3 lượt ResNet seed đầu | Bảng validation, attention thử | Kiểm tra 6 lượt rồi mới lặp seed |
| **Ngày 19–20** | Mục tiêu đủ 9 lượt ViT, chuẩn bị ablation | Mục tiêu đủ 9 lượt ResNet, chuẩn bị Swin | Bảng 18 lượt, nháp milestone | Ưu tiên bảng chính trước mở rộng |
| **Ngày 21** | Xác nhận phần ViT | Xác nhận CNN/tiến độ GPU | Nộp milestone ≤4 trang | Nêu đúng kết quả đã có, không chờ mở rộng |
| **Ngày 22–23** | Hoàn tất ablation | Hoàn tất Swin theo ngân sách | Chuẩn bị đánh giá cuối | Khóa phương pháp và checkpoint |
| **Ngày 24** | Giao checkpoint ViT/ablation đã khóa | Giao checkpoint ResNet/Swin đã khóa | Đánh giá test cuối, sinh bảng/hình | Không tuning theo test |
| **Ngày 25** | Giao Method–ViT/ablation | Giao CNN/Swin/thiết lập/README | Ghép PDF và Analysis | Đủ mục bắt buộc, đúng ACL |
| **Ngày 26–27** | Đọc chéo dữ liệu/đánh giá, luyện slide | Kiểm tra ViT, tái lập/demo, luyện slide | Kiểm tra trainer/CNN, ghép slide | Thuyết trình 10–12 phút, vấn đáp chéo |
| **Ngày 28** | Xác nhận sản phẩm cá nhân | Đóng gói code/log/môi trường | Đóng gói PDF/slide | Bộ sản phẩm nội bộ hoàn chỉnh |

## Thí nghiệm thống nhất

- Dữ liệu chính: CIFAR-10 RGB 32×32, 10 lớp. Tách cố định 45.000 train / 5.000 validation; giữ test chính thức 10.000 ảnh.
- Tỷ lệ 10/50/100% tính trên 45.000 train: 4.500 / 22.500 / 45.000 ảnh. Ghi rõ mẫu số trong báo cáo.
- Seed chia train/val đề xuất: **2026**. Ba seed lượt chạy đề xuất: **17, 42, 2026**; ghi riêng seed lấy tập con và seed huấn luyện. Xác nhận các giá trị này ở ngày 1.
- Tập con phân tầng, lồng nhau theo từng seed; ViT và ResNet dùng đúng cùng danh sách ảnh. ViT và ResNet chính train từ đầu.
- Mục tiêu nội bộ: chênh lệch tham số không quá 10%, tính bằng `abs(P_vit - P_resnet) / max(P_vit, P_resnet)`. Đây là đề xuất của nhóm, không phải ngưỡng giảng viên ấn định.
- Tuning trên validation, cùng ngân sách thử cấu hình cho hai mô hình. Khóa recipe từng kiến trúc và epoch trước loạt chính; giữ cố định theo tỷ lệ và seed.
- Kết quả sơ bộ và attention/lỗi tuần 2–3 lấy từ validation. Đánh giá test cuối ngày 24 sau khi khóa toàn bộ phương pháp/checkpoint; không chọn seed đẹp nhất hoặc tuning theo test.
- Bảng cuối có từng seed và trung bình ± độ lệch chuẩn mẫu, số tham số, thời gian train, số bước. Chỉ một seed thì ghi kết quả thăm dò, không tạo độ lệch chuẩn.

## Hàng đợi GPU và số lượt chạy

Minh điều phối lịch dùng RTX 5080. **Lan Anh trực tiếp chạy ViT/ablation; Minh trực tiếp chạy ResNet/Swin.** Hai người đặt lịch trước, không tự mở hai lượt tranh GPU.

| Cửa sổ dự kiến | Lượt chạy | Số lượt | Người chạy |
|---|---|---:|---|
| Ngày 9–12 | ViT/ResNet × 10/50/100% × seed 17 | 6 | Lan Anh 3 ViT; Minh 3 ResNet |
| Ngày 15–16 | ViT/ResNet × 10/50/100% × seed 42 | 6 | Lan Anh 3 ViT; Minh 3 ResNet |
| Ngày 17–19 | ViT/ResNet × 10/50/100% × seed 2026 | 6 | Lan Anh 3 ViT; Minh 3 ResNet |
| Ngày 20–23, sau bảng chính | ViT bỏ positional embedding ở 50% × 3 seed | 3 | Lan Anh |
| Ngày 20–23, sau bảng chính | Swin ở 10% và 100% × 3 seed | 6 | Minh |

Đích dự kiến **27 lượt**, chưa gồm smoke test, pilot/tuning và tái lập đại diện. Run ID gợi ý: `vit_d10_s17`, `resnet_d10_s17`, `vit_no_pos_d50_s17`, `swin_d10_s17`.

Ngày chạy là cửa sổ dự kiến. Sau pilot, ước lượng tổng GPU từ số epoch × thời gian/epoch cho từng mô hình/tỷ lệ, cộng đánh giá/lưu và khoảng 20% dự phòng. Nếu thiếu ngân sách, điều chỉnh epoch nhất quán trước loạt chính. Nếu chậm sau đó, ưu tiên 18 lượt chính, giảm phạm vi Swin trước; ghi trung thực phạm vi thực hiện.

## Gói bàn giao và cách xử lý trễ

| Người giao → nhận | Gói bàn giao |
|---|---|
| Hiền → Lan Anh, Minh | Mã dữ liệu, manifest split/tập con, seed, preprocessing, số lượng theo lớp |
| Minh → Lan Anh | Trainer, checkpoint/resume, cấu hình chung, cách ghi log và lịch GPU |
| Lan Anh → Hiền, Minh | ViT/cấu hình/số tham số; cách lấy attention; checkpoint và log các lượt ViT/ablation |
| Minh → Hiền, Lan Anh | ResNet/Swin/cấu hình/số tham số; checkpoint/log, thời gian, số bước |
| Hiền → cả nhóm | Bảng từng seed/mean ± std, biểu đồ, attention, lỗi, bản ghép ACL |
| Cả ba → Hiền | Đoạn báo cáo hoàn chỉnh, hình/bảng, BibTeX, nguồn mã/AI đã dùng, đóng góp thực tế |

Một lượt chạy cần có ID, trạng thái, model/tỷ lệ/seed, manifest, config, commit, params, checkpoint, best epoch/val, log, thời gian và số bước. Checkpoint/dataset lớn dùng link ngoài Git. Đường dẫn mã trong lộ trình là sản phẩm dự kiến, chưa phải file đã triển khai.

Nếu có nguy cơ trễ hơn một ngày, báo cả nhóm kèm lỗi và file đang có. Lỗi ViT/ablation: Lan Anh; lỗi ResNet/Swin/trainer/môi trường: Minh; lỗi dữ liệu/metric/attention: Hiền. Ghi rõ run bị ảnh hưởng và cần chạy lại; không sửa số bảng bằng tay.

Ngày 28: PDF ≤8 trang nội dung, Limitations, đóng góp/AI, repo/README/môi trường, script tái lập, log/link checkpoint, slide 10–12 phút và demo chạy thử. Tuần 5 rà soát/tái lập, tuần 6 luyện vấn đáp và nộp theo lịch môn. Mỗi người viết phần mình; Hiền tổng hợp.
