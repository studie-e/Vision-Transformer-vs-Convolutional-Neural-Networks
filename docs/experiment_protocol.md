# Giao thức ViT–ResNet cho tuần 2

Các điều kiện dữ liệu và đánh giá dưới đây đã đối chiếu với code. Số epoch và recipe chính thức còn chờ nhóm chốt. Hai YAML pilot dùng một epoch để kiểm tra tích hợp.

| Hạng mục | Điều kiện |
|---|---|
| Dữ liệu | CIFAR-10, 45.000 train và 5.000 validation phân tầng từ tập train gốc. Giữ riêng 10.000 test chính thức. |
| Tỷ lệ | 10%, 50%, 100% tương ứng 4.500, 22.500, 45.000 ảnh. Các tập con lồng nhau và dùng cùng danh sách chỉ số giữa hai mô hình trong từng cặp. |
| Seed đề xuất | (split, train) = (2026, 42), (2027, 43), (2028, 44). Sáu lượt P0 dùng cặp đầu tiên. |
| Mô hình | ViT 274.282 tham số, ResNet 272.474 tham số. Chênh 0,66% theo mẫu số ResNet. |
| Tiền xử lý | Cùng loader và transform. Crop có padding và lật ngang chỉ trên train. Validation dùng ToTensor và Normalize. |
| Ngân sách | Cùng số epoch tối đa cho sáu cấu hình. Cho phép recipe riêng từng kiến trúc, giữ cố định qua các tỷ lệ và dùng cùng ngân sách tuning trên validation. Ghi optimizer, LR, scheduler, batch size và weight decay trước loạt chính. |
| Chọn checkpoint | Top-1 validation cao nhất, hòa điểm giữ epoch sớm hơn. Dùng best.pt để đánh giá. Resume từ last.pt của cùng run. |
| Tập test | Chỉ dùng sau khi khóa lựa chọn mô hình, recipe, seed và checkpoint. |

Ma trận 18 lượt và trạng thái bàn giao nằm ở [run_matrix.csv](../results/hien_week2/run_matrix.csv). Lan Anh bàn giao ViT, Minh bàn giao ResNet. Mỗi lượt cần thư mục có config.yaml, metadata.json, metrics.csv, best.pt và last.pt. Kiểm tra đúng cặp seed, tỷ lệ và manifest, đối chiếu tham số, số bước và accuracy từ checkpoint trước khi đưa vào bảng. Ghi đường dẫn run trong ma trận sau khi đối chiếu.

Phần nhóm cần xác nhận trước loạt chính gồm số epoch, recipe từng mô hình và lịch GPU. Lưu bản YAML thực tế trong mỗi run. Hai lượt pilot hiện có đủ để kiểm tra công cụ xuất hình, chưa đủ để suy ra tác động của lượng dữ liệu.
