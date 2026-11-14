# Bàn giao tuần 2 của Hiền

## Đối chiếu pilot

Hai lượt CPU dùng cùng manifest 2026, train seed 42, 4.500 train và 5.000 validation. Phần đánh giá nạp lại best.pt và kiểm tra kết quả khớp log validation. ViT đạt 1.017/5.000 và ResNet đạt 846/5.000 ở pilot một epoch. Lượt ViT được chạy lại tại máy này từ configs/vit.yaml. Dùng [bảng validation](../results/hien_week2/pilot/validation.csv) để truy về log, config và hash checkpoint.

Pilot GPU của Minh dùng adapter và tập validation khác. Số đo GPU của lượt này phục vụ dự trù tài nguyên. Hai lượt CPU mới kiểm tra cùng pipeline, chưa có đủ các tỷ lệ và seed để kết luận mô hình nào học hiệu quả hơn.

## Hình validation

- [ViT attention](../results/hien_week2/pilot/2_vit_0.10_attention.png) hiển thị mẫu đúng và sai đầu tiên theo thứ tự manifest. Mỗi mẫu có chỉ số CIFAR-10, nhãn thật và nhãn dự đoán. Lấy attention từ CLS tới 64 patch ở block cuối, trung bình bốn head rồi đưa về lưới 8×8 theo thứ tự hàng. Mỗi ô phủ đúng patch 4×4 trên ảnh RGB gốc. Các màu biểu diễn mức attention tương đối trong từng ảnh.
- [Accuracy theo lượng dữ liệu](../results/hien_week2/pilot/accuracy_data.png) hiện chỉ có hai điểm ở 10%. Curve từng mô hình chỉ có một epoch. Confusion matrix đếm số ảnh theo hàng nhãn thật và cột dự đoán.

Cài matplotlib vào môi trường train nếu cần, rồi chạy từ gốc repo:

```powershell
python -X utf8 -m src.visualize --run runs/hien_week1_smoke --run runs/hien_week2_vit_preview --out results/hien_week2/pilot
```

Khi đủ các run chính, truyền mỗi thư mục bằng một đối số --run, chọn --label validation và một thư mục xuất khác. Dùng ma trận để kiểm tra lượt còn thiếu. Trước khi đọc hình chung, đối chiếu recipe, ngân sách epoch và cặp seed của từng run với bản nhóm đã chốt.

## Dữ liệu và đánh giá cho bản viết đầu

Chúng tôi dùng CIFAR-10 và tách phân tầng 50.000 ảnh huấn luyện gốc thành 45.000 ảnh train và 5.000 ảnh validation. Tập test chính thức gồm 10.000 ảnh được giữ riêng. Với mỗi seed, các tập con 4.500, 22.500 và 45.000 ảnh lồng nhau, giữ cân bằng lớp và được dùng chung giữa ViT và ResNet. Augmentation chỉ áp dụng khi train.

Chúng tôi tính top-1 accuracy bằng tổng số ảnh dự đoán đúng chia cho tổng số ảnh. Checkpoint tốt nhất được chọn bằng validation accuracy và giữ epoch sớm hơn khi hòa điểm. Các lượt được ghép theo tỷ lệ, split seed và train seed. Báo cáo ghi số tham số, thời gian và bước cập nhật cùng accuracy. Độ lệch chuẩn mẫu chỉ tính khi có từ hai seed trở lên. Tập test được dùng sau khi khóa toàn bộ lựa chọn.

## Đọc chéo trainer và ResNet

- Trainer cộng số dự đoán đúng theo ảnh, chia cho tổng số ảnh, và tính loss có trọng số theo kích thước batch. Evaluator độc lập tái hiện đúng accuracy trên hai pilot.
- ResNet trả logits 10 lớp trực tiếp cho CrossEntropy. Checkpoint tốt nhất dùng điều kiện tăng nghiêm ngặt của validation accuracy. Checkpoint cuối giữ trạng thái optimizer, scheduler và RNG để nối lại tại ranh giới epoch. Khi resume, trainer kiểm tra config và hash manifest.
- Các bài kiểm thử resume và kết quả pilot Minh cung cấp bằng chứng trong phạm vi đã chạy. Thời gian CPU và GPU được ghi riêng. Recipe P0 đã được chốt trong giao thức. Nhóm cần xác nhận nguồn lưu các checkpoint thực nghiệm.

## Trạng thái

Đã có bản giao thức, ma trận, công cụ đọc checkpoint và sinh hình, attention thử và bản viết phương pháp. Việc đọc chéo hình với Lan Anh và đối chiếu bảng với Minh còn chờ phản hồi. Checkpoint C còn mở vì 0/6 lượt chính đã được bàn giao.
