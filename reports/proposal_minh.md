# Nội dung Minh gửi Hiền để ghép proposal

Đây là phần CNN/thiết lập dự kiến, cần ghép với phần dữ liệu và ViT; chưa phải proposal ACL hoàn chỉnh.

## Baseline CNN và cách so sánh

Nhóm sử dụng ResNet cho ảnh RGB 32×32, dựa trên kiến trúc residual của He và cộng sự (2015). Mô hình gồm stem tích chập 3×3, ba stage basic block, global average pooling và lớp tuyến tính phân loại 10 lớp. Bản thử nghiệm ban đầu có ba block mỗi stage, độ rộng 16/32/64 và 272.474 tham số trainable; dùng projection 1×1 ở shortcut khi thay đổi kích thước. Nhóm sẽ điều chỉnh độ rộng sau khi có số tham số ViT tự cài để hai mô hình đạt quy mô tương đương; ngưỡng chênh lệch nội bộ đề xuất là 10%.

ViT và ResNet đều khởi tạo từ đầu, cùng dữ liệu/tập con 10%, 50%, 100%, cùng tiền xử lý và cách đánh giá. Minh phụ trách ResNet, môi trường/trainer và thí nghiệm CNN; Lan Anh phụ trách ViT; Hiền chuẩn bị dữ liệu và tổng hợp đánh giá. Cấu hình huấn luyện của từng mô hình được chọn trên validation với ngân sách thử tương đương, khóa trước loạt thí nghiệm chính; test chỉ dùng cho đánh giá cuối.

## Thiết lập và theo dõi

Môi trường đã kiểm tra: Windows, Python 3.13.7, PyTorch 2.11.0+cu128, CUDA runtime 12.8 và NVIDIA RTX 5080. Trainer chung dùng CrossEntropyLoss trên logits; hỗ trợ SGD/AdamW, cosine learning-rate schedule, chọn checkpoint bằng validation accuracy và resume tại ranh giới epoch. Mỗi lượt lưu seed, cấu hình, trạng thái mã nguồn, loss/accuracy, thời gian và bộ nhớ GPU; TensorBoard hỗ trợ theo dõi. Số epoch và siêu tham số chính sẽ được chốt sau pilot của cả hai mô hình.

Pilot tuần 1 chỉ dùng dữ liệu chẩn đoán riêng và ResNet; chưa thay cho split chính của Hiền hay kết quả so sánh ViT–CNN. Không đưa accuracy pilot vào bảng kết quả chính. Số đo pilot đã chạy được lưu trong `results/pilot.md` để cả nhóm cân đối lịch GPU.

## Phạm vi AI cần khai báo

AI hỗ trợ xây dựng mã ResNet/trainer/checkpoint, kiểm thử và bản nháp tài liệu. Minh cần đọc, kiểm tra, điều chỉnh và giải thích các đoạn mã sử dụng. Không ghi Minh đã đọc bài hoặc đã tích hợp ViT khi chưa hoàn tất thực tế.
