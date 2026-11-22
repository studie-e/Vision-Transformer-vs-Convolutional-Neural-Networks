# Bàn giao tuần 4 của Hiền

## Đối chiếu đầu vào

Đã nhận trong Git 18 bản ghi baseline, ba ablation và một Swin chạy thử. Máy hiện có hai checkpoint pilot CPU; các checkpoint chính còn nằm trên máy thực nghiệm.

Phát hiện trong log ViT/ablation tuần 3 có accuracy không tương ứng số ảnh đúng nguyên và một số thời gian âm. Ví dụ `vit_10_seed43_metrics.csv` epoch 1 ghi val_accuracy 0.2647 trên 5.000 ảnh, epoch 100 ghi train_seconds -0.0945237793272844. Cần lấy log/checkpoint gốc để xác minh. Các kiểm thử trước đây chỉ kiểm tra đủ epoch, cực đại và phép tổng hợp nên chưa bắt các lỗi này.

Quy tắc nghiệm thu mới kiểm tra count nguyên của accuracy train/val, giá trị hữu hạn, thời gian không âm, epoch, bước cập nhật, best epoch, total time và hash manifest. Các kiểm tra dựa trên định nghĩa metric hiện tại trong trainer.

## Sản phẩm

- `scripts/reproduce_week4.py` sinh lại `results/hien_week4/`, gồm per_seed, summary, audit_issues, checkpoint_registry, audit.json và data scaling.
- Summary giữ cả thống kê đã báo cáo và thống kê của các log qua kiểm tra. Đây là validation. Trạng thái nhất quán log còn cần kiểm tra checkpoint/config gốc để nghiệm thu độc lập.
- Báo cáo ACL tại `reports/final.tex`, phần Analysis và Limitations trong `reports/sections/`. Bản thảo ghi rõ tình trạng dữ liệu, Swin một epoch và test còn chờ. Tên/MSSV để trống theo yêu cầu trước.
- Slide tại `slides/week4.tex`, trình bày 11 phút với lịch và ghi chú bên dưới.
- PDF báo cáo 3 trang, Abstract 181 từ; PDF slide 10 trang. Đã biên dịch và render để kiểm tra bố cục. Source TeX đi kèm để nhóm chỉnh sửa.
- `scripts/evaluate_final.py` kiểm tra danh sách checkpoint trước khi chạy test với cờ `--execute`. `src/predict.py` xuất nhãn và xác suất từ ảnh 32×32 theo đúng transform của checkpoint.

## Lệnh thực hiện

Trong môi trường đã cài requirements:

```powershell
python -X utf8 -m scripts.reproduce_week4
python -X utf8 -m scripts.evaluate_final
python -X utf8 -m scripts.package_hien_week4
# Sau khi đã xử lý lỗi dữ liệu, khóa danh sách và có đủ checkpoint/config:
python -X utf8 -m scripts.evaluate_final --execute --device cuda
python -X utf8 -m src.predict --image path/to/image32.png --checkpoint path/to/best.pt --device cuda
```

Không gộp trực tiếp summary có protocol/source khác nhau. Khi nhận run gốc, đối chiếu cấu hình nhóm theo model, tỷ lệ và cặp seed, xác nhận số tham số, môi trường và source. Giữ run lỗi để truy vết. Sau khi thay bằng bằng chứng đúng, chạy lại script để cập nhật trạng thái.

Môi trường kiểm tra tuần 4 hiện chỉ có Python runtime tài liệu. Đã cài thư viện audit/vẽ/pytest vào tmp của repo để sinh lại bảng. PyTorch/CUDA và checkpoint chính còn thiếu nên chưa chạy demo suy luận, test cuối hoặc kiểm tra tái lập toàn bộ từ môi trường train sạch.

Ba kiểm thử mới cho accuracy đếm ảnh, thời gian và SD đều đạt. Đã chạy kiểm tra trước test, xác nhận cả 18 run còn thiếu checkpoint/config local và sáu log ViT cần xử lý. Đã sửa nhánh collect của `scripts/run_lanh_week3.py` để báo thiếu checkpoint thay vì in “Verified” chỉ sau khi đọc CSV. Gói `results/hien_week4/hien_week4_internal.zip` có report, slide, bảng kiểm tra và source tái lập kèm SHA-256; chưa gồm các trọng số thực nghiệm.

## Lịch trình slide 11 phút

| Slide | Nội dung | Người | Giây |
|---|---|---|---:|
| 1 | Bài toán | Lan Anh | 40 |
| 2 | Dữ liệu/split/seed | Hiền | 70 |
| 3 | ViT và ablation | Lan Anh | 90 |
| 4 | ResNet/tham số | Minh | 90 |
| 5 | Recipe/checkpoint | Minh | 60 |
| 6 | Bảng và data scaling | Hiền | 60 |
| 7 | Kiểm tra tính hợp lệ | Hiền | 60 |
| 8 | Attention | Hiền | 40 |
| 9 | Tái lập/test/demo | Minh | 60 |
| 10 | Giới hạn/phần còn lại | Lan Anh | 90 |

Tổng Lan Anh 220 giây, Minh 210 giây, Hiền 230 giây. Chưa xác nhận diễn tập thực tế.

## Các bước còn chờ

1. Lan Anh đối chiếu lại sáu ViT và ba ablation với checkpoint/log gốc, giải thích các lỗi accuracy/time và bàn giao dữ liệu sửa đúng.
2. Minh chuyển gói checkpoint/config cho 18 baseline và ablation, kèm checksum và phiên bản môi trường. Swin 100 epoch chờ kết quả nếu nhóm tiếp tục phần mở rộng.
3. Chạy ghép lỗi/attention đa lớp bằng công cụ tuần 3, rồi test cuối sau khi nhóm khóa lựa chọn. Cập nhật report và slide từ kết quả đã kiểm tra.
4. Điền họ tên/MSSV, mã nhóm, link checkpoint/log và tỷ lệ đóng góp do cả nhóm xác nhận. Khi có mã nhóm đổi tên PDF theo NhomXX_DeTai03.pdf.
5. Đọc chéo, diễn tập 10–12 phút và vấn đáp. Ghi biên bản tái lập khi thực sự chạy từ môi trường sạch.
