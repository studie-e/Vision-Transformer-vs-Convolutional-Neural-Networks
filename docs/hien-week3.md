# Bàn giao tuần 3 của Hiền

## Sản phẩm hiện có

- Script `python -X utf8 -m scripts.analyze_hien_week3` kiểm tra sáu bản ghi validation với log 100 epoch, manifest local, số ảnh, số bước, best epoch và thời gian train. Kết quả ở `results/hien_week3/`.
- `per_seed.csv` giữ từng seed cùng accuracy, params, time/steps và checkpoint nguồn. `summary.csv` tái dùng cách tính mean/std mẫu của tuần 1. Với một seed, std là NaN.
- `run_matrix.csv` phản ánh 6/18 bản ghi đã kiểm tra, thiếu 12 lượt seed bổ sung. Máy hiện tại có 0/6 checkpoint và 0/6 config run thực tế. Việc kiểm tra recipe của các lượt trên máy Minh được ghi trong bằng chứng tuần 2.
- `attention_selection.csv` khóa 20 ảnh mỗi split seed, lấy hai chỉ số CIFAR-10 nhỏ nhất mỗi lớp trong validation. Các vị trí chỉ số được lưu riêng để kiểm tra thứ tự. Quy tắc này chọn ảnh trước khi xem kết quả dự đoán.
- `reports/milestone.tex` ghép dữ liệu/đánh giá của Hiền, kiến trúc của Lan Anh và thiết lập/kết quả Minh bàn giao. Phần tên/MSSV để trống theo yêu cầu trước đó. Bản nộp và đọc chéo nhóm chờ xác nhận thực tế.
- [Milestone PDF](../reports/milestone.pdf) đã biên dịch với template ACL final, 2 trang A4 và kiểm tra trực quan từng trang. Bộ kiểm thử hiện tại đạt 42 ca. Các nhánh chạy ghép lỗi/attention trên checkpoint P0 cần được chạy thực tế khi có gói run trên máy Hiền.

## Phân tích từ sáu bản ghi

| Train | ViT | ResNet | Khoảng cách ResNet − ViT |
|---|---:|---:|---:|
| 10% | 52,16% | 74,24% | 22,08 điểm % |
| 50% | 67,66% | 88,40% | 20,74 điểm % |
| 100% | 75,90% | 91,80% | 15,90 điểm % |

Accuracy tăng ở cả hai mô hình và khoảng cách thu hẹp. Cùng số epoch khi tăng dữ liệu làm tăng số bước cập nhật. Hai optimizer và regularization cũng khác nhau. Nhận xét này có phạm vi một cặp seed và recipe đã khóa. Độ biến thiên seed chưa thể ước lượng từ các lượt hiện có.

Ở best epoch, train accuracy ResNet khoảng 99,2–99,3%, cao hơn validation 7,5–24,9 điểm phần trăm, gợi ý cần theo dõi khả năng overfitting, đặc biệt ở 10%. ViT có train accuracy 55,62%, 68,85%, 75,43% tại best epoch. Các số train được đo trong quá trình cập nhật với augmentation/dropout, còn validation đo ở eval, nên độ chênh này cần được đọc kèm learning curve. Chưa dùng độ chênh train/val để kết luận trực tiếp về nguyên nhân.

## Ghép lỗi và attention khi nhận checkpoint

Truyền mỗi cặp ViT/ResNet bằng hai đối số `--run`. Công cụ đánh giá validation bằng thiết bị ghi trong metadata, kiểm tra checkpoint khớp log, nhãn thật và thứ tự manifest trước khi ghép ảnh.

```powershell
python -X utf8 -m scripts.analyze_hien_week3 --run runs/week2_vit_10_seed42 --run runs/week2/resnet_d10_s42
```

Các file sinh thêm gồm mọi dự đoán ghép cặp theo chỉ số gốc, số lượng bốn nhóm đúng/sai, confusion matrix CSV, hình hai ảnh đầu theo chỉ số của mỗi nhóm, bộ attention cố định đủ mười lớp và provenance chứa hash checkpoint, layer/head/grid, chỉ số ảnh.

Attention dùng CLS query tới 64 patch ở block cuối, trung bình bốn head, xếp theo hàng thành lưới 8×8 rồi lặp mỗi ô thành patch 4×4. Overlay magma alpha 0,48 trên RGB gốc, màu chuẩn hóa tương đối theo ảnh. Kiểm tra nhãn dự đoán ở từng ảnh so với evaluator. Nhóm cần đối chiếu chú thích trước khi đưa vào báo cáo.

Để nhận seed mới, nối các bản ghi cùng schema vào CSV đầu vào, lưu config/metrics cạnh checkpoint và truyền `--csv`. Các cặp dự kiến là (2026,42), (2027,43), (2028,44). Script từ chối trùng cặp, sai manifest, sai số bước hoặc log không khớp. Summary còn tách protocol_id, giúp phát hiện nhóm recipe/source khác nhau để đọc chéo trước khi ghép bảng chính.

## Phần còn chờ và kế hoạch tuần 4

1. Minh chuyển sáu ZIP P0 theo checksum trong manifest tuần 2. Sau giải nén, chạy lại bằng CUDA để khớp provenance đã công bố.
2. Lan Anh và Minh bàn giao 12 lượt seed bổ sung cùng giao thức. Cập nhật bảng sau mỗi đợt và kiểm tra các nhóm protocol khác nhau nếu có.
3. Xuất ghép lỗi và attention đa lớp từ checkpoint. Thống kê lớp nhầm lẫn cần nhãn dự đoán từng ảnh. Các PNG confusion matrix tuần 2 chỉ dùng để xem, số liệu sẽ được sinh từ prediction gốc.
4. Cả nhóm đọc chéo milestone, điền thông tin thành viên và nộp theo hạn môn. Lưu bản đã nộp sau khi có xác nhận.
5. Tuần 4 tiếp nhận ablation/Swin, khóa checkpoint, đánh giá test cuối, hoàn thiện Analysis/Limitations và slide.
