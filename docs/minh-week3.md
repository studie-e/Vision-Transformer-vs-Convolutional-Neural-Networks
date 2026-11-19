# Minh — thực thi tuần 3

**Người phụ trách: Minh.**

Nhánh `Minh` đã đồng bộ từ `main` ở `8578f44` trước khi chạy. Tuần 3 tiếp tục giao thức nhóm với ResNet 272.474 tham số, batch 128, 100 epoch, SGD lr 0,05, momentum 0,9, weight decay 0,0005 và cosine annealing. Hai cặp `(split_seed, train_seed)` mới là `(2027, 43)` và `(2028, 44)`. Mỗi cặp có ba lượt ở 4.500, 22.500 và 45.000 ảnh train. Checkpoint chọn theo top-1 trên validation của manifest tương ứng, hòa điểm giữ epoch sớm hơn.

Sáu lượt ResNet đã hoàn tất. Swin đã chạy thử một epoch với 4.500 ảnh train trên RTX 5080, ghi đầy đủ checkpoint và log. [Kết quả và gói bàn giao](../results/week3/README.md) · [Phần Minh cho milestone](../reports/milestone_minh_week3.md).

## Tái lập và bàn giao

Chạy từ gốc repo với dữ liệu CIFAR-10 và môi trường CUDA của tuần 2.

```powershell
python -X utf8 -m src.data --verify --verify-labels
python -X utf8 -m scripts.run_minh_week3 --stage train --model resnet
python -X utf8 -m scripts.run_minh_week3 --stage collect --model resnet
```

Lệnh train chạy tuần tự trên GPU. Khi chạy lại, trainer đọc `last.pt` để tiếp tục lượt dở và kiểm tra các lượt đã hoàn tất. Các lượt ResNet nằm ở `runs/week3/resnet_{10,50,100}_seed{43,44}` và stdout nằm ở `runs/week3/logs/`. Lệnh collect đánh giá độc lập trên GPU, kiểm tra 100 epoch và đóng gói từng run kèm SHA-256 tại `runs/handoff/week3/resnet/`. Bảng kết quả và log gọn nằm ở `results/week3/`.

CSV từng run lưu nguyên commit, hash nguồn và hash manifest. Bảng tổng hợp gộp seed 42 với seed 43 và 44 sau khi đối chiếu recipe và cây cú pháp Python của mã huấn luyện. Hash thư mục `src` có thể đổi khi thêm chú thích hoặc cập nhật công cụ vẽ hình. Bước đối chiếu vẫn phát hiện thay đổi mã thực thi và phiên bản thư viện. Độ lệch chuẩn dùng `ddof=1` và phản ánh cả việc đổi tập train/validation lẫn seed huấn luyện. Mỗi kết quả tính trên 5.000 ảnh validation của split tương ứng.

## Swin cho ảnh 32×32

Hai YAML thăm dò được chuẩn bị từ prototype 261.824 tham số. Swin dùng patch 2×2, ba stage với độ rộng 26/52/104, hai block mỗi stage, 2/4/8 head và cửa sổ 4×4. Patch merging giảm lưới 16×16 xuống 8×8 rồi 4×4. Mô hình dùng `SwinTransformer` từ torchvision 0.26, khởi tạo từ đầu. [Bài báo Swin](https://arxiv.org/abs/2103.14030) · [Mã chính thức của torchvision](https://docs.pytorch.org/vision/0.26/_modules/torchvision/models/swin_transformer.html).

Tôi chọn cùng AdamW lr 0,0003, weight decay 0,05 và cosine 100 epoch như ViT làm recipe thăm dò ban đầu. Hai cấu hình dùng 10% và 100% dữ liệu với cặp `(2026, 42)`. Swin dùng stochastic depth 0,1 và có kiến trúc riêng, vì vậy đây là phần mở rộng để Hiền mô tả cùng các khác biệt về thiết lập. Hàng đợi dành ưu tiên cho các lượt ViT và ResNet chính.

```powershell
python -X utf8 -m scripts.run_minh_week3 --stage train --model swin
python -X utf8 -m scripts.run_minh_week3 --stage collect --model swin
```

## Việc Minh cần nắm khi đọc chéo

- Hai cặp seed bổ sung thay cả nguồn ảnh train/validation lẫn khởi tạo. Trong từng cặp, ViT và ResNet phải đọc cùng manifest.
- Tỷ lệ dữ liệu thay đổi số cập nhật ở cùng 100 epoch. Mỗi run có lần lượt 3.600, 17.600 hoặc 35.200 bước ở ba tỷ lệ.
- ResNet học đặc trưng qua tích chập cục bộ và chia sẻ trọng số. Swin dùng attention theo cửa sổ và dịch cửa sổ để kết nối vùng lân cận qua các block.
- `best.pt` là trạng thái dùng để đánh giá. `last.pt` chứa trạng thái để nối run và chi phí tới hết epoch 100.
