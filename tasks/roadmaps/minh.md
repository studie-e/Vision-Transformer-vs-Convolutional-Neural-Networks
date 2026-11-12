# Lộ trình Minh — CNN, trainer, môi trường và Swin

**Vai trò đổi từ cột Hiền trong bảng gốc:** cài/train ResNet, xây trainer/môi trường dùng chung, đo thời gian chạy, thử Swin; README, tái lập, demo và slide. Lan Anh vẫn trực tiếp train ViT/ablation. Hiền phụ trách dữ liệu, metric, attention, phân tích và ghép báo cáo.

Tham chiếu [lịch chung](README.md) và **T01, T02, T05, T06, T09, T11, T14, T17, T18–T21** trong [todo.md](../todo.md). Ngày 1–28 là ngày tương đối; cửa sổ GPU phải được kiểm tra bằng pilot. Cập nhật trạng thái tại todo; đường dẫn dưới đây là sản phẩm dự kiến.

**Cập nhật 12/11/2026:** đã triển khai mã và kiểm tra phần Minh; xem [bàn giao tuần 1](../../docs/minh-week1.md), [kết quả pilot](../../results/pilot.md). Chưa có ViT/split chính; không cần mời GitHub theo giảng viên. Trạng thái hoàn thành chỉ cập nhật theo mục tại todo.

## Tuần 1 — môi trường, ResNet và trainer

| Ngày | Việc cụ thể | Sản phẩm bàn giao |
|---|---|---|
| **1** | Kiểm tra PyTorch/CUDA thực tế trên RTX 5080; ghi phiên bản. Thiết lập repo/log/sao lưu, quyền chia sẻ trong nhóm. Đọc kiến trúc ResNet. | Môi trường chạy GPU, README ban đầu, ghi chú ResNet → cả nhóm |
| **2** | Dựng ResNet cho ảnh nhỏ, ghi biến thể/nguồn mã; thử forward/backward. Thống nhất với Lan Anh đầu vào ảnh và logits 10 lớp. | `src/models/resnet.py`, cấu hình dự kiến |
| **3** | Dựng vòng train/validation, optimizer/scheduler theo cấu hình; tích hợp dữ liệu Hiền giao. Log loss/accuracy/lr; kiểm tra train/eval. | `src/train.py`, `configs/train.yaml` bản đầu → Lan Anh |
| **4** | Bổ sung checkpoint/resume: trạng thái mô hình, optimizer, scheduler và thông tin cần để tiếp tục. Lưu config/seed/commit theo run. | `src/checkpoint.py`, log mẫu, hướng dẫn chạy/resume |
| **5** | Đối chiếu số tham số với ViT của Lan Anh; điều chỉnh ResNet trước khi train chính. Tích hợp hai mô hình vào trainer chung. | `configs/resnet.yaml`, bảng tham số, trainer dùng được → Hiền, Lan Anh |
| **6** | Trực tiếp chạy overfit batch nhỏ/pilot ResNet; Lan Anh chạy pilot ViT theo lịch. Đo thời gian/epoch/bộ nhớ; thử dừng-resume. Viết thiết lập cho proposal. | Log/checkpoint pilot, dự toán GPU, đoạn proposal → Hiền |
| **7** | Kiểm tra proposal, đường dẫn repo/log, sao lưu; không cần mời GitHub theo chỉ dẫn mới. Giải thích trainer và ResNet cho hai bạn. | Xác nhận mốc tuần 1; môi trường và trainer được bàn giao |

**Cuối tuần cần đạt:** GPU dùng được; hai mô hình dùng chung trainer; ResNet tương đương tham số mục tiêu; checkpoint/resume hoạt động; có số đo pilot, log chia sẻ và hướng dẫn chạy ViT cho Lan Anh.

## Tuần 2 — train ResNet với ba tỷ lệ

| Ngày | Việc cụ thể | Sản phẩm bàn giao |
|---|---|---|
| **8** | Tuning ResNet trên validation trong ngân sách ngang ViT; ghi mọi cấu hình thử. Tính tổng GPU dự kiến, cộng đánh giá/lưu và dự phòng. | Bảng thử cấu hình, đề xuất recipe/epoch/lịch GPU |
| **9** | Cùng nhóm khóa giao thức trước loạt chính: params, epoch, recipe từng mô hình, seed, checkpoint rule. Chia slot với Lan Anh. | Config/commit ResNet đã khóa, lịch GPU và quy tắc log |
| **10** | Train ResNet 10% seed đầu, đúng manifest ghép cặp ViT. Lưu best checkpoint theo validation và trạng thái resume. | Run `resnet_d10_s17` → Hiền |
| **11** | Train ResNet 50%, giữ recipe; ghi params, thời gian, số bước và val score. | Run `resnet_d50_s17` → Hiền |
| **12** | Train ResNet 100%, giữ recipe. Hoàn tất metadata; điều chỉnh cửa sổ chạy theo tốc độ pilot nếu cần. | Run `resnet_d100_s17` → Hiền |
| **13** | Kiểm tra đủ 3 lượt; đọc learning curve, hỗ trợ Hiền đối chiếu metric. Chuẩn bị Swin cho ảnh nhỏ, kiểm tra forward và số tham số. | Gói ResNet seed đầu; prototype/cấu hình Swin |
| **14** | Viết Method–CNN và README train/resume; đọc chéo ViT của Lan Anh. Dự phòng lượt lỗi/thiếu. | Phần viết, hình CNN, hướng dẫn train → Hiền, Lan Anh |

**Cuối tuần cần đạt:** 3 tỷ lệ ResNet seed đầu đủ log/config/checkpoint; recipe và số tham số khớp phần công bố; Lan Anh sử dụng được trainer; Swin mới chuẩn bị, chưa lấy GPU khỏi các lượt chính.

## Tuần 3 — thêm seed và bắt đầu Swin

| Ngày | Việc cụ thể | Sản phẩm bàn giao |
|---|---|---|
| **15** | Bắt đầu nhóm ResNet seed 42, ưu tiên 10%/50% theo lịch chung; dùng đúng tập con ghép cặp ViT. | Run/log và trạng thái → Hiền |
| **16** | Hoàn tất nhóm seed 42, gồm 100%; kiểm tra best epoch/config và checkpoint. | 3 lượt seed 42 hoặc danh sách thiếu rõ ràng |
| **17** | Bắt đầu nhóm ResNet seed 2026, giữ nguyên recipe và cùng manifest ViT. | Run/log seed 2026 → Hiền |
| **18** | Hoàn tất nhóm seed 2026; đối chiếu 3 tỷ lệ × 3 seed, sao lưu checkpoint. | Mục tiêu đủ 9 lượt ResNet chính |
| **19** | Khóa cấu hình Swin cho ảnh 32×32, nguồn mã, số tham số và khởi tạo từ đầu; xác nhận cách dùng dữ liệu/metric chung. | `src/models/swin.py`, `configs/swin.yaml`, phạm vi so sánh → Hiền |
| **20** | Khi cả ViT và ResNet đã ưu tiên đủ 18 lượt, chạy thử rồi bắt đầu Swin 10%/100%. Viết tiến độ, khó khăn, GPU/time cho milestone. | Các run Swin đã có, đoạn milestone → Hiền |
| **21** | Rà milestone ≤4 trang; xác nhận số lượt/seed thực tế, việc còn lại. Giải thích checkpoint/resume và kết quả ViT khi vấn đáp chéo. | Xác nhận nội dung CNN/train trước hạn nộp |

**Cuối tuần cần đạt:** mục tiêu đủ 9 lượt ResNet; Swin đã chuẩn bị/bắt đầu theo ngân sách; không hoãn milestone hoặc bỏ lượt chính để chạy mở rộng.

## Tuần 4 — chốt Swin, tái lập và demo

| Ngày | Việc cụ thể | Sản phẩm bàn giao |
|---|---|---|
| **22** | Tiếp tục Swin 10%/100% qua ba seed theo ngân sách. Ghi đầy đủ tham số, recipe, thời gian và các điểm khác bảng chính. | Run/log/checkpoint Swin → Hiền |
| **23** | Hoàn tất phạm vi Swin đã chốt; khóa checkpoint bằng validation. Nếu không đủ 6 lượt, báo đúng phạm vi/thăm dò; không tạo số liệu thay thế. | Danh sách checkpoint ResNet/Swin cuối, manifest cấu hình → Hiền |
| **24** | Hiền đánh giá test cuối; đối chiếu kết quả với checkpoint và log. Viết nhận xét CNN/Swin theo số thực tế. | Xác nhận bảng, kết luận và giới hạn so sánh |
| **25** | Hoàn thiện Method–CNN/Swin và thiết lập huấn luyện/phần cứng/thời gian. Chốt requirements/README, link checkpoint ngoài Git, khai báo nguồn mã/AI. | Phần viết và bộ tái lập → Hiền |
| **26** | Kiểm tra cài môi trường sạch, tải checkpoint, suy luận ảnh/nhãn/xác suất; phối hợp Hiền hiển thị attention. Chạy một lượt tái lập đại diện nếu ngân sách cho phép. | Demo chạy được; biên bản tái lập; slide cá nhân |
| **27** | Diễn tập 3–4 phút cá nhân trong bài nhóm 10–12 phút; vấn đáp trainer/ResNet/Swin, đổi siêu tham số và hỏi thêm ViT/dữ liệu. | Slide sửa cuối, danh sách câu cần đọc lại |
| **28** | Kiểm tra repo/log/link/README, sao lưu bản cuối. Xác nhận đóng góp và bàn giao cho cả nhóm. | Gói code/môi trường/checkpoint/log hoàn chỉnh |

## Điều kiện hoàn thành cá nhân

- ResNet, Swin và trainer có nguồn/config/params rõ ràng; môi trường chạy được trên máy của nhóm.
- Mục tiêu 9 lượt ResNet + 6 lượt Swin; mọi lượt có log/checkpoint/seed/manifest/commit/time/steps. Thiếu seed nào ghi đúng seed đó.
- Lan Anh dùng được trainer; Hiền đánh giá được checkpoint; repo/log có quyền cho giảng viên và file lớn không nằm trong Git.
- README có cách cài, train, resume, evaluate, tái lập; demo đã chạy thử; hiểu cả ViT để vấn đáp.

Nếu GPU chậm hơn dự kiến, cập nhật hàng đợi cùng Lan Anh, ưu tiên 18 lượt chính rồi mới Swin/ablation. Không âm thầm giảm epoch riêng một lượt chính. Khi chờ GPU, hoàn thiện README/Method/Swin và đọc chéo ViT. Tuần 5 kiểm tra tái lập/sửa lỗi thực tế, tuần 6 luyện vấn đáp và xác nhận bản nộp.
