# Lộ trình Lan Anh — ViT và ablation

**Vai trò giữ nguyên:** tự cài ViT, trực tiếp huấn luyện ViT và ablation; viết Method–ViT và slide. Hiền chuẩn bị dữ liệu/attention/đánh giá; Minh cung cấp trainer/môi trường và điều phối GPU.

Tham chiếu [lịch chung](README.md) và các đầu việc **T02, T04, T06, T09, T10, T13, T16, T18–T21** trong [todo.md](../todo.md). Ngày 1–28 là ngày tương đối. Cập nhật trạng thái ở todo; các đường dẫn dưới đây là sản phẩm dự kiến.

**Giao thức cập nhật 14/11/2026:** các cặp (split seed, train seed) dùng chung với ResNet là (2026,42), (2027,43), (2028,44). Các tên s17 trong bảng lịch cũ là đề xuất ban đầu; ưu tiên YAML đã khóa tại `configs/week2/vit_d{10,50,100}_s42.yaml` và [bàn giao Minh tuần 2](../../docs/minh-week2.md).

## Tuần 1 — tự cài ViT và chạy thử

| Ngày | Việc cụ thể | Sản phẩm bàn giao |
|---|---|---|
| **1** | Đọc bài ViT được dẫn trong plan; ghi patch embedding, positional embedding, attention, encoder, classifier. Thống nhất logits 10 lớp với Minh. | Ghi chú bài đọc, sơ đồ mô hình và cấu hình dự kiến → cả nhóm |
| **2** | Cài patch embedding, token phân lớp và positional embedding; kiểm tra số patch/token và kích thước tensor. | Phần đầu `src/models/vit.py`, ghi chú tensor |
| **3** | Cài self-attention; kiểm tra chia/gộp head và gradient. Nhận dữ liệu/tập con từ Hiền để thử forward. | Attention chạy với batch thật, danh sách lỗi nếu có |
| **4** | Cài encoder, normalization/MLP và classifier; ghép ViT hoàn chỉnh; tách cách lấy attention để trainer vẫn nhận logits thống nhất. | ViT forward/backward hợp lệ, cách xuất attention → Minh, Hiền |
| **5** | Đếm tham số, cùng Minh khớp ResNet; bàn giao config đầy đủ và tích hợp trainer chung. | `configs/vit.yaml`, số tham số, mã ViT có thể chạy → Minh kiểm tra |
| **6** | Trực tiếp chạy overfit batch nhỏ và pilot ViT trong lịch GPU; kiểm tra loss/val/gradient. Sửa lỗi mô hình; viết đoạn proposal về ViT. | Log/checkpoint pilot; đoạn proposal và trích dẫn → Hiền |
| **7** | Kiểm tra proposal, phần tự cài và phân công; giải thích ViT cho Minh/Hiền. Dự phòng sửa lỗi pilot. | Xác nhận proposal tuần 1, ViT bản đầu nghiệm thu |

**Cuối tuần cần đạt:** ViT tự cài chạy được với dữ liệu Hiền cung cấp; số tham số đã đối chiếu ResNet; có log pilot và cách lấy attention. Không dùng mô hình ViT hoàn chỉnh có sẵn thay phần tự cài.

## Tuần 2 — train ViT với ba tỷ lệ

| Ngày | Việc cụ thể | Sản phẩm bàn giao |
|---|---|---|
| **8** | Xem pilot cùng Minh; thử cấu hình ViT trên validation trong ngân sách tuning đã thống nhất. | Bảng cấu hình thử, val score và cấu hình chọn |
| **9** | Khóa kiến trúc/recipe/epoch/seed; kiểm tra đúng manifest ở từng tỷ lệ. Đặt lịch GPU với Minh. | Config/commit chính và lịch 3 lượt → Hiền, Minh |
| **10** | Train ViT 10% seed đầu; theo dõi loss/val, lưu checkpoint tốt nhất theo validation và trạng thái resume. | Run `vit_d10_s17` và log đầy đủ → Hiền |
| **11** | Train ViT 50% với cùng recipe; kiểm tra số ảnh/steps và xử lý lỗi nếu có. | Run `vit_d50_s17` → Hiền |
| **12** | Train ViT 100% với cùng recipe; lưu thời gian, số bước, best epoch/val. Cửa sổ chạy dịch theo pilot nếu cần. | Run `vit_d100_s17` → Hiền |
| **13** | Kiểm tra đủ 3 lượt; cung cấp checkpoint/cách lấy attention cho Hiền; đọc learning curve sơ bộ. | Bảng metadata ViT, nhận xét validation, phản hồi attention |
| **14** | Viết Method–ViT bản đầu và hình kiến trúc. Đọc chéo dữ liệu/metric của Hiền. Dự phòng lượt thiếu. | Phần viết và hình → Hiền; lỗi dữ liệu/metric nếu có |

**Cuối tuần cần đạt:** 3 tỷ lệ seed đầu có đủ log/config/checkpoint, không đổi recipe sau khi xem kết quả. Attention/bảng sơ bộ dùng validation; chưa dùng test để tuning.

## Tuần 3 — thêm seed và ablation

| Ngày | Việc cụ thể | Sản phẩm bàn giao |
|---|---|---|
| **15** | Bắt đầu 3 lượt ViT seed 42 theo lịch GPU; ưu tiên 10%/50%, giữ nguyên recipe. | Các run hoàn tất và trạng thái hàng đợi → Hiền |
| **16** | Hoàn tất nhóm seed 42, gồm 100%; kiểm tra checkpoint và manifest từng lượt. | Đủ 3 lượt seed 42 hoặc danh sách còn thiếu |
| **17** | Bắt đầu 3 lượt ViT seed 2026, cùng cấu hình và tập con ghép cặp ResNet. | Run/log seed 2026 → Hiền |
| **18** | Hoàn tất nhóm seed 2026; đối chiếu đủ 3 tỷ lệ × 3 seed. | Mục tiêu đủ 9 lượt ViT chính, metadata chuẩn |
| **19** | Chuẩn bị ViT bỏ positional embedding ở 50%; chỉ đổi yếu tố khảo sát, ghi chênh lệch tham số; khóa config ablation. | `configs/ablation.yaml`, baseline/3 seed đối chiếu → Hiền, Minh |
| **20** | Train ablation khi toàn nhóm đã ưu tiên đủ 18 lượt chính. Viết tiến độ ViT/kết quả thực tế/khó khăn cho milestone. | Log ablation đã có; đoạn milestone → Hiền |
| **21** | Rà milestone ≤4 trang; xác nhận số seed thật, phần còn thiếu; luyện giải thích ViT và kết quả CNN của Minh. | Xác nhận phần ViT trước hạn nộp |

**Cuối tuần cần đạt:** mục tiêu đủ 9 lượt ViT; ablation đã bắt đầu hoặc có config/lịch rõ ràng; milestone ghi đúng tiến độ, không hoãn để chờ mở rộng.

## Tuần 4 — chốt ViT, ablation, báo cáo và slide

| Ngày | Việc cụ thể | Sản phẩm bàn giao |
|---|---|---|
| **22** | Hoàn thành 3 lượt no-pos ở 50% trong ngân sách; đối chiếu từng seed với ViT chuẩn. | Log/checkpoint và kết luận ablation ban đầu → Hiền |
| **23** | Khóa toàn bộ checkpoint ViT/ablation bằng validation; kiểm tra config, manifest, best epoch. | Danh sách checkpoint cuối và bảng đối chiếu → Hiền |
| **24** | Hiền đánh giá test cuối; đối chiếu số liệu ViT/ablation và cách trích attention. Không sửa phương pháp theo test. | Bảng/hình đã đọc chéo, nhận xét ablation theo số thật |
| **25** | Hoàn thiện Method–ViT, hình kiến trúc và đoạn ablation; ghi phần tự cài/thư viện/AI, nguồn và đóng góp. | Phần viết hoàn chỉnh/BibTeX → Hiền ghép ACL |
| **26** | Làm slide khoảng 3–4 phút về ViT/ablation; làm theo README kiểm tra dữ liệu/đánh giá của Hiền. | Slide cá nhân, phản hồi tái lập |
| **27** | Diễn tập thuyết trình/vấn đáp: tensor, attention, positional embedding, loss, lỗi và kết quả; trả lời thêm dữ liệu/CNN. | Danh sách câu chưa rõ, sửa slide và diễn giải |
| **28** | Rà PDF/hình/link, xác nhận log và đóng góp; bàn giao bản cá nhân cuối, cập nhật mục nghiệm thu. | Phần Lan Anh hoàn chỉnh, được Minh/Hiền đọc chéo |

## Điều kiện hoàn thành cá nhân

- ViT tự cài, có config/số tham số và chạy được với trainer chung; giải thích được toàn bộ phần mình.
- Mục tiêu 9 lượt ViT chính + 3 lượt ablation, đủ metadata và checkpoint ngoài Git. Thiếu lượt nào phải ghi đúng lượt đó.
- Hiền dùng được cách lấy attention; Method/ablation có hình, số liệu và nguồn rõ ràng.
- Hiểu cách Hiền chia dữ liệu/tính metric và cách Minh chọn checkpoint/resume; đã luyện slide và vấn đáp chéo.

Nếu chờ GPU, tiếp tục viết Method/hình kiến trúc, chuẩn bị ablation và đọc chéo dữ liệu. Nếu có lỗi trainer, gửi Minh log/tensor/cấu hình để sửa; không tự đổi recipe một lượt chính để che lỗi. Tuần 5 rà soát/tái lập, tuần 6 luyện vấn đáp và xác nhận bản nộp.
