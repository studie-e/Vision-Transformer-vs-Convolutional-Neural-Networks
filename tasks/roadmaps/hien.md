# Lộ trình Hiền — dữ liệu, đánh giá, attention và phân tích

**Vai trò đổi từ cột Minh trong bảng gốc:** chuẩn bị dữ liệu/tập con; đánh giá, bảng/biểu đồ, attention, lỗi và inductive bias; tổng hợp proposal, milestone, báo cáo ACL và slide. Lan Anh trực tiếp train ViT/ablation; Minh trực tiếp train ResNet/Swin và xây trainer.

Tham chiếu [lịch chung](README.md) và **T02, T03, T07–T09, T12, T15, T18–T21** trong [todo.md](../todo.md). Ngày 1–28 là ngày tương đối. Cập nhật trạng thái ở todo; đường dẫn dưới đây là sản phẩm dự kiến. Mỗi người tự viết phần mình, Hiền ghép/kiểm tra format.

## Tuần 1 — dữ liệu, template ACL và proposal

| Ngày | Việc cụ thể | Sản phẩm bàn giao |
|---|---|---|
| **1** | Đọc yêu cầu chung và bài Swin trong plan; chốt CIFAR-10 với nhóm. Dựng ACL tiếng Việt, final không ẩn danh, thông tin 3 thành viên. | Ghi chú dữ liệu/bài đọc; khung proposal và BibTeX |
| **2** | Tách 45.000 train / 5.000 val, giữ test 10.000. Tạo tập con phân tầng/lồng nhau 10/50/100% cho ba seed; lưu chỉ số. | `src/data.py`, `configs/data.yaml`, `results/splits/` |
| **3** | Kiểm tra số ảnh/lớp, giao nhau, tính lồng nhau, batch/nhãn và transform train/eval. Bàn giao dữ liệu dùng chung. Dựng accuracy trên ví dụ biết đáp án. | Gói dữ liệu/manifest → Lan Anh, Minh; metric kiểm tra được |
| **4** | Hoàn thiện script đánh giá và mẫu metadata: model, tỷ lệ, seed, params, config, commit, best val, checkpoint, time/steps. Viết dữ liệu/đánh giá cho proposal. | `src/evaluate.py`, khung `results/runs.csv`, đoạn proposal |
| **5** | Dựng script tổng hợp từng seed và mean/std; kiểm tra log mẫu với Minh. Nhận số tham số ViT/ResNet và xác nhận ngưỡng nhóm đề xuất. | `scripts/summarize.py`, bảng mẫu và danh sách điểm cần sửa |
| **6** | Nhận phần ViT từ Lan Anh, CNN/thiết lập từ Minh. Ghép proposal: bài toán, đầu vào/ra, dữ liệu, baseline, metric, phân công; kiểm tra ≤2 trang. | Proposal PDF để cả ba đọc chéo |
| **7** | Nộp proposal theo hạn môn, lưu bản đã nộp; kiểm tra repo/log giảng viên truy cập được cùng Minh. | Mốc tuần 1 hoàn tất, tài liệu/trạng thái được lưu |

**Cuối tuần cần đạt:** dữ liệu đúng số lượng, không rò rỉ; hai mô hình dùng cùng manifest; metric kiểm tra bằng ví dụ biết đáp án; proposal đúng ACL/≤2 trang và đủ nội dung.

## Tuần 2 — kiểm tra kết quả và attention thử

| Ngày | Việc cụ thể | Sản phẩm bàn giao |
|---|---|---|
| **8** | Đối chiếu metric/log pilot; kiểm tra checkpoint được chọn bằng validation, preprocessing giống nhau cho hai mô hình. | Phản hồi pilot, metric đã thống nhất → Lan Anh, Minh |
| **9** | Kiểm tra giao thức khóa: tỷ lệ/mẫu số, split/seed, params, recipe từng mô hình, số epoch. Lập ma trận 18 lượt để theo dõi. | `docs/experiment_protocol.md`, bảng run dự kiến |
| **10** | Xây cách đọc attention từ ViT với hỗ trợ Lan Anh: patch grid, lớp/head, cách gộp và chồng lên ảnh. Khi chưa có checkpoint, kiểm tra kích thước trước. | `src/visualize.py` bản đầu, ghi chú cách tính |
| **11** | Nhận các lượt đầu từ hai bạn; đối chiếu log/config/số ảnh. Sinh bảng validation từng mô hình/tỷ lệ, kiểm tra thiếu metadata. | Bảng validation cập nhật, danh sách lượt cần bổ sung |
| **12** | Tạo attention thử trên validation từ checkpoint ViT đã có; kiểm tra mapping patch/ảnh gốc với Lan Anh. Chuẩn bị learning curve và confusion matrix. | Hình attention có chú thích và run nguồn |
| **13** | Kiểm tra đủ 6 lượt seed đầu; dựng accuracy–data, learning curve, ảnh đúng/sai trên validation. Chỉ ghi nhận xét sơ bộ. | Bảng/hình sơ bộ → cả nhóm |
| **14** | Viết phần Dữ liệu/Đánh giá bản đầu; đọc chéo trainer/ResNet của Minh, đặc biệt metric, chọn checkpoint, resume. Dự phòng lỗi dữ liệu/attention. | Phần viết, phản hồi đọc chéo, checkpoint tuần 2 |

**Cuối tuần cần đạt:** 6 lượt đầu có thể truy về log/checkpoint; bảng và attention validation đúng, không chỉ lấy ví dụ đẹp. Không dùng test để tuning hoặc chọn seed.

## Tuần 3 — thống kê, phân tích lỗi và milestone

| Ngày | Việc cụ thể | Sản phẩm bàn giao |
|---|---|---|
| **15** | Nhận các seed mới; tự động cập nhật ma trận đủ/thiếu lượt. Kiểm tra cách tính độ lệch chuẩn mẫu trên dữ liệu kiểm tra biết trước. | Bảng từng seed, mean/std nhất quán |
| **16** | Chọn quy tắc tập ảnh validation cố định, nhiều lớp; xuất attention ở layer/head/cách gộp đã công bố, kiểm tra với Lan Anh. | Bộ attention phân tích, caption và quy tắc chọn |
| **17** | Phân nhóm lỗi cùng ảnh: hai mô hình cùng đúng, chỉ ViT sai, chỉ CNN sai, cùng sai. Xem lớp nhầm lẫn/overfitting và độ biến thiên seed. | Bảng ví dụ lỗi, confusion matrix, ghi chú nguyên nhân có bằng chứng |
| **18** | Nhận thêm log/checkpoint; kiểm tra hai mô hình cùng manifest/recipe đã khóa và không chọn seed đẹp nhất. | Danh sách run hợp lệ/thiếu/lỗi → hai bạn xử lý |
| **19** | Tổng hợp mục tiêu 18 lượt validation, params/time/steps. Viết Analysis sơ bộ về data scaling/inductive bias, tránh kết luận nhân quả từ vài thí nghiệm. | Bảng 3 tỷ lệ × 2 mô hình và phân tích có số liệu |
| **20** | Ghép milestone với phần hai bạn gửi: baseline, tiến độ, khó khăn, phần còn lại; ghi rõ seed/thí nghiệm thực tế; kiểm tra ≤4 trang. | Milestone PDF để cả nhóm đọc chéo |
| **21** | Nộp milestone theo hạn; lưu bản nộp, cập nhật checklist. Chốt danh sách dữ liệu/hình/phân tích còn thiếu cho tuần 4. | Mốc tuần 3 hoàn tất, danh sách công việc cuối |

**Cuối tuần cần đạt:** số liệu thống kê khớp log, phân biệt kết quả đã có và dự kiến; milestone đúng hạn/≤4 trang; attention/lỗi và giải thích có thể kiểm tra lại.

## Tuần 4 — Analysis, báo cáo ACL và slide

| Ngày | Việc cụ thể | Sản phẩm bàn giao |
|---|---|---|
| **22** | Nhận ablation từ Lan Anh và Swin từ Minh; ghép kết quả từng seed, kiểm tra baseline/cấu hình/tham số, ghi trường hợp chỉ có kết quả thăm dò. | Bảng bổ sung và danh sách thiếu metadata |
| **23** | Chốt danh sách checkpoint với hai bạn, kiểm tra script đánh giá và chọn ảnh minh họa theo quy tắc. Xác nhận không còn tuning trước test cuối. | Danh sách checkpoint/giao thức đánh giá cuối |
| **24** | Đánh giá test cuối, sinh bảng mean ± std, learning curve/attention/confusion matrix/ví dụ đầu ra. Đối chiếu độc lập số trong bảng với log và checkpoint. | Bảng/hình cuối có caption, nguồn run, script sinh lại |
| **25** | Ghép ACL: Abstract 150–250 từ, Intro, Related Work, Method, Experiments, Analysis, Conclusion; thêm Limitations, đóng góp, References, khai báo AI. | PDF nội dung ≤8 trang, đầy đủ họ tên/MSSV, tên file đúng |
| **26** | Sinh lại bảng từ môi trường sạch cùng Minh; kiểm tra demo/attention. Ghép slide nhóm 10–12 phút; mỗi người gửi phần mình khoảng 3–4 phút. | Biên bản tái lập, bộ slide và bản PDF đọc chéo |
| **27** | Diễn tập trình bày/vấn đáp; giải thích split, seed, accuracy, biến thiên, lỗi, attention và inductive bias; hỏi thêm code ViT/trainer. | Slide/PDF sửa cuối, danh sách câu cần đọc lại |
| **28** | Rà giới hạn trang, hình/bảng/trích dẫn/AI, link repo/log/checkpoint, đóng góp thực tế. Đóng gói PDF/slide và xác nhận cùng nhóm. | Bộ tài liệu hoàn chỉnh nội bộ, checklist cập nhật |

## Điều kiện hoàn thành cá nhân

- Dữ liệu/split/tập con tái lập được; hai mô hình dùng cùng danh sách ảnh và đúng mẫu số tỷ lệ.
- Script sinh bảng/hình từ kết quả lưu; attention có layer/head/cách gộp/caption; số liệu có từng seed và giới hạn được khai báo.
- Proposal ≤2 trang, milestone ≤4 trang đúng hạn; báo cáo cuối ≤8 trang nội dung theo ACL, đủ Limitations/đóng góp/AI và trích dẫn.
- Slide nhóm đủ 10–12 phút; giải thích được phần mình và ít nhất một phần code của Lan Anh/Minh.

Nếu chờ checkpoint, tiếp tục script tổng hợp, khung Analysis/Limitations, kiểm tra template và tài liệu tham khảo. Không tự tạo số liệu để lấp bảng. Mỗi người phải gửi đoạn viết hoàn chỉnh để Hiền ghép, không dồn toàn bộ việc viết cho Hiền. Tuần 5 rà soát bản nháp/tái lập, tuần 6 kiểm tra bộ nộp và luyện vấn đáp.
