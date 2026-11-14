# Kế hoạch 4 tuần — Đề tài 3: So sánh Vision Transformer và CNN

## Phạm vi đã chốt với nhóm

- Thành viên: **Lan Anh, Hiền, Minh**; cả ba có năng lực tương đương.
- Phần cứng: **RTX 5080 trên máy cá nhân**, dùng chung một hàng đợi thí nghiệm.
- Phân công theo bảng gốc, chỉ đổi chỗ Minh và Hiền: **Lan Anh cài và train ViT; Minh phụ trách CNN, trainer và môi trường; Hiền phụ trách dữ liệu, đánh giá và phân tích**.
- Nhóm chủ động hoàn thiện sản phẩm trong **4 tuần đầu của lịch môn học 6 tuần**. Không thay đổi hạn nộp chính thức.
- Mục tiêu đề xuất: làm đủ yêu cầu tối thiểu, có attention map, ablation, phân tích inductive bias và thử thêm Swin để hướng tới mức giỏi. Đây là mục tiêu thực hiện, không phải cam kết điểm số.
- Chưa có ngày bắt đầu, lịch sử dụng máy và số giờ làm mỗi người; kế hoạch dùng tuần tương đối. Chốt lịch GPU trong tuần 1 sau khi đo thời gian chạy thực tế.
- Cập nhật 12/11/2026: đã triển khai phần tuần 1 của Minh; xem [bàn giao](../docs/minh-week1.md) và [pilot](../results/pilot.md). ViT và split chính còn chờ Lan Anh/Hiền. Theo xác nhận của Minh, giảng viên không yêu cầu gửi lời mời GitHub; các dòng về lời mời trong tài liệu gốc được điều chỉnh theo chỉ dẫn này.
- Cập nhật 14/11/2026: đã nhận ViT/split chính từ main và hoàn tất phần kỹ thuật tuần 2 của Minh. Giao thức thực tế khóa tại `configs/week2/locked_protocol.json`: cặp seed 2026/42, 100 epoch; ResNet SGD và ViT AdamW được chọn sau tuning ngang ngân sách. [Kết quả ResNet tuần 2](../results/week2/minh_week2.md); chưa chạy ViT chính thay Lan Anh hoặc đánh giá test cuối.

## 1. Yêu cầu rút ra từ tài liệu gốc

Nguồn: [Đề tài bài tập lớn môn Học sâu.docx](../Đề%20tài%20bài%20tập%20lớn%20môn%20Học%20sâu.docx). Đã đọc toàn bộ nội dung và cả 5 bảng; các yêu cầu chung phía sau danh sách đề tài cũng áp dụng cho Đề tài 3.

| Nhóm yêu cầu | Nội dung cần thực hiện | Người giữ đầu việc |
|---|---|---|
| Bài toán | So sánh ViT và CNN khi thay đổi lượng dữ liệu huấn luyện; bàn về vai trò inductive bias | Cả nhóm |
| Đầu vào/đầu ra | Ảnh RGB 32×32 hoặc 64×64; đầu ra là nhãn và phân phối xác suất; kết quả nghiên cứu gồm bảng so sánh và attention map | Lan Anh, Hiền, Minh theo phần |
| Dữ liệu | Chọn trong CIFAR-10/100, Tiny-ImageNet; chỉ số chính là Accuracy | Hiền |
| Tối thiểu | **Tự cài ViT nhỏ bằng PyTorch**; so sánh với **ResNet có số tham số tương đương** ở **10%, 50%, 100%** dữ liệu huấn luyện | Lan Anh: cài/train ViT; Minh: ResNet/train; Hiền: kiểm tra phép so sánh |
| Mở rộng trong đề | Attention map; thử DeiT hoặc Swin; phân tích inductive bias | Hiền: attention/phân tích; Minh: Swin; Lan Anh: ablation ViT |
| Proposal | Cuối tuần 1, tối đa 2 trang: bài toán, đầu vào/đầu ra, dữ liệu, baseline dự kiến, đánh giá, phân công | Hiền tổng hợp; cả ba viết phần mình |
| Milestone | Cuối tuần 3, tối đa 4 trang: kết quả baseline, tiến độ, khó khăn, kế hoạch còn lại | Hiền tổng hợp; cả ba cung cấp kết quả |
| Bản cuối | Tuần 6: báo cáo, mã nguồn, slide, thuyết trình; nhóm đặt hạn hoàn thiện nội bộ cuối tuần 4 | Cả nhóm |
| Theo dõi quá trình | Repo GitHub, log TensorBoard hoặc W&B; không cần mời giảng viên theo chỉ dẫn mới được Minh xác nhận 12/11/2026; commit và log đều đặn trong lịch 6 tuần | Minh điều phối repo/log; mỗi người lưu phần mình |
| Tái lập | README hướng dẫn cài đặt, tải dữ liệu, train, đánh giá; requirements.txt hoặc environment.yml; script/notebook tái tạo bảng chính | Minh: môi trường/trainer; Hiền: dữ liệu/bảng; Lan Anh: ViT |
| Sản phẩm kèm theo | Không đưa dữ liệu/trọng số lớn vào Git; dùng link ngoài; slide 10–12 phút, demo nếu có; PDF tên NhomXX_DeTai03.pdf | Cả nhóm |
| Liêm chính | Khai báo dùng AI; trích dẫn mã nguồn mở/pretrained; ghi rõ phần tự cài. Code có thể được kiểm tra MOSS/JPlag | Mỗi người ghi nguồn ngay khi dùng |

### Cách tính điểm ảnh hưởng đến ưu tiên công việc

- Kỹ thuật và thí nghiệm **25%**; phân tích và báo cáo **25%**; quá trình **10%**; thuyết trình và demo **10%**; vấn đáp cá nhân **30%**.
- Đề **không chấm theo accuracy cao/thấp**. Kết quả không vượt baseline vẫn có thể đạt điểm tốt nếu thí nghiệm đúng và giải thích được.
- Mỗi người vấn đáp khoảng **5–7 phút**, có ít nhất một câu về phần người khác và có thể phải giải thích đoạn code được chọn ngẫu nhiên.
- Rubric vấn đáp: hiểu code **3 điểm**, hiểu phương pháp **3**, giải thích kết quả **2**, xử lý tình huống **2**.
- **Vấn đáp dưới 4/10 thì tổng điểm bài tập lớn không vượt 5/10.** Vì vậy, cả ba phải đọc và giải thích được pipeline chung.
- Phần vượt giới hạn trang không được chấm; nộp muộn bị trừ theo quy định của môn.

## 2. Phạm vi thí nghiệm đề xuất

### Dữ liệu và câu hỏi nghiên cứu

Chọn **CIFAR-10, ảnh gốc 32×32** cho thí nghiệm chính. Tập này có 50.000 ảnh train và 10.000 ảnh test, phù hợp đầu vào đề yêu cầu. Không mở thêm dataset trước khi hoàn tất bảng chính. [Nguồn dữ liệu chính thức](https://cave.cs.toronto.edu/kriz/cifar.html).

Câu hỏi chính: khi tăng lượng dữ liệu huấn luyện, chênh lệch accuracy giữa ViT tự cài và ResNet có số tham số tương đương thay đổi thế nào? Nhận xét nào phù hợp với inductive bias của hai kiến trúc, và giới hạn nào khiến nhóm chưa thể kết luận rộng hơn?

### Quy ước so sánh phải chốt trước khi chạy chính

| Hạng mục | Quy ước đề xuất của nhóm |
|---|---|
| Chia dữ liệu | Tách cố định, phân tầng 45.000 train / 5.000 validation từ tập train gốc; giữ 10.000 test chính thức |
| Tỷ lệ | 10%, 50%, 100% tính trên **45.000 ảnh train sau khi tách validation**: 4.500 / 22.500 / 45.000 ảnh. Ghi rõ mẫu số trong proposal, báo cáo và bảng kết quả |
| Lấy tập con | Theo từng seed, lấy tập con phân tầng và lồng nhau: 10% nằm trong 50%, 50% nằm trong 100%. Hai mô hình dùng đúng cùng danh sách ảnh |
| Seed | Đề xuất 3 seed ghép cặp cho hai mô hình; lưu riêng seed chia dữ liệu/lấy tập con và seed khởi tạo/huấn luyện |
| Khởi tạo | ViT và ResNet chính đều huấn luyện từ đầu; không dùng pretrained cho một phía trong bảng chính |
| Kích thước mô hình | Đếm số tham số trainable thực tế; mục tiêu nội bộ chênh lệch không quá **10%**. Đây là ngưỡng nhóm đề xuất, tài liệu không ấn định con số |
| Cách khớp mô hình | Giữ ViT nhỏ; chọn/điều chỉnh độ rộng ResNet rồi đếm lại. Ghi đầy đủ biến thể ResNet, không chỉ ghi tên chung |
| Tiền xử lý | Cùng kích thước ảnh, chuẩn hóa và augmentation cơ bản cho hai mô hình; augmentation chỉ áp dụng trên train |
| Tuning | Dùng validation, cùng ngân sách thử cấu hình cho hai mô hình; ghi cả cấu hình thử không được chọn. Khóa recipe trước loạt chạy chính |
| Ngân sách train | Cùng số epoch tối đa cho hai mô hình ở mọi tỷ lệ; chốt số epoch sau pilot. Cho phép recipe riêng theo kiến trúc nhưng phải khai báo rõ và giữ cố định theo tỷ lệ |
| Chọn checkpoint | Dựa trên validation với cùng quy tắc; không dùng test để chọn cấu hình hoặc chọn seed đẹp nhất |
| Đánh giá theo giai đoạn | Tuần 2–3 dùng validation cho kết quả/attention sơ bộ; test cuối sau khi khóa toàn bộ phương pháp và checkpoint, dự kiến ngày 24 trong lộ trình chi tiết |
| Báo cáo chính | Accuracy test trung bình ± độ lệch chuẩn của 3 seed; kèm kết quả từng seed, số tham số, thời gian train và số bước cập nhật |
| Giới hạn cần ghi | Số tham số tương đương không đồng nghĩa chi phí tính toán bằng nhau; cùng epoch khi đổi lượng dữ liệu sẽ đổi số bước cập nhật; kết quả còn phụ thuộc recipe |

### Các lượt chạy và thứ tự ưu tiên

| Mức | Thí nghiệm | Số lượt dự kiến | Phụ trách |
|---|---|---:|---|
| P0 — bắt buộc theo đề | ViT và ResNet × 10%/50%/100% × seed đầu tiên | 6 | Lan Anh 3 ViT; Minh 3 ResNet |
| P1 — độ tin cậy đề xuất | Chạy thêm 2 seed cho cả 6 cấu hình | 12 | Lan Anh 6 ViT; Minh 6 ResNet |
| P2 — ablation | ViT bỏ positional embedding ở 50% dữ liệu × 3 seed; đối chiếu ViT chuẩn đã có | 3 | Lan Anh; Hiền tổng hợp |
| P2 — mở rộng | Swin nhỏ, train từ đầu, ở 10% và 100% × 3 seed | 6 | Minh; Lan Anh hỗ trợ |
| P2 — trực quan hóa | Attention của ViT, confusion matrix, ví dụ đúng/sai; dùng checkpoint đã có | Không cần train mới | Hiền |

Đích dự kiến là **27 lượt train chính/thêm**, chưa gồm smoke test và pilot. Đây là số lượng thiết kế, chưa phải ước tính thời gian GPU. Đo thời gian/epoch và bộ nhớ ở tuần 1 rồi xác nhận khả năng hoàn thành. Chạy đủ bảng ViT–ResNet trước khi dùng GPU cho phần mở rộng.

Với Swin, được dùng thư viện/mã nguồn mở có trích dẫn, cấu hình cho ảnh nhỏ và ghi rõ số tham số. Cố gắng khớp ngân sách với hai mô hình chính; nếu chưa khớp thì trình bày thành thí nghiệm bổ sung và nêu giới hạn so sánh. Nguồn: [bài báo Swin](https://arxiv.org/abs/2103.14030), [mã chính thức của Microsoft](https://github.com/microsoft/Swin-Transformer).

Nếu thiếu thời gian GPU: giảm Swin xuống một tỷ lệ trước; nếu chỉ có một seed thì ghi rõ kết quả thăm dò. Không bỏ tỷ lệ 10%/50%/100%, không thay phần ViT tự cài bằng mô hình có sẵn. Mọi thay đổi ngân sách phải được cả nhóm ghi vào nhật ký trước khi thực hiện.

## 3. Phân công xuyên suốt

| Người | Phần kỹ thuật chính | Phần thí nghiệm/phân tích | Phần báo cáo và trình bày |
|---|---|---|---|
| **Lan Anh** | Tự cài patch embedding, positional embedding, self-attention, Transformer encoder, classification head của ViT; hỗ trợ tích hợp trainer | Trực tiếp train ViT và ablation positional embedding; giải thích thiết kế ViT | Method–ViT, hình kiến trúc ViT, phần ablation; trình bày khoảng 3–4 phút |
| **Minh** | ResNet có số tham số tương đương; trainer/checkpoint/resume, môi trường RTX 5080; thử Swin | Trực tiếp train ResNet/Swin; thống kê tham số, thời gian; điều phối lịch GPU dùng chung với Lan Anh | Method–CNN/Swin, thiết lập train/phần cứng, README train và demo; trình bày khoảng 3–4 phút |
| **Hiền** | Dataset, train/val/test, tập con 10%/50%/100%; script đánh giá/tổng hợp và xuất attention map | Kiểm tra dữ liệu, accuracy, biểu đồ, confusion matrix, lỗi và inductive bias | Dữ liệu/đánh giá, Analysis, ghép bản ACL/slide và kiểm tra format; trình bày khoảng 3–4 phút |

Hiền tổng hợp tài liệu, nhưng **mỗi người tự viết phần mình** và cùng sửa Abstract, Introduction, Conclusion, Limitations. Mục tiêu chia tương đối đều; tỷ lệ đóng góp cuối kỳ ghi theo công việc thực tế, không mặc định mỗi người đúng 1/3. Lan Anh chịu trách nhiệm mô hình/lượt chạy ViT và ablation; Minh chịu trách nhiệm ResNet/Swin và trainer; Hiền chịu trách nhiệm dữ liệu/đánh giá.

### Đọc chéo để chuẩn bị vấn đáp

- Lan Anh kiểm tra pipeline dữ liệu/đánh giá của Hiền.
- Minh kiểm tra ViT của Lan Anh, đặc biệt kích thước tensor và attention.
- Hiền kiểm tra ResNet/trainer của Minh, đặc biệt tham số, checkpoint và cách chọn mô hình.
- Mỗi tuần mỗi người giải thích phần mình trong 10–15 phút; hai người còn lại hỏi và thử một thay đổi nhỏ.

## 4. Lịch 4 tuần và sản phẩm nghiệm thu

| Tuần | Lan Anh — ViT | Minh — CNN và huấn luyện | Hiền — dữ liệu và phân tích |
|---|---|---|---|
| **1** | Đọc bài ViT; tự cài ViT nhỏ, chạy thử | Đọc ResNet; dựng ResNet, trainer, môi trường RTX 5080; đo thời gian chạy | Chuẩn bị dữ liệu, tập con; dựng template ACL và tổng hợp proposal |
| **2** | Chạy ViT với **10%, 50%, 100%** dữ liệu | Chạy ResNet với ba tỷ lệ tương ứng; khớp số tham số với ViT | Kiểm tra kết quả; tạo bảng, biểu đồ và attention map thử |
| **3** | Chạy thêm seed; làm ablation positional embedding | Chạy thêm seed; bắt đầu thử Swin sau các lượt chính | Tổng hợp trung bình/độ lệch chuẩn, phân tích lỗi; hoàn thiện milestone |
| **4** | Chốt ablation, phần Method–ViT và slide | Chốt Swin, README, môi trường, demo và slide | Chốt Analysis, hình/bảng; ghép báo cáo ACL và slide |

Lịch từng ngày và các mốc bàn giao: [lộ trình chung](roadmaps/README.md), [Lan Anh](roadmaps/lan-anh.md), [Minh](roadmaps/minh.md), [Hiền](roadmaps/hien.md). Các lộ trình chi tiết giữ đúng phân công trong bảng trên; cập nhật hoàn thành tập trung ở [todo.md](todo.md).

### Mốc làm việc trong mỗi tuần

- Đầu tuần: họp khoảng 30 phút để chốt đầu ra, phụ thuộc và lịch GPU.
- Giữa tuần: kiểm tra ngắn bằng sản phẩm thực tế: commit, log, bảng hoặc PDF; chuyển việc nếu một người bị quá tải.
- Cuối tuần: mỗi người bàn giao mã/cấu hình, kết quả và đoạn báo cáo của mình; người đọc chéo xác nhận.
- Minh điều phối lịch GPU dùng chung. Lan Anh thực thi các lượt ViT/ablation; Minh thực thi ResNet/Swin; Hiền nhận log/checkpoint để đánh giá và phân tích.
- Lưu checkpoint có thể resume và sao lưu log sau mỗi lượt. Không suy ra thời gian train chỉ từ tên GPU.

### Hai tuần còn lại trong lịch môn học

- **Tuần 5:** rà soát bản nháp theo lịch môn; thử tái lập từ môi trường sạch, kiểm tra nguồn trích dẫn/AI, xử lý lỗi thực tế nếu có. Lưu commit/log của các kiểm tra thật.
- **Tuần 6:** luyện thuyết trình/vấn đáp, kiểm tra quyền truy cập repo và link tải, chốt PDF/slide và nộp đúng hạn chính thức.
- Không tạo commit hoặc log giả để đủ số tuần. Sản phẩm hoàn thiện sớm vẫn cần bảo đảm giảng viên xem được quá trình làm việc.

## 5. Phụ thuộc và sản phẩm phải bàn giao

Luồng công việc: **dữ liệu và quy ước đánh giá → hai mô hình chạy được → khớp tham số và khóa cấu hình → 6 lượt đầu → 18 lượt lặp → bảng/phân tích → báo cáo và tái lập**. Phần Swin và ablation dùng GPU sau các lượt chính; đọc bài, viết báo cáo và công cụ trực quan có thể làm song song.

| Bàn giao | Người giao → người nhận | Nội dung tối thiểu |
|---|---|---|
| Dữ liệu | Hiền → Lan Anh, Minh | Danh sách chỉ số train/val/test, tập con từng seed, preprocessing, số ảnh theo lớp |
| Mô hình/trainer | Lan Anh ↔ Minh; cả hai → Hiền | ViT tích hợp với trainer; ResNet/Swin, cấu hình và số tham số; cách xuất attention để Hiền dùng |
| Lượt chạy | Lan Anh/Minh → Hiền | Model, tỷ lệ, seed, cấu hình, commit, checkpoint, log, val score, thời gian và số bước |
| Tổng hợp | Hiền → cả nhóm | Bảng từng seed và mean ± std; hình có chú thích; đường dẫn kết quả gốc |
| Báo cáo | Cả ba → bản ghép ACL | Đoạn đã viết, hình/bảng, trích dẫn, nguồn mã/AI đã dùng; không chỉ gửi gạch đầu dòng |

Checklist công việc có tiêu chí nghiệm thu và phụ thuộc nằm trong [todo.md](todo.md). Các đường dẫn mã nguồn ghi ở đó là vị trí dự kiến, chưa phải file đã triển khai.

## 6. Báo cáo ACL và bộ nộp

### Format bắt buộc

- Tất cả các mốc dùng template ACL chính thức; khuyến khích LaTeX. Nếu dùng Word vẫn phải giữ đúng template Word ACL.
- Chế độ không ẩn danh: `\usepackage[final]{acl}`; đầy đủ họ tên và mã sinh viên của cả ba.
- Tiếng Việt, A4, hai cột; giữ nguyên font/cỡ chữ/lề, không chỉnh để nhồi nội dung. LaTeX dùng hỗ trợ tiếng Việt và XeLaTeX hoặc LuaLaTeX.
- Trích dẫn tác giả–năm bằng `\citet`, `\citep`, quản lý bằng BibTeX; chỉ trích tài liệu nhóm đã đọc.
- Nguồn template: [acl-org/acl-style-files](https://github.com/acl-org/acl-style-files). Quy định độ dài và nội dung của **môn học** là căn cứ nộp bài.

### Phân bổ nội dung bản cuối

| Mục bắt buộc | Người viết đầu tiên | Nội dung cần có |
|---|---|---|
| Abstract | Cả nhóm | 150–250 từ; bài toán, phương pháp, kết quả chính có con số thực tế |
| Introduction | Cả nhóm | Động lực, đầu vào/đầu ra và đóng góp |
| Related Work | Mỗi người viết bài mình đã đọc | ViT, ResNet, Swin; trích dẫn đúng |
| Method | Lan Anh + Minh | Kiến trúc, loss, hình; phân biệt tự cài và thư viện/mã có sẵn |
| Experiments | Hiền: dữ liệu/đánh giá; Minh: thiết lập; Lan Anh: chi tiết ViT | Dữ liệu/split, siêu tham số, phần cứng, thời gian, metric, bảng baseline |
| Analysis | Hiền + Lan Anh; Minh bổ sung Swin/diễn biến train | Data scaling, ablation, attention, lỗi; giải thích gắn với kết quả |
| Conclusion | Cả nhóm | Kết quả và hướng phát triển |
| Limitations | Cả nhóm | Bắt buộc, không tính vào 8 trang; giới hạn dữ liệu, phương pháp, đánh giá |
| Đóng góp thành viên | Mỗi người ghi, cả nhóm xác nhận | Bắt buộc, không tính vào 8 trang; phần việc và tỷ lệ ước tính theo thực tế |
| References | Cả nhóm | Không tính vào 8 trang; quản lý BibTeX |
| Appendix | Người có kết quả bổ sung | Tùy chọn, không tính vào 8 trang; không đặt kết quả thiết yếu chỉ ở đây vì có thể không được đọc khi chấm |

Nội dung từ Abstract đến Conclusion gói trong tối đa 8 trang. Lưu thêm khai báo AI và nguồn mã/thư viện đã dùng trong báo cáo. Chuẩn bị PDF `NhomXX_DeTai03.pdf`, repo công khai hoặc có quyền cho giảng viên, file môi trường, lệnh tái tạo bảng, link checkpoint ngoài repo và slide 10–12 phút. Chọn demo suy luận ảnh → nhãn + xác suất + attention để kiểm tra được đầu ra; đề ghi demo “nếu có”, nhóm chủ động chuẩn bị bản gọn.

## 7. Phân tích cần hoàn thành

- Biểu đồ accuracy theo 10%/50%/100%, có độ biến thiên qua seed; bảng chính kèm số tham số.
- Learning curve train/validation để kiểm tra hội tụ và overfitting.
- Attention map ViT trên một tập ảnh cố định, có ảnh đúng/sai và nhiều lớp. Ghi rõ layer/head, cách gộp và cách chồng lên ảnh. Không coi attention là bằng chứng nhân quả về quyết định của mô hình.
- Confusion matrix và nhóm ví dụ lỗi; chọn theo quy tắc công bố trước, tránh chỉ lấy ví dụ đẹp. Không dùng các quan sát trên test để tiếp tục tuning.
- Ablation positional embedding: chỉ đổi yếu tố khảo sát, giữ các yếu tố khác; giải thích đúng phạm vi phép thử.
- Liên hệ locality/weight sharing của CNN với thiết kế attention/positional embedding của ViT từ tài liệu đã đọc. Không mặc định CNN phải thắng ở 10% hay ViT phải thắng ở 100%; viết theo số liệu thật.
- Swin là phần bổ sung về kiến trúc có attention cục bộ; nêu khác biệt về tham số/recipe nếu có. Một vài phép thử không đủ tách riêng tác động nhân quả của inductive bias.

## 8. Rủi ro và cách xử lý

| Rủi ro | Xử lý |
|---|---|
| Chưa biết tốc độ chạy/khả năng dùng máy | Pilot sớm với hai mô hình; lập hàng đợi và ngân sách trước các lượt chính; dự phòng cho lượt lỗi |
| Mô hình không tương đương tham số | Đếm trước khi train chính, điều chỉnh độ rộng rồi khóa cấu hình |
| Sai tỷ lệ hoặc rò rỉ dữ liệu | Lưu chỉ số split, kiểm tra giao nhau và số lượng; tách augmentation train/eval |
| Chạy nhiều nhưng không tái lập được | Mỗi lượt có config, seed, commit, log, checkpoint; bảng được sinh bằng script |
| Dành quá nhiều thời gian cho mở rộng | Ưu tiên 6 rồi 18 lượt chính; giảm phạm vi Swin trước; ghi trung thực thí nghiệm chưa hoàn tất |
| Một người gánh toàn bộ báo cáo | Mỗi người giao phần viết cùng sản phẩm kỹ thuật; Hiền chỉ ghép và kiểm tra format |
| Chỉ hiểu phần mình | Đọc chéo hằng tuần, diễn tập hỏi code và sửa một thay đổi nhỏ |

## 9. Ba bài đọc chính trong tuần 1

1. **Lan Anh:** [An Image is Worth 16x16 Words — ViT](https://arxiv.org/abs/2010.11929).
2. **Minh:** [Deep Residual Learning for Image Recognition — ResNet](https://arxiv.org/abs/1512.03385).
3. **Hiền:** [Swin Transformer](https://arxiv.org/abs/2103.14030).

Mỗi người ghi khoảng nửa trang: ý tưởng, thành phần kiến trúc cần dùng, liên hệ với đề tài và giới hạn khi áp dụng cho ảnh nhỏ; trình bày lại cho cả nhóm. Chỉ đưa trích dẫn vào báo cáo sau khi đã đọc nội dung liên quan.
