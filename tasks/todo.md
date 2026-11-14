# Checklist phân công — Lan Anh, Hiền, Minh

Kế hoạch: [plan.md](plan.md). Trạng thái từng mục được cập nhật theo bằng chứng; phần tuần 1 của Minh xem [bàn giao](../docs/minh-week1.md). Tuần là tuần môn học; hoàn thiện nội bộ ở tuần 4, nộp chính thức ở tuần 6. Mỗi người cập nhật trạng thái và dẫn chứng của phần mình.

**Giữ bảng gốc, chỉ đổi chỗ Minh và Hiền:** Lan Anh cài/train ViT và ablation; Minh cài/train ResNet/Swin, phụ trách trainer/môi trường/lịch GPU; Hiền phụ trách dữ liệu, đánh giá, attention, phân tích và ghép báo cáo.

Lộ trình từng ngày: [Lan Anh](roadmaps/lan-anh.md), [Minh](roadmaps/minh.md), [Hiền](roadmaps/hien.md); [mốc phối hợp và hàng đợi GPU](roadmaps/README.md). Trạng thái nghiệm thu vẫn cập nhật ở checklist này.

Quy mô S/M chỉ phản ánh phạm vi sửa file dự kiến; thời gian GPU được đo bằng pilot, không tính bằng số file. Các đường dẫn chưa tồn tại là vị trí đề xuất.

## Tuần 1 — dữ liệu, hai mô hình chạy thử và proposal

### T01. Môi trường, repo và log — Minh; cả nhóm phối hợp

- [x] Có môi trường chạy PyTorch/CUDA thực tế trên RTX 5080; ghi phiên bản và lệnh cài trong README và báo cáo pilot.
- [ ] Đã có repo và log TensorBoard cục bộ, quy ước config/seed/commit; còn xác nhận nơi chia sẻ/sao lưu log. Không cần mời GitHub theo giảng viên (Minh xác nhận 12/11/2026).
- [ ] Dữ liệu/checkpoint đã được loại khỏi Git; còn xác nhận và thực hiện sao lưu sang nơi nhóm thống nhất.
- **Kiểm tra:** GPU và log cục bộ đã chạy; kiểm tra đường dẫn/quyền truy cập sau khi nhóm chọn nơi chia sẻ log, không yêu cầu lời mời GitHub.
- **Phụ thuộc:** không. **File dự kiến:** `README.md`, `requirements.txt`, `.gitignore`. **Quy mô:** M.

### T02. Đọc 3 bài chính — cả ba

- [x] Lan Anh đọc ViT (ghi chú tại `docs/lanh-week1-notes.md`); Minh đọc ResNet; Hiền đọc Swin (`docs/hien-week1.md`).
- [x] Cả ba trình bày lại ý chính, thống nhất mục tiêu so sánh và phần nào tự cài.
- **Kiểm tra:** hai người còn lại hỏi về kiến trúc và cách áp dụng cho dữ liệu nhỏ.
- **Phụ thuộc:** không. **File dự kiến:** `docs/literature.md`, `reports/references.bib`. **Quy mô:** S.

### T03. Dữ liệu và tập con — Hiền

- [x] Có train 45.000 / val 5.000 / test 10.000; split phân tầng, không giao nhau.
- [x] Có tập con train 4.500 / 22.500 / 45.000 theo từng seed; lồng nhau, cung cấp cùng manifest cho hai mô hình. ResNet đã có đường tích hợp; việc chạy ViT thực tế chờ Lan Anh.
- [x] Lưu chỉ số, seed, preprocessing; augmentation chỉ áp dụng train.
- **Kiểm tra:** Chỉ số, nhãn thật, loader và tái lập theo `docs/hien-week1-handoff.md`; bằng chứng dữ liệu thật ghi riêng tại `results/hien_week1_checks.json` sau khi chạy pilot.
- **Phụ thuộc:** T01. **File dự kiến:** `src/data.py`, `configs/data.yaml`, `results/splits/` chứa các manifest nhỏ. **Quy mô:** M.

### Checkpoint A

- [ ] Cả nhóm kiểm tra môi trường, dữ liệu và mục tiêu trước khi mở loạt train.
- [ ] Có kế hoạch chia sẻ repo/log với giảng viên trong tuần 1.

### T04. ViT tự cài bản đầu — Lan Anh

- [x] Tự cài patch embedding, positional embedding, self-attention, encoder và classifier; không dùng một mô hình ViT hoàn chỉnh có sẵn thay phần tự cài. (14/11/2026, `src/models/vit.py`, bằng chứng: `scripts/check_lanh_week1.py`)
- [x] Forward trả logits đúng số lớp; backward có gradient; có cách lấy attention phục vụ trực quan hóa. (logits [B,10] ✅, 56 tham số có gradient ✅, `get_attention_weights()` trả [B,4,65,65] ✅)
- [x] Ghi số tham số trainable và phân biệt thành phần tự viết/thành phần PyTorch. (ViT 274.282 vs ResNet 272.474, chênh 0.7%. Tự cài toàn bộ kiến trúc, dùng nn.Linear/LayerNorm/Conv2d/GELU/Dropout làm building block)
- **Kiểm tra:** kiểm tra kích thước tensor, Lan Anh chạy overfit một batch nhỏ; Minh đọc chéo.
- **Phụ thuộc:** T01, T02; chạy batch dữ liệu dùng T03. **File dự kiến:** `src/models/vit.py`, `configs/vit.yaml`. **Quy mô:** S.

### T05. ResNet tương đương tham số — Minh

- [x] Có ResNet cho ảnh nhỏ: CIFAR-style, width 16, ba block/stage, projection shortcut; nguồn và khác biệt ghi ở docs/minh-week1.md.
- [x] Đã đếm ResNet 272.474 tham số và đối chiếu ViT Lan Anh (274.282 tham số); chênh lệch 0,66% ≤ 10%.
- [x] ResNet khởi tạo từ đầu, trả logits [B,10]; quy ước tích hợp ViT đã hoàn tất và kiểm chứng.
- **Bằng chứng Minh:** [pilot GPU và overfit 32 ảnh thật](../results/pilot.md), [mã ResNet](../src/models/resnet.py), [công cụ khớp tham số](../src/inspect_model.py). Đã đối chiếu ViT Lan Anh tại `results/lanh_week1_checks.json`.
- **Kiểm tra:** forward/backward và overfit batch nhỏ; Hiền đọc chéo số tham số.
- **Phụ thuộc:** T01, T02; khớp tham số với T04. **File dự kiến:** `src/models/resnet.py`, `configs/resnet.yaml`. **Quy mô:** S.

### T06. Trainer tối thiểu và pilot — Minh chủ trì, Lan Anh phối hợp

- [x] ResNet đã qua trainer chung và có loss/accuracy/lr; Hiền đã cung cấp split chính và config tích hợp. ViT Lan Anh cũng đã chạy qua trainer chung thành công trên 4.500 ảnh CIFAR-10 thật (`runs/lanh_week1_smoke`).
- [x] Checkpoint lưu model/optimizer/scheduler/RNG/generator/epoch/step; kiểm thử resume và validation có bằng chứng trong results/pilot.md.
- [x] Đã đo thời gian/epoch ViT (16,57s trên CPU) và ResNet (trên RTX 5080 và CPU). Cả hai mô hình đều có bằng chứng pilot độc lập (`results/pilot.md`, `results/hien_pilot_*`, `results/lanh_pilot_*`).
- **Bằng chứng:** [kết quả pilot ResNet](../results/pilot.md), [kết quả pilot ViT](../results/lanh_pilot_summary.csv), [checks ViT](../results/lanh_week1_checks.json).
- **Kiểm tra:** Lan Anh chạy thử ViT, Minh chạy thử ResNet; dừng rồi resume, đối chiếu log và cấu hình; không coi pilot là bảng kết quả cuối.
- **Phụ thuộc:** T03–T05. **File dự kiến:** `src/train.py`, `src/checkpoint.py`, `configs/train.yaml`, `results/pilot.md`. **Quy mô:** M.

### Checkpoint B

- [x] Cả hai mô hình chạy thử trên dữ liệu thật; có số tham số và số đo pilot (ResNet 272.474 params, ViT 274.282 params, chênh 0,66%).
- [x] Cả hai mô hình đạt yêu cầu kiểm thử kỹ thuật tuần 1 trước khi mở loạt train tuần 2.

### T07. Metric và mẫu bảng — Hiền

- [x] Tính top-1 accuracy và xuất nhãn/phân phối xác suất khi suy luận (`src/evaluate.py`).
- [x] Mỗi lượt có model, tỷ lệ, seed, params, config, commit, checkpoint, val score, thời gian và số bước (đã chuẩn bị schema và `results/runs.csv`).
- [x] Có cách tổng hợp từng seed và mean ± std; không chọn riêng seed tốt nhất (`scripts/summarize.py`).
- **Kiểm tra:** Accuracy/mean/std có ví dụ biết đáp án; kiểm thử evaluator → metadata → CSV → summary, từ chối seed trùng, trộn recipe, config/manifest sai tại `tests/test_hien.py`. Kết quả pilot dữ liệu thật lưu riêng `results/hien_pilot_*`.
- **Phụ thuộc:** T03; tích hợp với T06. **File dự kiến:** `src/evaluate.py`, `scripts/summarize.py`, `results/runs.csv`. **Quy mô:** M.

### T08. Proposal ACL — Hiền ghép; Lan Anh, Minh cùng viết

Phần Minh đã chuẩn bị tại [proposal_minh.md](../reports/proposal_minh.md) và [BibTeX](../reports/references_minh.bib); Hiền đã chuẩn bị bản tổng hợp nội bộ tại [proposal_hien.md](../reports/proposal_hien.md), khung LaTeX tại [proposal.tex](../reports/proposal.tex), và [BibTeX](../reports/references.bib).

- [x] Proposal PDF đã biên dịch và kiểm tra trực quan: 2 trang A4, ACL final, hai cột; đủ bài toán, đầu vào/đầu ra, dữ liệu, baseline, đánh giá, phân công.
- [ ] Điền đầy đủ họ tên/MSSV trước khi nộp; bản dự thảo giữ chỗ trống theo yêu cầu người dùng.
- [x] Nêu rõ mẫu số của 10%/50%/100%, mô hình tự cài và thí nghiệm dự kiến.
- [ ] Cả ba rà soát và nộp cuối tuần 1; lưu bản đã nộp.
- **Kiểm tra:** mở PDF, kiểm tra số trang/template/tên thành viên và nội dung bắt buộc.
- **Phụ thuộc:** T02, T03, thiết kế T04–T07; không chờ hoàn tất train chính.
- **File dự kiến:** `reports/proposal.tex`, `reports/references.bib`, `reports/proposal.pdf`. **Quy mô:** M.

### Checkpoint cuối tuần 1

- [ ] Proposal nộp đúng hạn; repo/log chia sẻ được; các đầu việc T01–T08 có sản phẩm hoặc trạng thái rõ ràng.
- [ ] Ba người giải thích được dữ liệu, hai mô hình và cách so sánh.

## Tuần 2 — đủ 6 lượt đầu và trực quan hóa thử

### T09. Khóa giao thức thí nghiệm — cả nhóm, Minh ghi lại

- [x] Đã khóa tham số, dữ liệu, cặp seed 2026/42, augmentation và recipe riêng; cả hai mô hình dùng 100 epoch chính theo `configs/week2/locked_protocol.json` (14/11/2026, Minh giao quyền chốt).
- [x] Đã chạy đủ hai recipe × 10 epoch trên 50% train cho mỗi mô hình; chọn bằng validation, không mở test. Bằng chứng: `results/week2/tuning.csv`.
- [x] Giữ ưu tiên 6 rồi 18 lượt chính; tuần 2 chỉ kiểm tra forward/backward Swin, chưa huấn luyện mở rộng. Các cấu hình ViT chính đã bàn giao cho Lan Anh.
- **Kiểm tra:** Lan Anh đối chiếu ViT; Minh đối chiếu ResNet/config train; Hiền đối chiếu dữ liệu/metric; ghi quyết định trước khi chạy.
- **Phụ thuộc:** T03–T07. **File thực tế:** `configs/week2/locked_protocol.json`, `docs/minh-week2.md`. **Quy mô:** S.

### T10. ViT: 10%/50%/100%, seed đầu — Lan Anh

- [ ] Hoàn thành 3 lượt với giao thức đã khóa, đủ log/checkpoint và kết quả validation.
- [ ] Không đổi recipe giữa các tỷ lệ sau khi xem kết quả test; ghi lại mọi lỗi/rerun.
- **Kiểm tra:** Hiền đối chiếu số ảnh, config và learning curve; test cuối sau khi toàn bộ lựa chọn phương pháp/checkpoint đã khóa.
- **Phụ thuộc:** T09. **Sản phẩm:** 3 run manifest, checkpoint ngoài Git và log. **Quy mô:** S về cấu hình; GPU theo pilot.

### T11. ResNet: 10%/50%/100%, seed đầu — Minh

- [ ] Hoàn thành 3 lượt với đúng tập con/seed ghép cặp của T10; đủ log/checkpoint.
- [ ] Ghi params, thời gian, số bước và kết quả validation; lưu lỗi/rerun nếu có.
- **Kiểm tra:** Hiền đối chiếu manifest dữ liệu và điều kiện so sánh; test cuối sau khi toàn bộ lựa chọn phương pháp/checkpoint đã khóa.
- **Phụ thuộc:** T09. **Sản phẩm:** 3 run manifest, checkpoint ngoài Git và log. **Quy mô:** S về cấu hình; GPU theo pilot.

### Checkpoint C

- [ ] Có đủ 6 lượt đầu; kiểm tra lỗi dữ liệu, train/eval và khớp tham số trước khi chạy thêm seed.
- [ ] Nếu kết quả bất thường, tìm nguyên nhân và ghi thay đổi giao thức; không âm thầm trộn hai cấu hình trong cùng bảng.

### T12. Attention map và bảng sơ bộ — Hiền

- [ ] Có bảng/biểu đồ data scaling và learning curve từ 6 lượt đầu.
- [ ] Xuất attention ViT chồng lên ảnh; ghi layer/head/cách gộp, chọn tập ảnh cố định gồm đúng và sai.
- [ ] Có khung confusion matrix/phân tích lỗi; nhận xét chỉ là sơ bộ khi chưa đủ seed.
- **Kiểm tra:** Lan Anh đọc chéo cách trích attention/kích thước; Minh đối chiếu bảng với log gốc. Dùng validation cho kết quả sơ bộ.
- **Phụ thuộc:** T07, T10, T11. **File dự kiến:** `src/visualize.py`, `results/summary.csv`, `results/figures/`. **Quy mô:** M.

### Checkpoint cuối tuần 2

- [ ] Đủ 6 kết quả sơ bộ và attention map thử; cả ba đọc được bảng và giải thích pipeline.
- [ ] Các phần dữ liệu, phương pháp và thiết lập thí nghiệm đã có bản viết đầu.

## Tuần 3 — lặp seed, phân tích và milestone

### T13. Hai seed còn lại của ViT — Lan Anh

- [ ] Thêm 6 lượt ViT, giữ giao thức T09; tổng ViT đủ 3 tỷ lệ × 3 seed.
- [ ] Bàn giao config/log/checkpoint và giải thích các lượt lỗi nếu có.
- **Kiểm tra:** Hiền đối chiếu đủ lượt với ma trận thí nghiệm và các seed ResNet.
- **Phụ thuộc:** Checkpoint C. **Sản phẩm:** 6 run manifest, checkpoint và log. **Quy mô:** S về cấu hình; chia hàng đợi thành từng lượt.

### T14. Hai seed còn lại của ResNet — Minh

- [ ] Thêm 6 lượt ResNet, giữ giao thức T09; tổng ResNet đủ 3 tỷ lệ × 3 seed.
- [ ] Bàn giao config/log/checkpoint, thời gian/số bước và các lượt lỗi.
- **Kiểm tra:** Hiền đối chiếu ghép cặp dữ liệu/seed với T13 và cấu hình ResNet.
- **Phụ thuộc:** Checkpoint C. **Sản phẩm:** 6 run manifest, checkpoint và log. **Quy mô:** S về cấu hình; chia hàng đợi thành từng lượt.

### T15. Milestone ACL — Hiền ghép, cả ba viết phần mình

- [ ] Tối đa 4 trang, có kết quả baseline, tiến độ, khó khăn và kế hoạch còn lại.
- [ ] Phân biệt kết quả đã có với mục tiêu; ghi số seed thực tế và việc chưa hoàn tất.
- [ ] Nộp cuối tuần 3, không chờ phần mở rộng nếu chưa xong.
- **Kiểm tra:** cả ba đối chiếu số liệu với log, kiểm tra PDF/template/số trang.
- **Phụ thuộc:** T10–T12; cập nhật T13–T14 theo kết quả thực tế tại hạn nộp.
- **File dự kiến:** `reports/milestone.tex`, `reports/milestone.pdf`, `reports/references.bib`. **Quy mô:** M.

### Checkpoint cuối tuần 3

- [ ] Milestone đã nộp đúng hạn; mục tiêu đủ 18 lượt chính hoặc có danh sách chính xác các lượt còn thiếu.
- [ ] Bảng có mean ± std nếu đủ seed; mỗi người giải thích được kết quả phần người khác.

## Tuần 3–4 — mở rộng sau khi ưu tiên bảng chính

### T16. Ablation positional embedding — Lan Anh

- [ ] ViT bỏ positional embedding ở 50% dữ liệu, mục tiêu 3 seed; chỉ đổi yếu tố khảo sát.
- [ ] Đối chiếu baseline ViT cùng tỷ lệ/seed đã có; ghi mọi chênh lệch cấu hình/tham số.
- [ ] Viết kết luận đúng phạm vi ablation, không coi đây là phép chứng minh toàn bộ inductive bias.
- **Kiểm tra:** Hiền đối chiếu config, bảng và nhận xét với log do Lan Anh bàn giao.
- **Phụ thuộc:** T13 và bảng chính được ưu tiên. **File dự kiến:** `configs/ablation.yaml`, `results/ablation.csv`, `reports/sections/ablation.tex`. **Quy mô:** M.

### T17. Thử Swin — Minh chủ trì, Lan Anh hỗ trợ

- [ ] Dùng Swin nhỏ cho ảnh 32×32; ghi nguồn mã, số tham số, recipe và khởi tạo từ đầu.
- [ ] Mục tiêu 10% và 100% × 3 seed; nếu phải thu hẹp thì ghi phạm vi và tính thăm dò trước khi báo cáo.
- [ ] Có bảng bổ sung và nhận xét về giới hạn so sánh khi khác số tham số/recipe.
- **Kiểm tra:** Minh chạy thử kiến trúc trên ảnh nhỏ trước khi train chính; Hiền đối chiếu dataset/metric và cách ghi kết quả.
- **Phụ thuộc:** T09, ưu tiên hoàn tất T13–T14 trước các lượt train Swin.
- **File dự kiến:** `src/models/swin.py`, `configs/swin.yaml`, `results/swin.csv`. **Quy mô:** M; GPU theo pilot.

### T18. Chốt bảng, hình và phân tích — Hiền; cả ba kiểm tra

- [ ] Sinh bảng chính từ log bằng script, có từng seed, mean ± std, params, thời gian và số bước.
- [ ] Có data scaling, learning curve, attention, confusion matrix, ví dụ lỗi và kết quả mở rộng đã hoàn tất.
- [ ] Viết Analysis/Limitations theo số liệu thật; nêu ảnh hưởng của recipe, ngân sách bước cập nhật và phạm vi dataset.
- **Kiểm tra:** Lan Anh đối chiếu ViT/attention/ablation; Minh kiểm tra ResNet/Swin/chi phí; cả ba đối chiếu bảng với log gốc.
- **Phụ thuộc:** T12–T14; bổ sung T16–T17 khi xong. **File dự kiến:** `scripts/summarize.py`, `results/summary.csv`, `reports/sections/analysis.tex`, `reports/sections/limitations.tex`. **Quy mô:** M.

### Checkpoint D

- [ ] Bảng chính đủ và có thể truy về log; mọi thí nghiệm mở rộng ghi đúng phạm vi đã làm.
- [ ] Khóa số liệu và lựa chọn phương pháp trước khi đóng gói; không tiếp tục tuning theo test.

## Tuần 4 — bản hoàn chỉnh nội bộ

### T19. Báo cáo cuối ACL — mỗi người viết, Hiền ghép

- [ ] Đủ Abstract 150–250 từ, Introduction, Related Work, Method có hình/loss/nguồn mã, Experiments, Analysis, Conclusion.
- [ ] Có Limitations, đóng góp từng người/tỷ lệ thực tế, References, khai báo AI; nội dung chính ≤8 trang, không dồn kết quả thiết yếu vào Appendix.
- [ ] Đúng tiếng Việt, final không ẩn danh, đủ họ tên/MSSV, A4/hai cột, font/lề gốc, BibTeX tác giả–năm; PDF NhomXX_DeTai03.pdf.
- **Kiểm tra:** cả ba mở PDF đọc chéo, kiểm tra trang/bảng/hình/trích dẫn; phân biệt mục miễn tính trang theo đề.
- **Phụ thuộc:** T18; dùng kết quả T16–T17 đã hoàn tất. **File dự kiến:** `reports/final.tex`, `reports/sections/`, `reports/references.bib`, `reports/NhomXX_DeTai03.pdf`. **Quy mô:** chia theo mục, mỗi người 1–3 file.

### T20. Tái lập và demo — Minh + Hiền, Lan Anh kiểm tra chéo

- [ ] README có lệnh cài, tải dữ liệu, train, evaluate, sinh bảng; môi trường có phiên bản; checkpoint lớn dùng link ngoài.
- [ ] Chạy thử từ môi trường sạch: suy luận nhãn/xác suất, xuất attention và sinh lại bảng chính từ kết quả lưu.
- [ ] Có lệnh tái huấn luyện từng cấu hình và một lượt tái lập đại diện để kiểm tra; không bắt buộc chạy lại toàn bộ 27 lượt trong tuần 4.
- **Kiểm tra:** người không viết phần tương ứng làm theo README; so sánh accuracy từ checkpoint và bảng với bản khóa, ghi giới hạn tái lập nếu có.
- **Phụ thuộc:** T18. **File dự kiến:** `README.md`, `requirements.txt`, `scripts/reproduce.py`, `src/predict.py`. **Quy mô:** M.

### T21. Slide và vấn đáp chéo — cả ba

- [ ] Slide 10–12 phút, mỗi người khoảng 3–4 phút; số liệu khớp báo cáo; demo chạy thử.
- [ ] Mỗi người giải thích được code mình, ít nhất một phần người khác, kết quả và một tình huống đổi siêu tham số/sửa code.
- [ ] Diễn tập thuyết trình và vấn đáp 5–7 phút/người; ghi câu chưa trả lời được để đọc lại.
- **Kiểm tra:** bấm giờ, hỏi ngẫu nhiên code, kiểm tra demo trên máy sẽ dùng.
- **Phụ thuộc:** T19, T20. **File dự kiến:** `slides/`, `docs/defense_questions.md`. **Quy mô:** M.

### Checkpoint cuối tuần 4

- [ ] Sản phẩm nội bộ hoàn chỉnh: PDF, repo, môi trường, script tái lập, log, link checkpoint, slide và demo.
- [ ] Cả ba đã đọc chéo và xác nhận số liệu/đóng góp; lưu danh sách lỗi còn lại nếu có.

## Tuần 5–6 — dự phòng và nộp theo lịch chính thức

- [ ] Tuần 5: rà soát bản nháp; kiểm tra tái lập/nguồn/AI và sửa lỗi thực tế; lưu quá trình thật.
- [ ] Tuần 6: diễn tập lần cuối; kiểm tra quyền repo/log/link tải; nộp PDF, code và slide theo hạn môn học.
- [ ] Chốt tỷ lệ đóng góp theo công việc thực tế và lưu bản đã nộp.
