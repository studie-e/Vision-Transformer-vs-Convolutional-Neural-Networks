# Bàn giao tuần 1 — Hiền

Phạm vi: T02 (Swin), T03 (dữ liệu), T07 (metric/tổng hợp), T08 (proposal).
Thông tin họ tên đầy đủ/MSSV được để trống theo yêu cầu; không coi bản dự thảo là bản đã nộp.

## Kết quả nghiệm thu ngày 12/11/2026

- Toàn bộ suite: **23 test đạt**, không bỏ qua test dữ liệu thật. Môi trường phát cảnh báo dọn dẹp `pyreadline3` khi thoát, nhưng pytest trả mã 0; đây không phải lỗi test.
- Ba manifest đã kiểm tra bằng nhãn CIFAR-10 thật và sinh lại trùng toàn bộ chỉ số, không chỉ trùng số lượng.
- Pilot ResNet 272.474 tham số: 4.500 train, 5.000 validation, một epoch, 36 cập nhật; validation **846/5.000 = 0,1692**. Evaluator khớp chính xác trainer. Đây là kết quả kiểm tra kỹ thuật sau một epoch, không phải kết quả so sánh chính.
- Train 20,23 giây; train + validation 27,25 giây trên CPU trong phiên này. Không dùng con số này để suy ra lịch GPU RTX 5080 của Minh.
- Bằng chứng: [kiểm tra dữ liệu/pilot](../results/hien_week1_checks.json), [metadata từng run](../results/hien_pilot_runs.csv), [bảng tổng hợp](../results/hien_pilot_summary.csv), [log epoch](../results/hien_pilot_metrics.csv).
- Môi trường thực tế Python 3.12.3, torch 2.12.1+cpu, torchvision 0.27.1; khác môi trường CUDA đã ghim trong requirements của Minh. Phiên bản chi tiết được lưu trong JSON, chưa tuyên bố đã kiểm tra cài sạch theo requirements.
- Checkpoint/nhãn/xác suất/TensorBoard nằm tại `runs/hien_week1_smoke/` (không theo dõi bằng Git); cần sao lưu thư mục này khi bàn giao sang máy khác.
- Proposal: [PDF 2 trang](../reports/proposal.pdf). Họ tên/MSSV, xác nhận của nhóm, số tham số ViT và việc nộp bài vẫn chờ thông tin/thao tác thực tế.

## Giao thức dữ liệu

- CIFAR-10 bản Python chính thức của torchvision, giữ thứ tự `data_batch_1` đến `data_batch_5`.
- 45.000 train / 5.000 validation; mỗi lớp lần lượt 4.500 / 500 ảnh.
- Manifest `results/splits/split_seed{2026,2027,2028}_manifest.json`; mỗi seed có 4.500 / 22.500 / 45.000 ảnh train, lồng nhau.
- Tỷ lệ tính trên 45.000 ảnh, không tính trên 50.000. Hai mô hình phải dùng cùng manifest cho cùng split seed và tỷ lệ.
- Test là toàn bộ 10.000 ảnh của `CIFAR10(train=False)`, không phải chỉ số trong train. Chưa đánh giá test trong tuần 1.
- Train: RandomCrop(32, padding=4), RandomHorizontalFlip, ToTensor, Normalize. Validation/test: ToTensor, Normalize.
- Mean `(0.4914, 0.4822, 0.4465)`, std `(0.2470, 0.2435, 0.2616)` là hằng số chuẩn hóa cố định; không ước lượng lại từ test hoặc từng subset.
- `split_seed` kiểm soát cả train/val và tập con. `seed` kiểm soát khởi tạo/train. Khi thay cả hai, std phản ánh cả hai nguồn biến thiên; không gọi đó là riêng biến thiên khởi tạo.
- Loader kiểm tra phân tầng từ nhãn thật. Checkpoint lưu SHA-256 của manifest; resume/evaluate từ chối nếu manifest thay đổi.

## Lệnh tái lập

Chạy từ thư mục gốc repo, với môi trường đã cài `requirements-dev.txt`:

```powershell
python -X utf8 -m src.data --generate --download
python -X utf8 -m src.data --verify --verify-labels
python -X utf8 -m pytest -q
python -X utf8 -m scripts.check_hien_week1 --run-dir runs/hien_week1_smoke --device cpu
```

Không ghi đè thư mục run đã tồn tại; dùng tên mới nếu chạy lại pilot. Script cuối chạy một epoch ResNet trên 4.500 ảnh thật, validation 5.000 ảnh, so accuracy evaluator với trainer và xuất bằng chứng riêng `results/hien_*`. Không dùng pilot để kết luận CNN tốt hơn ViT.

`configs/hien_smoke.yaml` là config hoàn chỉnh, chạy được qua trainer chung. `configs/data.yaml` là cấu hình tham khảo dữ liệu, không phải config train độc lập. Khi tích hợp ViT, giữ nguyên phần `data` và thay `model.factory` bằng factory của Lan Anh; chưa có mã ViT thì không xác nhận tích hợp đã đạt.

```powershell
python -X utf8 -m src.evaluate --config runs/hien_week1_smoke/config.yaml --checkpoint runs/hien_week1_smoke/best.pt --split val --out-csv results/manual_pilot_runs.csv
python -X utf8 -m scripts.summarize --csv results/manual_pilot_runs.csv --split val --out results/manual_pilot_summary.csv
```

Evaluator mặc định lưu nhãn/xác suất vào `predictions_val.json` cạnh checkpoint. Phải truyền đúng config đã lưu trong run; không lấy checkpoint pilot cũ dùng `src.pilot_data` để đánh giá bằng split mới.

## Metadata và thống kê

Mỗi hàng ghi model, ratio, hai seed, params, config, commit, trạng thái Git, hash source/checkpoint/manifest, protocol, accuracy, best val/epoch, bước cập nhật, thời gian và thời điểm đánh giá. Commit đi kèm trạng thái Git/hash mã nguồn vì khi phát triển có thể còn thay đổi chưa commit.

`checkpoint_step`/`checkpoint_epoch` mô tả checkpoint đang đánh giá. `global_step`, `train_seconds_total` và thời gian trung bình mỗi epoch lấy từ `last.pt` nếu có, để tính cả chi phí train sau epoch tốt nhất.

`protocol_id` phân biệt recipe và phiên bản source; bỏ train/split seed khi nhóm kết quả. Mỗi nhóm còn phân biệt model, ratio, split. Không trộn validation/test ngay cả với `--split all`; không trộn recipe hoặc số tham số khác nhau.

Seed trùng bị từ chối, không tự chọn lần chạy có điểm cao hơn. Với rerun, lưu CSV riêng và công bố lý do chọn run trước khi ghép bảng. Ba seed đề xuất: `(split_seed, seed) = (2026,42), (2027,43), (2028,44)`. Cần giữ các cặp giống nhau cho ViT và ResNet.

Độ lệch chuẩn mẫu dùng `ddof=1`; một seed ghi `NaN/N/A`, không ghi 0 để giả định đã ổn định. `results/runs.csv` dành cho thí nghiệm chính, nên chỉ có header trước khi chạy loạt chính. Bảng pilot và fixture kiểm thử được tách riêng.

## Proposal và việc cần nhóm xác nhận

- Source: `reports/proposal.tex`, BibTeX: `reports/references.bib`; template ACL chính thức được lưu cạnh source.
- Biên dịch từ repo: `tectonic reports/proposal.tex --keep-logs`, hoặc trong `reports/`: XeLaTeX → BibTeX → XeLaTeX hai lần.
- Điền họ tên/MSSV, đọc chéo cả ba, xác nhận phần ViT/số tham số và lưu bản đã nộp sau khi nhóm thực sự nộp.
- PDF đã biên dịch với Tectonic/XeTeX, kiểm tra 2 trang A4 và render từng trang; BibTeX không còn lỗi, không có chữ tràn cột. Còn cảnh báo underfull do căn đều văn bản, không sửa lề/font template.
- ResNet hiện có 272.474 tham số; ngưỡng chênh lệch ≤10% là mục tiêu nhóm, chưa xác nhận khi chưa có ViT.
- Chia sẻ/sao lưu log và checkpoint theo nơi nhóm thống nhất. Các file trong `runs/` không được đưa vào Git.

## Câu hỏi tự kiểm tra khi vấn đáp

1. Vì sao 10% là 4.500 ảnh, không phải 5.000?
2. Vì sao validation không dùng random crop/flip, và test không tham gia chọn checkpoint?
3. Tập con lồng nhau khác ba lần lấy mẫu độc lập như thế nào?
4. Vì sao ba lần đánh giá cùng checkpoint không phải ba seed?
5. Vì sao một seed không đủ để ước lượng std mẫu?
6. Vì sao thay manifest mà giữ nguyên tên file vẫn làm mất tính tái lập?
7. Swin dùng shifted windows và patch merging để làm gì; vì sao không mặc định nó tốt hơn trên CIFAR-10?
