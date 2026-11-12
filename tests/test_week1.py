import csv

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.resnet import CifarResNet, count_parameters
from src.train import fit, run_epoch, seed_everything


# Các bài test nhỏ chạy một luồng CPU để tránh chi phí điều phối nhiều luồng lấn thời gian tính.
torch.set_num_threads(1)


# Test này kiểm tra ResNet trả đúng số logits và mọi tham số đều nhận gradient hữu hạn.
# Không có đối số; pytest coi test đạt khi mọi assert thành công, không cần trả dữ liệu.
def test_resnet_classifies_small_rgb_images_and_has_finite_gradients():
    # Dùng mạng hẹp để test nhanh, vẫn giữ kiến trúc residual cần kiểm tra.
    model = CifarResNet(width=8)
    # Tạo 4 ảnh RGB giả kích thước 32×32 rồi chạy forward.
    logits = model(torch.randn(4, 3, 32, 32))
    # Mỗi ảnh phải có 10 điểm lớp; không nhầm batch size với số lớp.
    assert logits.shape == (4, 10)
    # Nhãn giả 0/1/2/3 hợp lệ trong 10 lớp; backward kiểm tra đường truyền gradient toàn mạng.
    nn.functional.cross_entropy(logits, torch.arange(4)).backward()
    # grad=None là chưa nhận gradient; NaN/Inf là bất thường số học; tất cả tham số phải hợp lệ.
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    # Tăng width phải tăng số tham số, phù hợp cách điều chỉnh quy mô để khớp ViT.
    assert count_parameters(CifarResNet(width=16)) > count_parameters(model)


# Test này kiểm tra mạng thực sự học được một batch cố định, không chỉ forward chạy không lỗi.
# Loss phải giảm mạnh sau cập nhật; đây không phải test accuracy trên dữ liệu chưa thấy.
def test_resnet_can_memorize_a_fixed_batch():
    # Cố định khởi tạo/dữ liệu để kết quả test không thay đổi tùy lần chạy.
    seed_everything(17)
    # Mạng nhỏ (mỗi stage một block) đủ kiểm tra học batch, chạy nhanh trên CPU.
    model = CifarResNet(width=4, blocks_per_stage=1)
    # 8 ảnh ngẫu nhiên và 8 nhãn khác nhau; giữ nguyên chúng suốt 40 bước.
    images, labels = torch.randn(8, 3, 32, 32), torch.arange(8)
    # AdamW với lr tương đối lớn cho mục tiêu ghi nhớ nhanh trong bài test nhỏ.
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    # Loss trước học là mốc đối chiếu, item() chuyển tensor thành số để so sánh sau đó.
    first_loss = nn.functional.cross_entropy(model(images), labels).item()
    # Dấu _ thể hiện không dùng giá trị số thứ tự; chỉ cần thực hiện 40 lần cập nhật.
    for _ in range(40):
        # Reset gradient mỗi bước, nếu không gradient cũ sẽ bị cộng dồn ngoài ý muốn.
        optimizer.zero_grad(set_to_none=True)
        # Forward trên đúng batch cố định, tính loss giữa logits và nhãn giả.
        loss = nn.functional.cross_entropy(model(images), labels)
        # Tính gradient rồi cập nhật tham số như vòng train thật.
        loss.backward()
        optimizer.step()
    # Loss lần forward cuối phải nhỏ hơn 1/5 ban đầu, thay vì chỉ yêu cầu giảm một chút.
    assert loss.item() < first_loss / 5


# Test này kiểm tra validation: tính loss đúng theo từng ảnh và không đổi thống kê BatchNorm.
# Dùng 5 ảnh/batch_size=2 để batch cuối chỉ có 1 ảnh, làm lộ lỗi lấy trung bình các batch sai trọng số.
def test_validation_uses_all_samples_and_does_not_change_batchnorm():
    # Mạng nhỏ có BatchNorm để phát hiện nếu run_epoch vô tình để chế độ train khi validation.
    model = nn.Sequential(nn.BatchNorm1d(2), nn.Linear(2, 2))
    # Mỗi mẫu có 2 đặc trưng và nhãn thuộc 2 lớp; không cần ảnh thật cho phép kiểm tra metric này.
    x, y = torch.randn(5, 2), torch.tensor([0, 1, 0, 1, 0])
    # Chia thành ba batch có 2/2/1 mẫu.
    loader = DataLoader(TensorDataset(x, y), batch_size=2)
    # clone() tạo bản chụp độc lập; nếu giữ tham chiếu trực tiếp thì thay đổi có thể làm cả hai cùng đổi.
    before = model[0].running_mean.clone()
    # Tạo kết quả chuẩn ở eval trước, dùng toàn bộ 5 mẫu cùng lúc.
    model.eval()
    # Không giữ đồ thị gradient trong phép tính đối chiếu.
    with torch.no_grad():
        expected = nn.functional.cross_entropy(model(x), y).item()
    # Bỏ optimizer để run_epoch đi đúng nhánh validation.
    result = run_epoch(model, loader, torch.device('cpu'))
    # Cho phép sai khác số thực rất nhỏ do cách chia batch, nhưng phải khớp loss theo 5 ảnh.
    assert result['loss'] == pytest.approx(expected)
    # Không được bỏ batch cuối chỉ vì nó nhỏ hơn batch_size.
    assert result['samples'] == 5
    # Eval không cập nhật running_mean; đổi giá trị này là dấu hiệu chế độ chạy sai.
    assert torch.equal(model[0].running_mean, before)
    # Validation không backward nên các tham số chưa từng train vẫn không có gradient.
    assert all(p.grad is None for p in model.parameters())


# Hàm phụ loaders: tạo lại cùng bộ dữ liệu giả và hai DataLoader cho test resume.
# Không có đối số; trả (train_loader, val_loader), dùng dữ liệu nhỏ để test nhanh.
# Hai loader cố ý dùng cùng dataset để kiểm tra cơ chế resume, không dùng để đánh giá khái quát hóa.
def loaders():
    # Generator riêng tạo dữ liệu luôn giống nhau dù RNG toàn cục đã đi tới đâu.
    g = torch.Generator().manual_seed(123)
    # 12 mẫu × 4 đặc trưng; nhãn xen kẽ 0/1 do phép chia lấy dư 2.
    data = TensorDataset(torch.randn(12, 4, generator=g), torch.arange(12) % 2)
    # Mỗi lần gọi trả loader mới để mô phỏng khởi động lại tiến trình khi resume.
    return (
        # Train có 3 batch/epoch và shuffle để test việc khôi phục generator thay vì chỉ khôi phục model.
        DataLoader(data, batch_size=4, shuffle=True, generator=torch.Generator().manual_seed(17)),
        # Val có batch_size khác; generator tách riêng và cũng được checkpoint lưu trạng thái.
        DataLoader(data, batch_size=5, generator=torch.Generator().manual_seed(18)),
    )


# Hàm phụ model_for_resume: tạo mạng nhỏ với dropout để thử lưu/khôi phục RNG.
# Không có đối số; trả nn.Sequential mới với cùng trọng số khởi tạo nhờ seed=17.
def model_for_resume():
    # Seed trước khi tạo Linear để hai lần dựng mạng có cùng trọng số ban đầu.
    seed_everything(17)
    # Mạng 4 -> 8 -> 2, dropout ngẫu nhiên bỏ 30% phần tử khi train; eval không bỏ phần tử.
    # Nếu resume sai RNG thì các mask dropout sau khi tiếp tục có thể khác lượt chạy liền.
    return nn.Sequential(nn.Linear(4, 8), nn.Dropout(0.3), nn.Linear(8, 2))


# Cấu hình thử resume có tổng 3 epoch; vẫn giữ epochs=3 ngay cả lần gọi chỉ chạy tới epoch 1.
CONFIG = {'seed': 17, 'training': {'epochs': 3, 'lr': 0.05, 'optimizer': 'sgd',
          'momentum': 0.9, 'weight_decay': 0.0}}


# Test này so chạy liền 3 epoch với chạy 1 epoch rồi dựng lại đối tượng và resume 2 epoch.
# tmp_path là thư mục tạm pytest tự cấp, tránh test ghi vào run thật của Minh.
# Mục tiêu: bắt lỗi mất trọng số/buffer, RNG/dropout/shuffle, lịch lr, bộ đếm và lịch sử log.
def test_resume_matches_uninterrupted_training_including_optimizer_rng_and_schedule(tmp_path):
    # Nhánh chuẩn: một model mới, chạy liên tục đủ 3 epoch.
    full = model_for_resume()
    fit(full, *loaders(), CONFIG, tmp_path / 'full', device='cpu')
    # Nhánh ngắt: cùng khởi tạo/loader/config, chỉ chạy đến hết epoch 1.
    interrupted = model_for_resume()
    fit(interrupted, *loaders(), CONFIG, tmp_path / 'resumed', device='cpu', stop_after_epoch=1)
    # Tạo model khác, không tái dùng đối tượng/optimizer đang giữ trong RAM của lần gọi trước.
    resumed = model_for_resume()
    # *loaders() trải tuple thành hai đối số train_loader/val_loader; nạp last.pt rồi tiếp tục 2–3.
    fit(resumed, *loaders(), CONFIG, tmp_path / 'resumed', device='cpu',
        resume=tmp_path / 'resumed' / 'last.pt')
    # So mọi tensor trọng số/buffer, không chỉ mỗi accuracy (hai mô hình khác vẫn có thể cùng accuracy).
    for key, value in full.state_dict().items():
        # rtol=atol=0 yêu cầu giá trị đúng bằng nhau, không chấp nhận chênh lệch làm tròn.
        torch.testing.assert_close(value, resumed.state_dict()[key], rtol=0, atol=0)
    # Đọc checkpoint cuối hai nhánh để kiểm tra phần trạng thái ngoài trọng số.
    a = torch.load(tmp_path / 'full' / 'last.pt', weights_only=True)
    b = torch.load(tmp_path / 'resumed' / 'last.pt', weights_only=True)
    # 12 mẫu / batch 4 = 3 cập nhật/epoch; 3 epoch phải có đúng 9 bước.
    assert a['global_step'] == b['global_step'] == 9
    # Lịch learning rate phải ở cùng vị trí cuối, không bị khởi động lại sau resume.
    assert a['scheduler'] == b['scheduler']
    # Tiếp tục train vẫn phải nhớ validation accuracy tốt nhất từng đạt, không reset best.
    assert a['best_val_accuracy'] == b['best_val_accuracy']
    # Đọc lại CSV để phát hiện mất hoặc lặp dòng khi resume.
    rows = list(csv.DictReader((tmp_path / 'resumed' / 'metrics.csv').open()))
    assert [int(r['epoch']) for r in rows] == [1, 2, 3]
    # Bên cạnh last.pt phải có best.pt để dùng đánh giá/suy luận.
    assert (tmp_path / 'resumed' / 'best.pt').exists()
    # Có file sự kiện TensorBoard; kiểm tra sự tồn tại, chưa kiểm tra nội dung mọi scalar ở test này.
    assert list((tmp_path / 'resumed' / 'tensorboard').glob('events.*'))


# Test này bảo vệ kết quả: run mới không ghi đè run cũ, resume không được âm thầm đổi config.
# Nhận tmp_path do pytest cấp; thành công khi fit phát đúng loại lỗi mong đợi.
def test_trainer_refuses_overwriting_an_existing_run_or_changed_resume_config(tmp_path):
    # Tạo một run có kết quả epoch 1 để làm tình huống có dữ liệu cần bảo vệ.
    model = model_for_resume()
    fit(model, *loaders(), CONFIG, tmp_path, device='cpu', stop_after_epoch=1)
    # pytest.raises nghĩa là lần gọi bên trong phải báo FileExistsError; không báo lỗi thì test thất bại.
    with pytest.raises(FileExistsError):
        fit(model_for_resume(), *loaders(), CONFIG, tmp_path, device='cpu')
    # Tạo dict cấu hình mới chỉ khác seed, giữ nguyên CONFIG gốc cho các test khác.
    changed = {**CONFIG, 'seed': 42}
    # Đường dẫn đúng nhưng config khác vẫn phải bị chặn; match yêu cầu thông báo có chữ 'config'.
    with pytest.raises(ValueError, match='config'):
        fit(model_for_resume(), *loaders(), changed, tmp_path, device='cpu', resume=tmp_path / 'last.pt')


# Test này kiểm tra lấy mẫu pilot lặp lại được theo seed, không trùng train/val và không lấy quá số ảnh.
# Không tải dataset thật: chỉ cần kiểm tra logic chia chỉ số với giả định có 100 mẫu.
def test_pilot_indices_are_repeatable_disjoint_and_reject_invalid_sizes():
    # Import hàm cần kiểm tra; việc import không tự tải dữ liệu vì chưa gọi build_loaders.
    from src.pilot_data import pilot_indices

    # Lấy 40 train và 20 val, còn 40 chỉ số chưa sử dụng.
    train, val = pilot_indices(100, 40, 20, 2026)
    # Cùng đầu vào/seed phải cho cùng danh sách và thứ tự chỉ số.
    assert (train, val) == pilot_indices(100, 40, 20, 2026)
    # set bỏ phần tử trùng; độ dài đúng chứng minh mỗi nhóm có đủ chỉ số khác nhau.
    assert len(set(train)) == 40 and len(set(val)) == 20
    # isdisjoint=True khi hai nhóm không có bất kỳ chỉ số chung nào.
    assert set(train).isdisjoint(val)
    # 90+20 vượt 100 nên phải bị chặn, không được cắt thiếu ảnh âm thầm.
    with pytest.raises(ValueError):
        pilot_indices(100, 90, 20, 2026)
