"""Prototype Swin ảnh nhỏ dùng torchvision, chưa phải lượt mở rộng đã huấn luyện.

Nguồn API/kiến trúc thư viện:
https://docs.pytorch.org/vision/0.26/_modules/torchvision/models/swin_transformer.html
"""

from torchvision.models.swin_transformer import SwinTransformer


# Hàm build_model: tạo Swin cho CIFAR-10 theo factory mà trainer đang sử dụng.
# Nhận config đầy đủ; trả mạng mới khởi tạo, không tải bất kỳ pretrained weights nào.
# Prototype dùng 3 stage: patch 2×2 tạo lưới 16×16, patch merging giảm còn 8×8 rồi 4×4.
def build_model(config):
    # Độ rộng 26/52/104 chia hết cho 2/4/8 head; cửa sổ 4×4 hợp với ba lưới trên.
    # Giữ mô hình nhỏ để thuận tiện khảo sát sau; số tham số vẫn phải được báo cáo riêng.
    kwargs = {
        'patch_size': [2, 2], 'embed_dim': 26, 'depths': [2, 2, 2],
        'num_heads': [2, 4, 8], 'window_size': [4, 4], 'mlp_ratio': 2.0,
        'dropout': 0.0, 'attention_dropout': 0.0,
        'stochastic_depth_prob': 0.1, 'num_classes': 10,
    }
    # Cho phép YAML ghi rõ/thay từng giá trị, đồng thời không sửa dict config bên gọi.
    kwargs.update(config['model'].get('kwargs', {}))
    # Constructor này tạo mạng ngẫu nhiên; output forward là logits [B,10].
    return SwinTransformer(**kwargs)
