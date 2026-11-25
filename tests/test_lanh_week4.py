# Người phụ trách: Lan Anh (kiểm thử tự động bàn giao hoàn chỉnh Tuần 4).
"""Kiểm thử tự động các sản phẩm Tuần 4 của Lan Anh:
- Audit tuần 4 không còn bất kỳ bản ghi lỗi nào (invalid_run_ids == [], audit_issues.csv rỗng).
- Không còn script sinh dữ liệu mô phỏng trong scripts/; thư mục lưu vết thu hồi đầy đủ.
- Demo suy luận đơn ảnh (demo_prediction.json) hợp lệ, tổng xác suất = 1.
- Đủ các mục LaTeX, sơ đồ kiến trúc, PDF báo cáo/slide và tài liệu vấn đáp tuần 4.
"""

import csv
import json
from pathlib import Path

import pytest


def test_week4_audit_has_zero_invalid_runs():
    audit = json.loads(Path('results/hien_week4/audit.json').read_text(encoding='utf-8'))
    assert audit['invalid_run_ids'] == []
    assert audit['main_records'] >= 14
    assert audit['test_evaluated'] is False

    with Path('results/hien_week4/audit_issues.csv').open(encoding='utf-8', newline='') as f:
        issues = list(csv.DictReader(f))
    assert issues == []


def test_no_synthetic_generators_and_retraction_traceability_preserved():
    assert not Path('scripts/generate_metrics_lanh_week3.py').exists()
    assert not Path('scripts/build_lanh_week3_results.py').exists()
    assert Path('results/week3/retracted_afe90b0/README.md').is_file()


def test_demo_single_image_prediction_output():
    pred_path = Path('results/week4_lanh/demo_prediction.json')
    assert pred_path.is_file()
    data = json.loads(pred_path.read_text(encoding='utf-8'))
    assert data['label'] == 0
    assert data['class_name'] == 'airplane'
    probs = data['probabilities']
    assert len(probs) == 10
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-5)


def test_lanh_week4_report_slides_and_docs_exist():
    required_files = [
        Path('reports/sections/method_vit.tex'),
        Path('reports/sections/ablation.tex'),
        Path('reports/figures/vit_architecture_lanh.pdf'),
        Path('reports/figures/vit_architecture_lanh.png'),
        Path('reports/final.tex'),
        Path('reports/final.pdf'),
        Path('slides/week4.tex'),
        Path('slides/week4.pdf'),
        Path('docs/lanh-week4.md'),
        Path('docs/lanh-defense-qa-week4.md'),
        Path('docs/vit-gpu-week4-minh-handoff.md'),
        Path('results/week4_lanh/visualizations/1_vit_0.10_attention.png'),
        Path('results/week4_lanh/visualizations/2_vit_0.10_attention.png'),
    ]
    for path in required_files:
        assert path.is_file(), f'Thiếu sản phẩm tuần 4: {path}'
