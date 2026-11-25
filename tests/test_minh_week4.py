# Người đóng góp: Minh (kiểm thử quy trình tái lập, xuất bảng và khóa checkpoint tuần 4).
"""Kiểm thử đơn vị cho các công cụ tuần 4: bảng số LaTeX, kiểm tra khóa test và đối chiếu tái lập."""

import json
import statistics
from pathlib import Path

import pytest
import torch

from scripts.evaluate_final import preflight
from scripts.handoff_week4 import compare_repeat
from scripts.reproduce_week4 import write_csv
from scripts.week4_tables import check_test, check_validation, generate_tables


def _build_week4_fixtures(tmp_path, *, ablation_offset=0.012):
    """Dựng bộ CSV fixture hợp lệ biết trước đáp án để kiểm thử logic kiểm tra và sinh bảng."""
    audit_dir = tmp_path / 'audit'
    test_dir = tmp_path / 'final_test'
    paired_dir = tmp_path / 'paired'
    out_dir = tmp_path / 'out'
    audit_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    paired_dir.mkdir(parents=True, exist_ok=True)

    write_csv(audit_dir / 'audit_issues.csv', [], ['run_id', 'evidence', 'issue'])

    per_seed = []
    registry = []
    summaries = []
    test_runs = []
    test_summaries = []

    groups = [
        ('main', 'resnet', 'src.models.resnet:build_model', '0.10', 272474, [3712, 3707, 3732]),
        ('main', 'resnet', 'src.models.resnet:build_model', '0.50', 272474, [4420, 4414, 4414]),
        ('main', 'resnet', 'src.models.resnet:build_model', '1.00', 272474, [4590, 4591, 4591]),
        ('main', 'vit', 'src.models.vit:build_model', '0.10', 274282, [2608, 2610, 2615]),
        ('main', 'vit', 'src.models.vit:build_model', '0.50', 274282, [3383, 3409, 3408]),
        ('main', 'vit', 'src.models.vit:build_model', '1.00', 274282, [3795, 3790, 3800]),
        ('ablation', 'vit', 'src.models.vit:build_model', '0.50', 268042,
         [round((c / 5000 + ablation_offset) * 5000) for c in (3383, 3409, 3408)]),
        ('extension', 'swin', 'src.models.swin:build_model', '0.10', 261824, [2450, 2460, 2455]),
        ('extension', 'swin', 'src.models.swin:build_model', '1.00', 261824, [3950, 3960, 3955]),
    ]

    for scope, arch, factory, ratio, params, corrects in groups:
        accs = [c / 5000 for c in corrects]
        for (seed, split_seed), correct, acc in zip([(42, 2026), (43, 2027), (44, 2028)], corrects, accs, strict=True):
            run_id = f'{scope}_{arch}_{round(float(ratio)*100)}_s{seed}'
            sha = f'sha_{run_id}'
            per_seed.append(dict(
                run_id=run_id, scope=scope, architecture=arch, model=factory, ratio=ratio,
                split_seed=str(split_seed), train_seed=str(seed), split='val',
                correct=str(correct), total='5000', accuracy=str(acc), params=str(params),
                checkpoint_sha256=sha, audit_status='saved_log_consistent',
            ))
            if scope == 'main':
                reg_row = dict(
                    run_id=run_id, scope=scope, architecture=arch, ratio=ratio,
                    split_seed=str(split_seed), train_seed=str(seed),
                    checkpoint_sha256=sha, audit_status='saved_log_consistent',
                )
                registry.append(reg_row)
                test_correct = correct * 2 - 10
                test_runs.append(dict(
                    run_id=run_id, model=factory, ratio=ratio,
                    split_seed=str(split_seed), train_seed=str(seed),
                    checkpoint_sha256=sha, split='test',
                    correct=str(test_correct), total='10000', accuracy=str(test_correct / 10000),
                ))
        summaries.append(dict(
            scope=scope, architecture=arch, ratio=ratio,
            reported_seeds='3', consistent_seeds='3',
            consistent_mean=str(statistics.mean(accs)), consistent_std=str(statistics.stdev(accs)),
            reported_mean=str(statistics.mean(accs)), reported_std=str(statistics.stdev(accs)),
            params=str(params), train_seconds='100.0',
        ))
        if scope == 'main':
            t_scores = [float(r['accuracy']) for r in test_runs if r['model'] == factory and r['ratio'] == ratio]
            test_summaries.append(dict(
                model=factory, ratio=ratio, split='test', n_seeds='3',
                mean_accuracy=str(statistics.mean(t_scores)), std_accuracy=str(statistics.stdev(t_scores)),
            ))

    write_csv(audit_dir / 'per_seed.csv', per_seed, list(per_seed[0]))
    write_csv(audit_dir / 'summary.csv', summaries, list(summaries[0]))
    write_csv(audit_dir / 'checkpoint_registry.csv', registry, list(registry[0]))
    write_csv(test_dir / 'selected_registry.csv', registry, list(registry[0]))
    write_csv(test_dir / 'runs.csv', test_runs, list(test_runs[0]))
    write_csv(test_dir / 'summary.csv', test_summaries, list(test_summaries[0]))

    (paired_dir / 'd50_s42_error_counts.json').write_text(
        json.dumps({'both_correct': 3300, 'vit_only_wrong': 1120, 'resnet_only_wrong': 83, 'both_wrong': 497}),
        encoding='utf-8',
    )
    demo_path = tmp_path / 'demo_resnet.json'
    demo_path.write_text(
        json.dumps({
            'label': 0, 'class_name': 'airplane',
            'probabilities': {'airplane': 0.999908, 'bird': 0.000092},
            'checkpoint_sha256': 'f0a24168f6d2228b8e0cecddae34cf1d7ea17b89f7f7a7350caa47263cdc69e8',
            'device': 'cpu',
        }),
        encoding='utf-8',
    )
    return audit_dir, test_dir, paired_dir, demo_path, out_dir


def test_week4_tables_supports_ablation_better_and_worse_than_baseline(tmp_path):
    audit_dir, test_dir, paired_dir, demo_path, out_dir = _build_week4_fixtures(tmp_path, ablation_offset=0.012)
    generate_tables(audit_dir, test_dir, paired_dir, demo_path, out_dir)
    paired_tex = (out_dir / 'paired_effects.tex').read_text(encoding='utf-8')
    ext_tex = (out_dir / 'extension_table.tex').read_text(encoding='utf-8')
    base_tex = (out_dir / 'baseline_table.tex').read_text(encoding='utf-8')
    assert '+1,20' in paired_tex
    assert 'tăng trung bình +1,20' in paired_tex
    assert '3.300' in paired_tex and '23171' in paired_tex
    assert 'ViT bỏ PE & 50\\% & 268.042' in ext_tex
    assert len(base_tex.strip().splitlines()) == 3

    # Kiểm tra cả trường hợp ablation thấp hơn baseline
    audit_neg, test_neg, paired_neg, demo_neg, out_neg = _build_week4_fixtures(tmp_path / 'neg', ablation_offset=-0.015)
    generate_tables(audit_neg, test_neg, paired_neg, demo_neg, out_neg)
    neg_tex = (out_neg / 'paired_effects.tex').read_text(encoding='utf-8')
    assert '-1,50' in neg_tex
    assert 'giảm trung bình -1,50' in neg_tex


def test_week4_tables_rejects_corrupted_audit_or_unlocked_test_hash(tmp_path):
    audit_dir, test_dir, _, _, _ = _build_week4_fixtures(tmp_path)
    check_validation(audit_dir)
    check_test(audit_dir, test_dir)

    # Đổi một hash trong test/runs.csv => phải bị từ chối vì lệch danh sách khóa
    runs_text = (test_dir / 'runs.csv').read_text(encoding='utf-8').replace('sha_main_vit_50_s42', 'tampered_sha')
    (test_dir / 'runs.csv').write_text(runs_text, encoding='utf-8')
    with pytest.raises(ValueError, match='locked checkpoint'):
        check_test(audit_dir, test_dir)

    # Thêm một dòng lỗi vào audit_issues.csv => check_validation phải chặn ngay
    write_csv(audit_dir / 'audit_issues.csv',
              [{'run_id': 'bad', 'evidence': 'x.csv', 'issue': 'total_time'}],
              ['run_id', 'evidence', 'issue'])
    with pytest.raises(ValueError, match='issue'):
        check_validation(audit_dir)


def test_compare_repeat_verifies_exact_tensors_and_learning_history(tmp_path):
    ref = tmp_path / 'ref'
    rep = tmp_path / 'rep'
    ref.mkdir()
    rep.mkdir()
    payload = dict(
        config={'seed': 42}, manifest_sha256='abc', best_epoch=1, epoch=1,
        global_step=36, best_val_accuracy=0.75,
        model={'w': torch.tensor([1.0, -2.5])},
        history=[dict(epoch=1, global_step=36, lr=0.05, train_loss=1.2,
                      val_loss=1.0, train_accuracy=0.6, val_accuracy=0.75)],
    )
    for name in ('best.pt', 'last.pt'):
        torch.save(payload, ref / name)
        torch.save({**payload, 'model': {'w': payload['model']['w'].clone()}}, rep / name)
    result = compare_repeat(ref, rep)
    assert result['model_tensors_equal'] is True
    assert result['epoch_metrics_equal'] is True

    # Thay đổi 1 phần tử trọng số => compare_repeat phải phát hiện
    bad = {**payload, 'model': {'w': torch.tensor([1.0, -2.4999])}}
    torch.save(bad, rep / 'best.pt')
    with pytest.raises(ValueError, match='different tensor'):
        compare_repeat(ref, rep)


def test_preflight_blocks_incomplete_registry():
    errors = preflight([dict(
        run_id='vit_10_seed43', architecture='vit', ratio='0.10',
        split_seed='2027', train_seed='43', audit_status='invalid_saved_log',
        checkpoint_path='runs/missing/best.pt', config_path='runs/missing/config.yaml',
        checkpoint_sha256='deadbeef',
    )])
    assert any('18 locked' in e for e in errors)
    assert any('vit_10_seed43' in e for e in errors)
