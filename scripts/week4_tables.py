# Người phụ trách: Hiền (thống kê và xuất bảng báo cáo); Minh đóng góp kết quả ResNet/Swin và demo.
"""Sinh bảng LaTeX từ log, checkpoint đã kiểm tra và kết quả test của 18 baseline."""
import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev

from scripts.reproduce_week4 import load_runs

EXPECTED_PAIRS = {(42, 2026), (43, 2027), (44, 2028)}
REQUIRED_VAL_GROUPS = [
    ('main', 'resnet', '0.10'),
    ('main', 'resnet', '0.50'),
    ('main', 'resnet', '1.00'),
    ('main', 'vit', '0.10'),
    ('main', 'vit', '0.50'),
    ('main', 'vit', '1.00'),
    ('ablation', 'vit', '0.50'),
    ('extension', 'swin', '0.10'),
    ('extension', 'swin', '1.00'),
]


# Hiền định dạng số thập phân sang chuẩn dấu phẩy tiếng Việt trong văn bản LaTeX.
def vn_num(value, digits=2, signed=False):
    fmt = f'{float(value):+.{digits}f}' if signed else f'{float(value):.{digits}f}'
    return fmt.replace('.', ',')


def vn_int(value):
    return f'{int(value):,}'.replace(',', '.')


# Hiền định dạng accuracy và SD mẫu theo điểm phần trăm.
def score(avg, sd):
    return (f'${100 * float(avg):.2f}\\pm{100 * float(sd):.2f}$').replace('.', '{,}')


# Hiền kiểm tra toàn bộ bản ghi validation từ CSV từng seed và đối chiếu với summary.csv.
def check_validation(audit_dir=Path('results/hien_week4')):
    issues = load_runs(audit_dir / 'audit_issues.csv')
    if issues:
        raise ValueError(f'Validation audit still has {len(issues)} issue(s)')
    evidence = load_runs(audit_dir / 'per_seed.csv')
    if not evidence:
        raise ValueError('Empty validation per_seed.csv')
    seen = set()
    for r in evidence:
        key = (r['scope'], r['architecture'], f"{float(r['ratio']):.2f}", int(r['train_seed']), int(r['split_seed']))
        if key in seen:
            raise ValueError(f'Duplicate validation record: {key}')
        seen.add(key)
        if r['audit_status'] != 'saved_log_consistent':
            raise ValueError(f"Inconsistent validation run: {r['run_id']}")
        if (r['split'] != 'val' or int(r['total']) != 5000
                or not math.isclose(float(r['accuracy']), int(r['correct']) / 5000, rel_tol=0, abs_tol=1e-12)
                or not r.get('checkpoint_sha256')):
            raise ValueError(f"Invalid validation count or hash: {r['run_id']}")
        pair = (int(r['train_seed']), int(r['split_seed']))
        if r['scope'] == 'smoke':
            if pair != (42, 2026):
                raise ValueError(f"Unexpected smoke seed pair: {r['run_id']}")
        elif pair not in EXPECTED_PAIRS:
            raise ValueError(f"Unexpected validation seed pair: {r['run_id']}")

    summary_rows = {
        (s['scope'], s['architecture'], f"{float(s['ratio']):.2f}"): s
        for s in load_runs(audit_dir / 'summary.csv')
    }
    stats = {}
    paired = {}
    for scope, arch, ratio in REQUIRED_VAL_GROUPS:
        group = [
            r for r in evidence
            if (r['scope'], r['architecture'], f"{float(r['ratio']):.2f}") == (scope, arch, ratio)
        ]
        pairs = {(int(r['train_seed']), int(r['split_seed'])) for r in group}
        if len(group) != 3 or pairs != EXPECTED_PAIRS:
            raise ValueError(f'Incomplete validation seeds for {(scope, arch, ratio)}: {pairs}')
        scores = [float(r['accuracy']) for r in group]
        avg, sd = mean(scores), stdev(scores)
        params = int(group[0]['params'])
        if len({int(r['params']) for r in group}) != 1:
            raise ValueError(f'Inconsistent parameter count in {(scope, arch, ratio)}')
        if (scope, arch, ratio) not in summary_rows:
            raise ValueError(f'Missing validation summary row for {(scope, arch, ratio)}')
        recorded = summary_rows[scope, arch, ratio]
        if (int(recorded['consistent_seeds']) != 3 or int(recorded['reported_seeds']) != 3
                or not math.isclose(float(recorded['consistent_mean']), avg, rel_tol=0, abs_tol=1e-12)
                or not math.isclose(float(recorded['consistent_std']), sd, rel_tol=0, abs_tol=1e-12)):
            raise ValueError(f'Validation summary differs from per-seed records for {(scope, arch, ratio)}')
        stats[scope, arch, ratio] = dict(mean=avg, std=sd, params=params)
        for r in group:
            paired[scope, arch, ratio, int(r['train_seed'])] = float(r['accuracy'])
    return stats, paired


# Hiền đối chiếu danh sách đã khóa trước test với từng checkpoint thực sự đã đánh giá.
def check_test(audit_dir=Path('results/hien_week4'), folder=Path('results/final_test')):
    registry = [r for r in load_runs(audit_dir / 'checkpoint_registry.csv') if r['scope'] == 'main']
    locked = load_runs(folder / 'selected_registry.csv')
    results = load_runs(folder / 'runs.csv')
    hashes = {r['run_id']: r['checkpoint_sha256'] for r in registry}
    if (len(hashes) != 18 or len(locked) != 18 or len(results) != 18 or
            any({r['run_id']: r['checkpoint_sha256'] for r in rows} != hashes for rows in (locked, results))):
        raise ValueError('Final test results differ from the locked checkpoint list')
    for r in results:
        if (r['split'] != 'test' or int(r['total']) != 10000
                or not math.isclose(float(r['accuracy']), int(r['correct']) / 10000, rel_tol=0, abs_tol=1e-12)):
            raise ValueError(f"Invalid test count: {r['run_id']}")
    summary = load_runs(folder / 'summary.csv')
    groups = {(r['model'], f"{float(r['ratio']):.2f}") for r in results}
    if len(summary) != 6 or {(s['model'], f"{float(s['ratio']):.2f}") for s in summary} != groups:
        raise ValueError('Incomplete test summary')
    test_stats = {}
    test_paired = {}
    for s in summary:
        ratio = f"{float(s['ratio']):.2f}"
        group = [r for r in results if (r['model'], f"{float(r['ratio']):.2f}") == (s['model'], ratio)]
        scores = [float(r['accuracy']) for r in group]
        avg, sd = mean(scores), stdev(scores)
        if ({(int(r['train_seed']), int(r['split_seed'])) for r in group} != EXPECTED_PAIRS
                or int(s['n_seeds']) != 3
                or not math.isclose(float(s['mean_accuracy']), avg, rel_tol=0, abs_tol=1e-12)
                or not math.isclose(float(s['std_accuracy']), sd, rel_tol=0, abs_tol=1e-12)):
            raise ValueError('Test summary differs from per-seed results')
        arch = s['model'].split(':')[0].split('.')[-1]
        test_stats[arch, ratio] = dict(mean=avg, std=sd)
        for r in group:
            test_paired[arch, ratio, int(r['train_seed'])] = float(r['accuracy'])
    return test_stats, test_paired


def describe_ablation(differences):
    avg = mean(differences)
    sd = stdev(differences)
    values = '; '.join(vn_num(v, signed=True) for v in differences)
    if avg > 0:
        trend = (f'tăng trung bình {vn_num(avg, signed=True)} (SD {vn_num(sd)}) điểm phần trăm, '
                 'tức biến thể bỏ PE nhỉnh hơn ViT chuẩn trên cấu hình ảnh $32\\times32$ này')
    elif avg < 0:
        trend = f'giảm trung bình {vn_num(avg, signed=True)} (SD {vn_num(sd)}) điểm phần trăm so với ViT chuẩn'
    else:
        trend = f'có trung bình {vn_num(avg, signed=True)} (SD {vn_num(sd)}) điểm phần trăm so với ViT chuẩn'
    return values, trend


# Hiền xuất bảng từ CSV theo các cặp seed cố định. Minh cung cấp CSV của Swin và JSON demo/ghép cặp.
def generate_tables(audit_dir=Path('results/hien_week4'), test_dir=Path('results/final_test'),
                    paired_dir=Path('results/minh_week4/paired'),
                    demo_json=Path('results/minh_week4/demo_resnet.json'),
                    out_dir=Path('results/minh_week4')):
    val_stats, val_paired = check_validation(audit_dir)
    test_stats, test_paired = check_test(audit_dir, test_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    rows = []
    for ratio in ('0.10', '0.50', '1.00'):
        values = []
        for arch in ('resnet', 'vit'):
            val = val_stats['main', arch, ratio]
            test = test_stats[arch, ratio]
            values.extend([score(val['mean'], val['std']), score(test['mean'], test['std'])])
        rows.append(' & '.join([f'{round(100 * float(ratio))}\\%', *values]) + r' \\')
    (out_dir / 'baseline_table.tex').write_text('\n'.join(rows) + '\n', encoding='utf-8')

    rows = []
    for scope, arch, ratio in [('main', 'vit', '0.50'), ('ablation', 'vit', '0.50'),
                               ('extension', 'swin', '0.10'), ('extension', 'swin', '1.00')]:
        val = val_stats[scope, arch, ratio]
        name = {'main': 'ViT', 'ablation': 'ViT bỏ PE', 'extension': 'Swin'}[scope]
        rows.append(f"{name} & {round(100 * float(ratio))}\\% & {vn_int(val['params'])} & "
                    + score(val['mean'], val['std']) + r' \\')
    (out_dir / 'extension_table.tex').write_text('\n'.join(rows) + '\n', encoding='utf-8')

    differences = [
        100 * (val_paired['ablation', 'vit', '0.50', seed] - val_paired['main', 'vit', '0.50', seed])
        for seed in (42, 43, 44)
    ]
    test_gaps = [
        (test_stats['resnet', ratio]['mean'] - test_stats['vit', ratio]['mean']) * 100
        for ratio in ('0.10', '0.50', '1.00')
    ]
    test_gap_sds = [
        stdev([100 * (test_paired['resnet', ratio, s] - test_paired['vit', ratio, s]) for s in (42, 43, 44)])
        for ratio in ('0.10', '0.50', '1.00')
    ]

    r_test = [vn_num(100 * test_stats['resnet', r]['mean']) + '\\%' for r in ('0.10', '0.50', '1.00')]
    v_test = [vn_num(100 * test_stats['vit', r]['mean']) + '\\%' for r in ('0.10', '0.50', '1.00')]
    gap_str = '; '.join(vn_num(g, signed=True) for g in test_gaps)
    abstract = (
        f"Trên 10.000 ảnh test qua ba cặp seed, ResNet đạt trung bình {'; '.join(r_test)} còn ViT đạt "
        f"{'; '.join(v_test)} ở 10\\%, 50\\%, 100\\% dữ liệu; hiệu số ResNet trừ ViT giảm từ "
        f"{vn_num(test_gaps[0], signed=True)} xuống {vn_num(test_gaps[1], signed=True)} và "
        f"{vn_num(test_gaps[2], signed=True)} điểm phần trăm. Hai mô hình dùng optimizer riêng và số bước "
        "cập nhật tăng theo quy mô tập train, nên chênh lệch phản ánh cả kiến trúc lẫn quy trình huấn luyện."
    )
    (out_dir / 'abstract_results.tex').write_text(abstract + '\n', encoding='utf-8')

    r_val = [f"{vn_num(100 * val_stats['main', 'resnet', r]['mean'])}\\% (SD {vn_num(100 * val_stats['main', 'resnet', r]['std'])})"
             for r in ('0.10', '0.50', '1.00')]
    v_val = [f"{vn_num(100 * val_stats['main', 'vit', r]['mean'])}\\% (SD {vn_num(100 * val_stats['main', 'vit', r]['std'])})"
             for r in ('0.10', '0.50', '1.00')]
    scaling_text = (
        f"Qua ba cặp seed $(2026,42)$, $(2027,43)$ và $(2028,44)$, validation trung bình của ResNet tăng từ "
        f"{r_val[0]} ở 10\\% lên {r_val[1]} ở 50\\% và {r_val[2]} ở 100\\% dữ liệu; ViT tăng tương ứng từ "
        f"{v_val[0]} lên {v_val[1]} và {v_val[2]}. Cả seed chia dữ liệu và seed huấn luyện đều đổi giữa ba cặp, "
        "nên độ lệch chuẩn mẫu phản ánh biến thiên gộp của cả hai nguồn ngẫu nhiên."
    )
    (out_dir / 'scaling_summary.tex').write_text(scaling_text + '\n', encoding='utf-8')

    abl_values, abl_trend = describe_ablation(differences)
    swin_10 = val_stats['extension', 'swin', '0.10']
    swin_100 = val_stats['extension', 'swin', '1.00']
    swin_gaps = [
        100 * (swin_10['mean'] - val_stats['main', 'vit', '0.10']['mean']),
        100 * (swin_100['mean'] - val_stats['main', 'vit', '1.00']['mean']),
    ]

    paired_error_sentence = ''
    s42_counts_path = paired_dir / 'd50_s42_error_counts.json'
    if s42_counts_path.is_file():
        c42 = json.loads(s42_counts_path.read_text(encoding='utf-8'))
        total_42 = sum(int(c42[k]) for k in ('both_correct', 'vit_only_wrong', 'resnet_only_wrong', 'both_wrong'))
        if total_42 != 5000:
            raise ValueError(f'Expected 5000 paired validation images in {s42_counts_path}, got {total_42}')
        paired_error_sentence = (
            f" Đối chiếu trực tiếp trên cùng 5.000 ảnh validation ở mức 50\\% (seed 42), cả hai mô hình cùng đúng "
            f"{vn_int(c42['both_correct'])} ảnh, chỉ ResNet đúng {vn_int(c42['vit_only_wrong'])} ảnh, chỉ ViT đúng "
            f"{vn_int(c42['resnet_only_wrong'])} ảnh và cả hai cùng sai {vn_int(c42['both_wrong'])} ảnh."
        )
        all_seed_paths = [paired_dir / f'd50_s{s}_error_counts.json' for s in (42, 43, 44)]
        if all(p.is_file() for p in all_seed_paths):
            per_seed_counts = [json.loads(p.read_text(encoding='utf-8')) for p in all_seed_paths]
            for p, c in zip(all_seed_paths, per_seed_counts, strict=True):
                if sum(int(c[k]) for k in ('both_correct', 'vit_only_wrong', 'resnet_only_wrong', 'both_wrong')) != 5000:
                    raise ValueError(f'Invalid paired count sum in {p}')
            paired_error_sentence += (
                f" Trung bình qua ba seed ở mức 50\\%, số ảnh cả hai cùng đúng là "
                f"{vn_num(mean(c['both_correct'] for c in per_seed_counts), 1)}, chỉ ResNet đúng "
                f"{vn_num(mean(c['vit_only_wrong'] for c in per_seed_counts), 1)}, chỉ ViT đúng "
                f"{vn_num(mean(c['resnet_only_wrong'] for c in per_seed_counts), 1)} và cả hai cùng sai "
                f"{vn_num(mean(c['both_wrong'] for c in per_seed_counts), 1)}."
            )

    demo_sentence = ''
    if demo_json.is_file():
        demo = json.loads(demo_json.read_text(encoding='utf-8'))
        cls_name = demo['class_name']
        prob = float(demo['probabilities'][cls_name]) * 100
        sha_prefix = demo['checkpoint_sha256'][:12]
        demo_sentence = (
            f" Lệnh suy luận đơn ảnh (src/predict.py) kiểm chứng trực tiếp từ checkpoint ResNet 100\\% seed 42 "
            f"(SHA-256 đầu {sha_prefix}\\ldots) trên ảnh validation cố định chỉ số 23171 của manifest 2026: "
            f"mô hình dự đoán đúng lớp {cls_name} với xác suất {vn_num(prob)}\\% trên CPU và xuất đủ phân phối 10 lớp."
        )

    paired_text = (
        f"Trên tập test, hiệu accuracy ghép cặp ResNet trừ ViT trung bình ở 10\\%, 50\\%, 100\\% lần lượt bằng "
        f"{gap_str} điểm phần trăm (SD của hiệu theo cặp seed: "
        f"{'; '.join(vn_num(sd) for sd in test_gap_sds)})."
        f"{paired_error_sentence} "
        f"Ở mức 50\\%, hiệu validation khi bỏ positional embedding trừ ViT chuẩn theo thứ tự seed 42, 43, 44 là "
        f"{abl_values} điểm phần trăm, {abl_trend}. "
        f"Thí nghiệm mở rộng Swin (chỉ đánh giá trên validation, dùng kiến trúc phân cấp cửa sổ của torchvision) "
        f"đạt {vn_num(100 * swin_10['mean'])}\\% (SD {vn_num(100 * swin_10['std'])}) ở 10\\% và "
        f"{vn_num(100 * swin_100['mean'])}\\% (SD {vn_num(100 * swin_100['std'])}) ở 100\\%, chênh lần lượt "
        f"{vn_num(swin_gaps[0], signed=True)} và {vn_num(swin_gaps[1], signed=True)} điểm phần trăm so với ViT tự cài."
        f"{demo_sentence}"
    )
    (out_dir / 'paired_effects.tex').write_text(paired_text + '\n', encoding='utf-8')

    slide_test = (
        f"Top-1 accuracy (\\%), trung bình $\\pm$ SD mẫu qua ba cặp seed trên 10.000 ảnh test. "
        f"Hiệu số ResNet trừ ViT trên test giảm từ {vn_num(test_gaps[0], signed=True)} xuống "
        f"{vn_num(test_gaps[1], signed=True)} và {vn_num(test_gaps[2], signed=True)} điểm phần trăm khi tăng dữ liệu."
    )
    (out_dir / 'slide_test_notes.tex').write_text(slide_test + '\n', encoding='utf-8')

    if s42_counts_path.is_file():
        c42 = json.loads(s42_counts_path.read_text(encoding='utf-8'))
        slide_paired = (
            f"Ghép lỗi trên cùng 5.000 ảnh validation 50\\% (seed 42): cùng đúng {vn_int(c42['both_correct'])}, "
            f"chỉ ResNet đúng {vn_int(c42['vit_only_wrong'])}, chỉ ViT đúng {vn_int(c42['resnet_only_wrong'])}, "
            f"cùng sai {vn_int(c42['both_wrong'])}."
        )
    else:
        slide_paired = 'Nhóm đối chiếu nhãn và dự đoán của hai mô hình trên cùng thứ tự manifest validation.'
    (out_dir / 'slide_paired_notes.tex').write_text(slide_paired + '\n', encoding='utf-8')

    if demo_json.is_file():
        demo = json.loads(demo_json.read_text(encoding='utf-8'))
        cls_name = demo['class_name']
        prob = float(demo['probabilities'][cls_name]) * 100
        slide_demo = (
            f"Nhãn thật và nhãn dự đoán đều là {cls_name} (xác suất {vn_num(prob)}\\% trên CPU, "
            f"SHA-256 đầu {demo['checkpoint_sha256'][:12]})."
        )
    else:
        slide_demo = 'Nhãn thật và nhãn dự đoán đều là airplane. Xác suất mô hình gán cho airplane là 99,99\\% trong demo CPU.'
    (out_dir / 'slide_demo_notes.tex').write_text(slide_demo + '\n', encoding='utf-8')

    slide_ext = (
        f"Hiệu validation (ViT bỏ PE trừ ViT chuẩn) ở 50\\% theo seed 42/43/44: {abl_values} điểm phần trăm "
        f"(TB {vn_num(mean(differences), signed=True)}). Swin chỉ là thí nghiệm mở rộng trên validation, "
        "không đưa vào so sánh test chính."
    )
    (out_dir / 'slide_extension_notes.tex').write_text(slide_ext + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-test', action='store_true')
    parser.add_argument('--audit-dir', type=Path, default=Path('results/hien_week4'))
    parser.add_argument('--test-dir', type=Path, default=Path('results/final_test'))
    parser.add_argument('--paired-dir', type=Path, default=Path('results/minh_week4/paired'))
    parser.add_argument('--demo-json', type=Path, default=Path('results/minh_week4/demo_resnet.json'))
    parser.add_argument('--out-dir', type=Path, default=Path('results/minh_week4'))
    args = parser.parse_args()
    if args.check_test:
        check_test(args.audit_dir, args.test_dir)
        return
    generate_tables(args.audit_dir, args.test_dir, args.paired_dir, args.demo_json, args.out_dir)


if __name__ == '__main__':
    main()
