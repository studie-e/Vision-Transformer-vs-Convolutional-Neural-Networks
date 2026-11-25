# Người phụ trách: Hiền (đóng gói báo cáo nội bộ, slide, bảng kiểm tra và mã tái lập tuần 4).
"""Package the internal report, slides, audit and reproduction sources."""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


def main():
    output = Path('results/hien_week4')
    paths = [
        Path(name) for name in (
            'docs/hien-week4.md', 'docs/minh-week4.md', 'docs/defense_questions.md', 'docs/experiment_protocol.md',
            'reports/final.tex', 'reports/final.pdf', 'reports/milestone.tex', 'reports/references.bib',
            'reports/acl.sty', 'reports/acl_natbib.bst', 'slides/week4.tex', 'slides/week4.pdf',
            'scripts/reproduce_week4.py', 'scripts/evaluate_final.py',
            'scripts/package_hien_week4.py',
            'scripts/run_lanh_week3.py', 'scripts/run_minh_week3.py', 'src/predict.py', 'tests/test_week4_audit.py',
            'requirements.txt', 'requirements-dev.txt')
    ]
    if Path('reports/milestone.pdf').is_file():
        paths.append(Path('reports/milestone.pdf'))
    paths += list(Path('reports/sections').glob('*.tex'))
    # Hiền đóng gói nguồn thực sự dùng để dựng lại bảng và PDF sau khi hoàn thành test.
    if Path('results/final_test/summary.csv').is_file():
        paths += list(Path('results/final_test').glob('*.csv'))
        paths += list(Path('results/minh_week4').glob('*.tex'))
        paths += [Path(p) for p in ('scripts/week4_tables.py', 'scripts/handoff_week4.py',
                  'scripts/finish_week4.ps1', 'reports/acl.sty', 'reports/acl_natbib.bst',
                  'reports/references.bib', 'results/minh_week4/handoff.json',
                  'results/minh_week4/verified.csv', 'results/minh_week4/demo_input.png',
                  'results/minh_week4/demo_resnet.json', 'results/team_week2/visualizations/5_vit_0.50_attention.png')]
        if Path('results/minh_week4/baseline_18_val_runs.csv').is_file():
            paths.append(Path('results/minh_week4/baseline_18_val_runs.csv'))
        if Path('results/minh_week4/paired').is_dir():
            paths += [p for p in Path('results/minh_week4/paired').iterdir() if p.is_file()]
    paths += [Path(p) for p in json.loads((output / 'audit.json').read_text(encoding='utf-8'))['sources']]
    paths += [Path(r['metrics_path']) for r in csv_rows(output / 'per_seed.csv')]
    paths += list(Path('results/splits').glob('*manifest.json'))
    paths += [p for p in output.iterdir() if p.suffix in ('.csv', '.json', '.png')
              and p.name not in ('package_manifest.json',)]
    entries = {p.as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))}
    manifest = output / 'package_manifest.json'
    manifest.write_text(json.dumps(entries, indent=2), encoding='utf-8')
    archive = output / 'hien_week4_internal.zip'
    temporary = archive.with_suffix('.tmp')
    with ZipFile(temporary, 'w', ZIP_DEFLATED) as bundle:
        for filename in entries:
            bundle.write(filename, filename)
        bundle.write(manifest, manifest.as_posix())
    with ZipFile(temporary) as bundle:
        if bundle.testzip() is not None:
            raise ValueError('ZIP CRC verification failed')
    temporary.replace(archive)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_suffix('.sha256.txt').write_text(f'{digest}  {archive.name}\n', encoding='utf-8')
    print(f'{archive}: {len(entries)} files plus SHA-256 manifest ({digest[:12]}...)')


def csv_rows(path):
    import csv
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


if __name__ == '__main__':
    main()
