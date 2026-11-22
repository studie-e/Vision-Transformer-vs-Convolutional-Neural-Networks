"""Package the internal report, slides, audit and reproduction sources."""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


def main():
    output = Path('results/hien_week4')
    paths = [
        Path(name) for name in (
            'docs/hien-week4.md', 'docs/defense_questions.md', 'docs/experiment_protocol.md',
            'reports/final.tex', 'reports/final.pdf', 'reports/references.bib',
            'reports/acl.sty', 'reports/acl_natbib.bst', 'slides/week4.tex', 'slides/week4.pdf',
            'scripts/reproduce_week4.py', 'scripts/evaluate_final.py',
            'scripts/package_hien_week4.py',
            'scripts/run_lanh_week3.py', 'src/predict.py', 'tests/test_week4_audit.py',
            'requirements.txt', 'requirements-dev.txt')
    ]
    paths += list(Path('reports/sections').glob('*.tex'))
    paths += [Path(p) for p in json.loads((output / 'audit.json').read_text())['sources']]
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
    print(f'{archive}: {len(entries)} files plus SHA-256 manifest')


def csv_rows(path):
    import csv
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


if __name__ == '__main__':
    main()
