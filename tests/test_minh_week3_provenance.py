# Người phụ trách: Minh (kiểm thử đối chiếu phiên bản mã huấn luyện).
# Kiểm tra khả năng phân biệt sửa chú thích với thay đổi mã chạy hoặc thư viện khi so sánh các lượt train.
"""Kiểm tra đối chiếu mã huấn luyện giữa các commit của tuần 2 và tuần 3."""

import subprocess
from pathlib import Path

import pytest

from scripts.run_minh_week3 import verify_training_source


# Dựng lịch sử Git riêng để kiểm tra nhận diện thay đổi thật và ghi chú người phụ trách.
@pytest.mark.parametrize('modified', ['labels_only', 'python_logic', 'dependency', 'new_module'])
def test_training_provenance_accepts_only_non_executable_edits(tmp_path, monkeypatch, modified):
    monkeypatch.chdir(tmp_path)
    subprocess.run(['git', 'init', '-q'], check=True)
    train = Path('src/train.py')
    train.parent.mkdir()
    train.write_text('epochs = 100\n', encoding='utf-8')
    requirements = Path('requirements.txt')
    requirements.write_text('torch==2.11.0\n', encoding='utf-8')

    # Lưu snapshot tại một commit để verify_training_source đọc đúng phiên bản qua Git.
    def snapshot():
        subprocess.run(['git', 'add', '.'], check=True)
        subprocess.run(['git', '-c', 'user.name=Test', '-c', 'user.email=test@example.com',
                        'commit', '-qm', 'fixture'], check=True)
        return {'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                'git_status': '', 'run_id': 'fixture'}

    baseline = snapshot()
    label = '# Người phụ trách: Minh\n'
    train.write_text(label + ('epochs = 50\n' if modified == 'python_logic' else 'epochs = 100\n'),
                     encoding='utf-8')
    requirements.write_text(label + ('torch==2.10.0\n' if modified == 'dependency' else 'torch==2.11.0\n'),
                            encoding='utf-8')
    if modified == 'new_module':
        (train.parent / 'extra.py').write_text('x = 1\n', encoding='utf-8')
    repeat = snapshot()
    if modified == 'labels_only':
        verify_training_source(baseline, repeat)
    else:
        with pytest.raises(ValueError, match='Different training sources'):
            verify_training_source(baseline, repeat)
