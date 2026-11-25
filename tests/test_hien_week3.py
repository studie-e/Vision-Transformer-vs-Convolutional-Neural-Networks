import pytest

from scripts.analyze_hien_week3 import paired_errors, fixed_positions
from scripts.summarize import _mean_std


def test_paired_errors_preserve_image_identity_and_four_groups():
    rows = paired_errors([0, 2, 1, 2], [0, 1, 2, 3], [0, 1, 1, 1], [40, 10, 30, 20])
    assert [r['index'] for r in rows] == [40, 10, 30, 20]
    assert [r['group'] for r in rows] == [
        'both_correct', 'vit_only_wrong', 'resnet_only_wrong', 'both_wrong']
    with pytest.raises(ValueError):
        paired_errors([0], [0, 1], [0], [40])
    with pytest.raises(ValueError):
        paired_errors([0, 0], [0, 0], [0, 0], [40, 40])


def test_fixed_selection_is_class_balanced_independent_of_manifest_order():
    targets = list(range(10)) * 3
    manifest = {'val_indices': list(reversed(range(30)))}
    positions = fixed_positions(manifest, targets)
    indices = [manifest['val_indices'][p] for p in positions]
    assert indices == [i for label in range(10) for i in (label, label + 10)]


def test_sample_std_known_answer():
    mean, std = _mean_std([0.70, 0.72, 0.74])
    assert mean == pytest.approx(0.72)
    assert std == pytest.approx(0.02)
