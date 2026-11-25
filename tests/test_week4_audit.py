import pytest
from scripts.reproduce_week4 import check_history, mean_std


def known_run():
    history = [dict(epoch='1', global_step='36', train_accuracy=str(1000/4500),
                    val_accuracy=str(2608/5000), train_loss='2', val_loss='2',
                    train_seconds='1', epoch_seconds='2', lr='0.0003', peak_cuda_mb='300')]
    row = dict(ratio='0.10', best_epoch='1', total='5000', accuracy=str(2608/5000),
               correct='2608', split='val', global_step='36', train_seconds_total='1')
    return history, row


def test_accuracy_must_correspond_to_integer_number_of_images():
    history, row = known_run()
    assert check_history(history, row, epochs=1) == []
    history[0]['val_accuracy'] = '0.2647'
    assert 'epoch_1:val_accuracy_noninteger_correct' in check_history(history, row, epochs=1)


def test_negative_time_is_rejected():
    history, row = known_run()
    history[0]['train_seconds'] = '-0.09'
    assert 'epoch_1:train_seconds_invalid' in check_history(history, row, epochs=1)


def test_standard_deviation_requires_repeated_seeds():
    assert mean_std([0.7, 0.72, 0.74]) == pytest.approx((0.72, 0.02))
    import math
    assert math.isnan(mean_std([0.7])[1])
