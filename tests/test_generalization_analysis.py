"""Regression checks for CPU diagnostic math; do not alter accepted inference."""
import importlib.util
from pathlib import Path
import numpy as np
from upar.metrics import calibrate_thresholds, score_one_domain

spec = importlib.util.spec_from_file_location('diagnostic_analysis', Path(__file__).parents[1] / 'scripts/generalization_analysis.py')
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def test_fast_calibration_matches_original_with_missing_support():
    rng = np.random.default_rng(2027)
    p = rng.random((21, 40), dtype=np.float32)
    y = rng.integers(0, 2, size=(21, 40)).astype(np.float32)
    valid = rng.random((21, 40)) > .1
    valid[:, 3] = False
    y[:, 4] = 0
    domains = np.array(['a'] * 10 + ['b'] * 11)
    expected = calibrate_thresholds(p, y, valid, domains.tolist())
    assert np.array_equal(expected, analysis.fit_fast(p, y, valid, domains))


def test_mean_image_f1_differs_from_harmonic_of_mean_precision_recall():
    # One image has precision=1, recall=1/2; the other precision=1/2, recall=1.
    y = np.zeros((2, 40), dtype=np.float32); p = y.copy(); valid = np.ones_like(y, bool)
    y[0, :2] = 1; y[1, 0] = 1
    p[0, 0] = 1; p[1, :2] = 1
    legacy = score_one_domain(p, y, valid, np.full(40, .5))
    assert np.isclose(legacy['Inst_F1'], .75)
    tested = analysis.sensitivity(p, y, valid, np.array(['a', 'a']), np.full(40, .5))
    assert np.isclose(tested['mean_image_f1_sensitivity']['macro']['Inst_F1'], 2/3)
