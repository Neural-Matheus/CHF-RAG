"""Regression checks for corrections to the CHF-RAG code."""
import numpy as np
import pytest
from scipy import stats

from src.question_typing import _welch_t_p_value
from src.coupled_heatflow import build_coupled_operator
from src.heatflow_metrics import integrate_channel_flow


def test_welch_small_sample_uses_student_tail():
    a, b = np.array([1., 2., 3.]), np.array([4., 7., 12., 16.])
    p = _welch_t_p_value(a, b)
    assert p == pytest.approx(stats.ttest_ind(a, b, equal_var=False).pvalue)
    assert p > .05  # The former normal approximation falsely rejected here.


def test_asymmetric_coupling_is_rejected():
    lap = np.zeros((2, 2))
    with pytest.raises(ValueError, match="symmetric"):
        build_coupled_operator(lap, lap, lap, beta_sc=.2, beta_cs=.1)


def test_flow_includes_regularization_decay():
    lap = np.zeros((1, 1))
    op = build_coupled_operator(lap, lap, lap, tikhonov=.2)
    # Identical channels have no coupling/within-layer flow, but do decay from regularization.
    flow = integrate_channel_flow(op, np.ones(3), lap, lap, lap, ["doc"], [0], 1., n_steps=1000)
    np.testing.assert_allclose(flow[0][1:], 1 - np.exp(-.2), rtol=1e-6)
