import numpy as np

from utils import TIME_STEPS, get_t_span


def test_get_t_span_includes_exact_non_aligned_endpoint_across_frames():
    frame_dt = TIME_STEPS['DT_ANIM']
    integration_dt = TIME_STEPS['DT_INT']
    t_start = 0.0

    for _ in range(5000):
        t_stop = t_start + frame_dt
        t_span = get_t_span(t_start, t_stop, integration_dt)

        assert t_span[-1] == t_stop
        assert len(t_span) - 1 == 17
        assert t_span[-1] - t_span[-2] < integration_dt
        t_start = t_stop


def test_get_t_span_keeps_aligned_steps():
    t_span = get_t_span(0.0, 0.016, 0.001)

    assert t_span[-1] == 0.016
    assert len(t_span) == 17
    assert np.allclose(np.diff(t_span), 0.001)
