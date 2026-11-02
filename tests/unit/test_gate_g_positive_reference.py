"""The supervised observability reference handles decayed motor traces."""
import importlib.util
from pathlib import Path

import numpy as np

_SCRIPT = Path(__file__).resolve().parents[2] / 'scripts/gates/gate_g_task.py'
_SPEC = importlib.util.spec_from_file_location('gate_g_task', _SCRIPT)
gate_g_task = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(gate_g_task)


def test_subnormal_frozen_motor_traces_do_not_break_positive_reference():
    labels = np.tile([0, 1], 100)
    train = np.column_stack((np.full(200, 1e-45, np.float32),
                             labels.astype(np.float32),
                             np.zeros(200, np.float32)))
    train[0, 0] = np.float32(8e-45)
    test = train.copy()
    test[0, 0] = np.float32(3e-45)
    assert gate_g_task.positive_reference(train, labels, test, labels) == 1.0
