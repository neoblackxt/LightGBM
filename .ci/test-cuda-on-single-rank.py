"""Single-rank NCCLGBDT regression tests for ENABLE_MULTI_GPU=ON builds.

This is run explicitly by CUDA CI jobs with one visible GPU and num_gpu=2.
The caller checks NCCL logs to confirm communicator initialization and
collectives. These tests do not claim to validate multi-GPU communication.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]


def load_example(name):
    values = np.loadtxt(ROOT / "examples" / "regression" / f"{name}.train")
    return values[:, 1:], values[:, 0]


def train_binary(features, labels, model_path, weights=None):
    train_set = lgb.Dataset(features, label=labels, weight=weights)
    params = {
        "objective": "binary",
        "device_type": "cuda",
        "num_gpu": 2,
        "num_threads": 4,
        "num_leaves": 15,
        "min_data_in_leaf": 20,
        "min_sum_hessian_in_leaf": 0.001,
        "force_row_wise": True,
        "verbosity": 1,
    }
    booster = lgb.train(params, train_set, num_boost_round=3)
    assert booster.num_trees() == 3
    assert np.isfinite(booster.predict(features)).all()
    booster.save_model(str(model_path))
    assert model_path.stat().st_size > 0


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in {"unweighted", "weighted"}:
        raise SystemExit("usage: test-cuda-on-single-rank.py {unweighted|weighted}")
    binary_case = sys.argv[1]
    features, labels = load_example("regression")
    weight_path = ROOT / "examples" / "binary_classification" / "binary.train.weight"
    weights = np.loadtxt(weight_path) if binary_case == "weighted" else None

    with tempfile.TemporaryDirectory(prefix="lightgbm-cuda-on-single-rank-") as tmp_dir:
        model_path = Path(tmp_dir) / f"binary-{binary_case}.txt"
        train_binary(features, labels, model_path, weights=weights)


if __name__ == "__main__":
    main()
