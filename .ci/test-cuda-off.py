"""Regression tests for CUDA builds with ENABLE_MULTI_GPU=OFF.

This is run explicitly by CUDA CI jobs with ENABLE_MULTI_GPU=OFF. It is not
part of the default pytest discovery because these tests require a CUDA GPU.
CI sets CUDA_VISIBLE_DEVICES before Python imports LightGBM.
"""

import tempfile
from pathlib import Path

import numpy as np

import lightgbm as lgb
from lightgbm.basic import LightGBMError

EXPECTED_ERROR = "Multi-GPU training is not available in this LightGBM build."
ROOT = Path(__file__).resolve().parents[1]


def load_example(name):
    values = np.loadtxt(ROOT / "examples" / "regression" / f"{name}.train")
    return values[:, 1:], values[:, 0]


def dataset(features, labels, weights=None):
    return lgb.Dataset(features, label=labels, weight=weights)


def params(objective="regression", num_gpu=1, boosting="gbdt"):
    return {
        "objective": objective,
        "boosting": boosting,
        "device_type": "cuda",
        "num_gpu": num_gpu,
        "num_threads": 4,
        "num_leaves": 15,
        "min_data_in_leaf": 20,
        "min_sum_hessian_in_leaf": 0.001,
        "force_row_wise": True,
        "verbosity": 1,
    }


def train_and_save(train_set, model_path, extra_params=None):
    model_params = params()
    if extra_params:
        model_params.update(extra_params)
    booster = lgb.train(model_params, train_set, num_boost_round=3)
    booster.save_model(str(model_path))
    assert model_path.stat().st_size > 0
    return booster


def expect_multi_gpu_error(train_set, init_model=None):
    try:
        lgb.train(params(num_gpu=2), train_set, num_boost_round=3, init_model=init_model)
    except LightGBMError as exc:
        error_message = str(exc)
    else:
        raise AssertionError("num_gpu=2 unexpectedly succeeded in a single-GPU-only build")
    assert EXPECTED_ERROR in error_message


def main():
    features, labels = load_example("regression")
    weight_path = ROOT / "examples" / "binary_classification" / "binary.train.weight"
    weights = np.loadtxt(weight_path)

    def new_dataset(sample_weights=None):
        return dataset(features, labels, sample_weights)

    with tempfile.TemporaryDirectory(prefix="lightgbm-cuda-off-") as tmp_dir:
        tmp = Path(tmp_dir)
        regression_model = tmp / "regression.txt"
        train_and_save(new_dataset(), regression_model)

        train_and_save(new_dataset(), tmp / "binary-unweighted.txt", {"objective": "binary"})
        train_and_save(new_dataset(weights), tmp / "binary-weighted.txt", {"objective": "binary"})
        train_and_save(new_dataset(), tmp / "quantized.txt", {"use_quantized_grad": True})

        expect_multi_gpu_error(new_dataset())
        expect_multi_gpu_error(new_dataset(), init_model=str(regression_model))

        # Model loading remains available when the request does not ask for multi-GPU.
        loaded = lgb.Booster(model_file=str(regression_model), params=params())
        predictions = loaded.predict(features)
        assert predictions.shape == (features.shape[0],)

        # These boosting types do not select NCCLGBDT through their direct factory branches.
        train_and_save(new_dataset(), tmp / "dart.txt", {"boosting": "dart", "num_gpu": 2})
        train_and_save(
            new_dataset(),
            tmp / "rf.txt",
            {"boosting": "rf", "num_gpu": 2, "bagging_fraction": 0.8, "bagging_freq": 1},
        )

        # The public GOSS alias is normalized to GBDT plus GOSS sampling before factory dispatch.
        try:
            lgb.train(params(num_gpu=2, boosting="goss"), new_dataset(), num_boost_round=3)
        except LightGBMError as exc:
            error_message = str(exc)
        else:
            raise AssertionError("GOSS alias with num_gpu=2 did not follow the GBDT rejection path")
        assert EXPECTED_ERROR in error_message


if __name__ == "__main__":
    main()
