import pytest

from app.classifier_training import (
    TrainingConfigError,
    TrainingDevicePreference,
    TrainingRunConfig,
    select_training_device,
)


def test_training_config_exposes_reproducible_model_and_optimization_defaults(tmp_path):
    config = TrainingRunConfig(
        train_path=tmp_path / "train.jsonl",
        validation_path=tmp_path / "validation.jsonl",
        output_dir=tmp_path / "model",
    )

    assert config.model_name_or_path == "hfl/chinese-roberta-wwm-ext"
    assert config.epochs == 3
    assert config.batch_size == 16
    assert config.learning_rate == 2e-5
    assert config.max_length == 128
    assert config.seed == 42
    assert config.device is TrainingDevicePreference.AUTO
    assert config.train_path == tmp_path / "train.jsonl"


def test_training_config_accepts_explicit_cpu_and_custom_hyperparameters(tmp_path):
    config = TrainingRunConfig(
        train_path="train.jsonl",
        validation_path="validation.jsonl",
        output_dir="model",
        model_name_or_path="local/checkpoint",
        epochs=5,
        batch_size=8,
        learning_rate=1e-5,
        max_length=256,
        seed=7,
        device="cpu",
    )

    assert config.model_name_or_path == "local/checkpoint"
    assert config.epochs == 5
    assert config.batch_size == 8
    assert config.max_length == 256
    assert config.seed == 7
    assert config.device is TrainingDevicePreference.CPU


@pytest.mark.parametrize(
    "overrides",
    [
        {"epochs": 0},
        {"batch_size": 0},
        {"learning_rate": 0},
        {"learning_rate": float("nan")},
        {"max_length": 1},
        {"seed": -1},
        {"model_name_or_path": "  "},
        {"device": "tpu"},
    ],
)
def test_training_config_rejects_invalid_settings(tmp_path, overrides):
    values = {
        "train_path": tmp_path / "train.jsonl",
        "validation_path": tmp_path / "validation.jsonl",
        "output_dir": tmp_path / "model",
        **overrides,
    }

    with pytest.raises(TrainingConfigError):
        TrainingRunConfig(**values)


@pytest.mark.parametrize(
    ("preference", "cuda_available", "expected"),
    [
        ("auto", True, "cuda"),
        ("auto", False, "cpu"),
        ("cpu", True, "cpu"),
        ("cuda", True, "cuda"),
    ],
)
def test_training_device_selection_respects_preference_and_availability(
    preference, cuda_available, expected
):
    assert select_training_device(preference, cuda_available) == expected


def test_explicit_cuda_selection_fails_when_no_compatible_device_is_available():
    with pytest.raises(TrainingConfigError, match="CUDA was requested"):
        select_training_device("cuda", False)


def test_device_selection_rejects_unknown_preferences():
    with pytest.raises(TrainingConfigError, match="unsupported device preference"):
        select_training_device("tpu", True)
