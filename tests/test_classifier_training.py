import pytest
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

from app.classifier_training import (
    compute_micro_f1,
    run_training,
    TrainingConfigError,
    TrainingDevicePreference,
    TrainingRunConfig,
    select_training_device,
)
from app.topic_taxonomy import LABEL2ID, NUM_TOPIC_LABELS


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


def test_compute_micro_f1_handles_multilabel_logits_and_tuple_predictions():
    logits = [[-1.0] * NUM_TOPIC_LABELS for _ in range(2)]
    labels = [[0.0] * NUM_TOPIC_LABELS for _ in range(2)]
    logits[0][0] = 2.0
    logits[0][2] = 1.0
    labels[0][0] = 1.0
    labels[0][2] = 1.0
    logits[1][0] = 3.0
    labels[1][1] = 1.0

    assert compute_micro_f1(((logits, "auxiliary"), labels)) == pytest.approx(2 / 3)


def test_run_training_configures_multilabel_roberta_and_saves_artifacts(
    tmp_path, monkeypatch
):
    calls = SimpleNamespace(
        model_load=None,
        training_arguments=None,
        trainer_arguments=None,
        tokenizer_load=None,
        seeded=None,
    )

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, model_name):
            calls.tokenizer_load = model_name
            return cls()

        def save_pretrained(self, output_dir):
            (tmp_path / "model" / "tokenizer.json").write_text("{}", encoding="utf-8")

    class FakeModel:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            calls.model_load = (model_name, kwargs)
            return cls()

    class FakeTrainingArguments:
        def __init__(self, **kwargs):
            calls.training_arguments = kwargs

    class FakeTrainer:
        def __init__(self, **kwargs):
            calls.trainer_arguments = kwargs

        def train(self):
            return SimpleNamespace(metrics={"train_loss": 0.25})

        def evaluate(self):
            return {"eval_micro_f1": 0.8}

        def save_model(self, output_dir):
            from pathlib import Path

            model_dir = Path(output_dir)
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "model.bin").write_bytes(b"fake model")

    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(is_available=Mock(return_value=True))
    transformers = ModuleType("transformers")
    transformers.AutoModelForSequenceClassification = FakeModel
    transformers.AutoTokenizer = FakeTokenizer
    transformers.Trainer = FakeTrainer
    transformers.TrainingArguments = FakeTrainingArguments
    transformers.default_data_collator = object()
    transformers.set_seed = lambda seed: setattr(calls, "seeded", seed)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    import app.classifier_training as training

    splits = SimpleNamespace(train=("train-example",), validation=("validation-example",))
    monkeypatch.setattr(training, "load_train_validation_splits", lambda *_: splits)
    monkeypatch.setattr(
        training,
        "tokenize_examples",
        lambda examples, tokenizer, *, max_length: [
            {"examples": examples, "max_length": max_length}
        ],
    )
    config = TrainingRunConfig(
        train_path=tmp_path / "train.jsonl",
        validation_path=tmp_path / "validation.jsonl",
        output_dir=tmp_path / "model",
        device="cuda",
    )

    result = run_training(config)

    model_name, model_config = calls.model_load
    assert model_name == config.model_name_or_path
    assert model_config["problem_type"] == "multi_label_classification"
    assert model_config["num_labels"] == NUM_TOPIC_LABELS
    assert model_config["label2id"] == {label.value: index for label, index in LABEL2ID.items()}
    assert model_config["id2label"] == {
        index: label.value for label, index in LABEL2ID.items()
    }
    assert calls.training_arguments["use_cpu"] is False
    assert calls.training_arguments["save_strategy"] == "no"
    assert calls.training_arguments["seed"] == config.seed
    assert calls.seeded == config.seed
    assert calls.trainer_arguments["train_dataset"][0]["max_length"] == config.max_length
    assert calls.trainer_arguments["eval_dataset"][0]["examples"] == splits.validation
    assert result["device"] == "cuda"
    assert result["validation_metrics"] == {"eval_micro_f1": 0.8}
    assert (config.output_dir / "model.bin").is_file()
    assert (config.output_dir / "tokenizer.json").is_file()
    assert (config.output_dir / "training_metrics.json").is_file()
