import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from app.classifier_training import (
    build_multilabel_report,
    compute_classification_metrics,
    compute_micro_f1,
    run_training,
    TrainingConfigError,
    TrainingDevicePreference,
    TrainingRunConfig,
    main,
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


def test_training_cli_maps_arguments_to_config_and_prints_summary(tmp_path, monkeypatch, capsys):
    captured = {}

    def fake_run_training(config):
        captured["config"] = config
        return {"device": "cuda", "validation_metrics": {"validation_micro_f1": 0.9}}

    monkeypatch.setattr("app.classifier_training.run_training", fake_run_training)
    main(
        [
            "--train",
            str(tmp_path / "train.jsonl"),
            "--validation",
            str(tmp_path / "validation.jsonl"),
            "--output-dir",
            str(tmp_path / "model"),
            "--model",
            "local/roberta",
            "--epochs",
            "5",
            "--batch-size",
            "8",
            "--learning-rate",
            "0.00001",
            "--max-length",
            "256",
            "--seed",
            "7",
            "--device",
            "cuda",
        ]
    )

    config = captured["config"]
    assert config.train_path == tmp_path / "train.jsonl"
    assert config.validation_path == tmp_path / "validation.jsonl"
    assert config.output_dir == tmp_path / "model"
    assert config.model_name_or_path == "local/roberta"
    assert config.epochs == 5
    assert config.batch_size == 8
    assert config.learning_rate == pytest.approx(1e-5)
    assert config.max_length == 256
    assert config.seed == 7
    assert config.device is TrainingDevicePreference.CUDA
    assert json.loads(capsys.readouterr().out)["device"] == "cuda"


def test_training_cli_uses_project_defaults(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.chdir(tmp_path)

    def fake_run_training(config):
        captured["config"] = config
        return {}

    monkeypatch.setattr(
        "app.classifier_training.run_training",
        fake_run_training,
    )

    main([])

    assert captured["config"].train_path == Path("data/ch10/dataset/train.jsonl")
    assert captured["config"].validation_path == Path("data/ch10/dataset/validation.jsonl")
    assert captured["config"].output_dir == Path("data/ch10/model")
    assert captured["config"].device is TrainingDevicePreference.AUTO


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
    assert compute_classification_metrics(((logits, "auxiliary"), labels)) == {
        "micro_f1": pytest.approx(2 / 3),
        "macro_f1": pytest.approx(5 / (3 * NUM_TOPIC_LABELS)),
    }


def test_multilabel_report_includes_per_topic_f1_and_binary_confusion_counts():
    logits = [[-2.0] * NUM_TOPIC_LABELS for _ in range(2)]
    labels = [[0.0] * NUM_TOPIC_LABELS for _ in range(2)]
    logistics_id = LABEL2ID[next(label for label in LABEL2ID if label.value == "物流")]
    size_id = LABEL2ID[next(label for label in LABEL2ID if label.value == "尺码")]
    logits[0][logistics_id] = 2.0
    logits[0][size_id] = 2.0
    labels[0][logistics_id] = 1.0
    labels[1][size_id] = 1.0

    report = build_multilabel_report(logits, labels)

    assert report["micro_f1"] == pytest.approx(0.5)
    assert report["macro_f1"] == pytest.approx(1 / NUM_TOPIC_LABELS)
    assert report["per_class"]["物流"] == {
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "support": 1,
        "confusion_matrix": [[1, 0], [0, 1]],
    }
    assert report["per_class"]["尺码"]["confusion_matrix"] == [
        [0, 1],
        [1, 0],
    ]


@pytest.mark.parametrize(
    ("logits", "labels", "threshold"),
    [
        ([[0.0]], [[0.0]], 0.5),
        ([[0.0] * NUM_TOPIC_LABELS], [], 0.5),
        ([[0.0] * NUM_TOPIC_LABELS], [[0.0] * (NUM_TOPIC_LABELS - 1)], 0.5),
        ([[0.0] * NUM_TOPIC_LABELS], [[0.0] * NUM_TOPIC_LABELS], 1.1),
    ],
)
def test_multilabel_report_rejects_inconsistent_dimensions_and_threshold(
    logits, labels, threshold
):
    with pytest.raises(ValueError):
        build_multilabel_report(logits, labels, threshold=threshold)


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

        def predict(self, dataset, metric_key_prefix):
            logits = [[-2.0] * NUM_TOPIC_LABELS]
            labels = [[0.0] * NUM_TOPIC_LABELS]
            logits[0][0] = 2.0
            labels[0][0] = 1.0
            return SimpleNamespace(
                predictions=logits,
                label_ids=labels,
                metrics={"validation_micro_f1": 1.0, "validation_macro_f1": 1.0},
            )

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
    assert calls.trainer_arguments["compute_metrics"] is compute_classification_metrics
    assert result["device"] == "cuda"
    assert result["validation_metrics"] == {
        "validation_micro_f1": 1.0,
        "validation_macro_f1": 1.0,
    }
    report_path = config.output_dir / "validation_report.json"
    assert report_path.is_file()
    assert json.loads(report_path.read_text(encoding="utf-8"))["per_class"]["退换货"][
        "f1"
    ] == 1.0
    assert json.loads(
        (config.output_dir / "threshold.json").read_text(encoding="utf-8")
    ) == {"threshold": 0.5}
    assert (config.output_dir / "model.bin").is_file()
    assert (config.output_dir / "tokenizer.json").is_file()
    assert (config.output_dir / "training_metrics.json").is_file()
