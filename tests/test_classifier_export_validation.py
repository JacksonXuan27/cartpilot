import json
import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace

import pytest

from app.classifier_data import TrainingDataSource, TrainingDataSplit, TrainingExample
from app.classifier_export_validation import (
    ExportValidationConfig,
    ExportValidationError,
    compare_logits,
    validate_onnx_parity,
)
from app.topic_taxonomy import NUM_TOPIC_LABELS, TopicLabel


def _write_test_set(path, records=2):
    examples = [
        TrainingExample(
            example_id=f"sample-{index}",
            text=f"test message {index}",
            labels=(TopicLabel.RETURN_EXCHANGE,),
            source=TrainingDataSource.HUMAN_ANNOTATED,
            split=TrainingDataSplit.TEST,
        )
        for index in range(records)
    ]
    path.write_text(
        "\n".join(example.model_dump_json() for example in examples) + "\n",
        encoding="utf-8",
    )


def test_compare_logits_reports_close_values_and_thresholded_predictions():
    reference = [[2.0] + [-2.0] * (NUM_TOPIC_LABELS - 1)]
    exported = [[2.00001] + [-2.0] * (NUM_TOPIC_LABELS - 1)]

    report = compare_logits(reference, exported, threshold=0.5, atol=1e-3)

    assert report == {
        "checked": 1,
        "logit_mismatch_rows": 0,
        "prediction_mismatch_rows": 0,
        "max_absolute_logit_difference": pytest.approx(1e-5),
        "passed": True,
    }


def test_compare_logits_detects_logit_and_prediction_drift():
    reference = [[-2.0] * NUM_TOPIC_LABELS]
    exported = [[-2.0] * NUM_TOPIC_LABELS]
    exported[0][3] = 1.0

    report = compare_logits(reference, exported, threshold=0.5, atol=1e-3)

    assert report["logit_mismatch_rows"] == 1
    assert report["prediction_mismatch_rows"] == 1
    assert report["passed"] is False


@pytest.mark.parametrize(
    ("reference", "exported", "threshold", "atol"),
    [
        ([], [], 0.5, 1e-3),
        ([[0.0] * (NUM_TOPIC_LABELS - 1)], [[0.0] * (NUM_TOPIC_LABELS - 1)], 0.5, 1e-3),
        ([[0.0] * NUM_TOPIC_LABELS], [], 0.5, 1e-3),
        ([[0.0] * NUM_TOPIC_LABELS], [[0.0] * NUM_TOPIC_LABELS], -0.1, 1e-3),
        ([[0.0] * NUM_TOPIC_LABELS], [[0.0] * NUM_TOPIC_LABELS], 0.5, -1.0),
    ],
)
def test_compare_logits_rejects_invalid_inputs(reference, exported, threshold, atol):
    with pytest.raises(ExportValidationError):
        compare_logits(reference, exported, threshold=threshold, atol=atol)


def test_parity_validation_batches_test_examples_and_writes_report(tmp_path, monkeypatch):
    model_dir = tmp_path / "model"
    onnx_dir = tmp_path / "onnx"
    model_dir.mkdir()
    onnx_dir.mkdir()
    (onnx_dir / "model.onnx").write_bytes(b"onnx")
    (onnx_dir / "threshold.json").write_text('{"threshold": 0.5}', encoding="utf-8")
    test_path = tmp_path / "test.jsonl"
    _write_test_set(test_path, records=3)

    class Tensor:
        def __init__(self, values):
            self.values = values

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.values

        def tolist(self):
            return self.values

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, path):
            assert path == str(model_dir)
            return cls()

        def __call__(self, texts, **kwargs):
            assert kwargs["return_tensors"] == "pt"
            return {
                "input_ids": Tensor([[index] for index, _ in enumerate(texts)]),
                "attention_mask": Tensor([[1] for _ in texts]),
                "token_type_ids": Tensor([[0] for _ in texts]),
            }

    class FakeModel:
        @classmethod
        def from_pretrained(cls, path):
            assert path == str(model_dir)
            return cls()

        def to(self, device):
            assert device == "cpu"
            return self

        def eval(self):
            return self

        def __call__(self, **inputs):
            batch = len(inputs["input_ids"].values)
            logits = [[2.0] + [-2.0] * (NUM_TOPIC_LABELS - 1) for _ in range(batch)]
            return SimpleNamespace(logits=Tensor(logits))

    class FakeSession:
        def __init__(self, model_path, providers):
            assert model_path == str(onnx_dir / "model.onnx")
            assert providers == ["CPUExecutionProvider"]

        def get_inputs(self):
            return [
                SimpleNamespace(name="input_ids"),
                SimpleNamespace(name="attention_mask"),
                SimpleNamespace(name="token_type_ids"),
            ]

        def run(self, output_names, inputs):
            batch = len(inputs["input_ids"])
            return [[[2.0] + [-2.0] * (NUM_TOPIC_LABELS - 1) for _ in range(batch)]]

    torch = ModuleType("torch")
    torch.no_grad = nullcontext
    transformers = ModuleType("transformers")
    transformers.AutoModelForSequenceClassification = FakeModel
    transformers.AutoTokenizer = FakeTokenizer
    runtime = ModuleType("onnxruntime")
    runtime.InferenceSession = FakeSession
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "onnxruntime", runtime)

    report = validate_onnx_parity(
        ExportValidationConfig(
            model_dir=model_dir,
            onnx_dir=onnx_dir,
            test_path=test_path,
            batch_size=2,
        )
    )

    assert report["checked"] == 3
    assert report["passed"] is True
    assert (onnx_dir / "export_validation.json").is_file()
    assert json.loads((onnx_dir / "export_validation.json").read_text()) == report


def test_parity_validation_rejects_non_test_data(tmp_path):
    test_path = tmp_path / "validation.jsonl"
    example = TrainingExample(
        example_id="validation-1",
        text="not test split",
        labels=(TopicLabel.RETURN_EXCHANGE,),
        source=TrainingDataSource.HUMAN_ANNOTATED,
        split=TrainingDataSplit.VALIDATION,
    )
    test_path.write_text(example.model_dump_json() + "\n", encoding="utf-8")

    with pytest.raises(ExportValidationError, match="test split"):
        validate_onnx_parity(
            ExportValidationConfig(
                model_dir=tmp_path / "model",
                onnx_dir=tmp_path / "onnx",
                test_path=test_path,
            )
        )
