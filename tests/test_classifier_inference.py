import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from app.classifier_inference import (
    ClassificationResult,
    InferenceConfig,
    InferenceConfigError,
    OnnxTopicClassifier,
    apply_threshold,
    probabilities_from_logits,
)
from app.topic_taxonomy import NUM_TOPIC_LABELS, TopicLabel


def test_inference_config_validates_runtime_options(tmp_path):
    config = InferenceConfig(onnx_dir=tmp_path, providers=("CPUExecutionProvider",))

    assert config.onnx_dir == tmp_path
    assert config.providers == ("CPUExecutionProvider",)
    assert config.max_length == 128
    assert config.threshold is None

    with pytest.raises(InferenceConfigError, match="provider"):
        InferenceConfig(onnx_dir=tmp_path, providers=())
    with pytest.raises(InferenceConfigError, match="max_length"):
        InferenceConfig(onnx_dir=tmp_path, max_length=1)
    with pytest.raises(InferenceConfigError, match="threshold"):
        InferenceConfig(onnx_dir=tmp_path, threshold=1.1)


def test_thresholding_keeps_all_above_threshold_and_falls_back_to_top_score():
    probabilities = [
        [0.8, 0.2] + [0.1] * (NUM_TOPIC_LABELS - 2),
        [0.1, 0.2] + [0.3] + [0.1] * (NUM_TOPIC_LABELS - 3),
    ]

    selected = apply_threshold(probabilities, threshold=0.5)

    assert selected[0][0] is True
    assert sum(selected[0]) == 1
    assert selected[1][2] is True
    assert sum(selected[1]) == 1
    assert len(selected[0]) == NUM_TOPIC_LABELS


def test_probabilities_from_logits_uses_stable_sigmoid_and_validates_shape():
    logits = [[100.0] + [-100.0] * (NUM_TOPIC_LABELS - 1)]

    probabilities = probabilities_from_logits(logits)

    assert probabilities[0][0] == pytest.approx(1.0)
    assert probabilities[0][1] == pytest.approx(0.0)
    with pytest.raises(InferenceConfigError, match="topic scores"):
        probabilities_from_logits([[0.0]])


def test_onnx_classifier_batches_text_and_returns_topic_predictions(tmp_path, monkeypatch):
    (tmp_path / "model.onnx").write_bytes(b"fake")
    (tmp_path / "threshold.json").write_text('{"threshold": 0.5}\n', encoding="utf-8")
    calls = SimpleNamespace(providers=None, texts=None, runtime_inputs=None)

    class FakeArray:
        def __init__(self, values):
            self.values = values

        def tolist(self):
            return self.values

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, path):
            assert path == str(tmp_path)
            return cls()

        def __call__(self, texts, **kwargs):
            calls.texts = texts
            assert kwargs == {
                "truncation": True,
                "padding": True,
                "max_length": 128,
                "return_tensors": "np",
            }
            return {
                "input_ids": FakeArray([[1] for _ in texts]),
                "attention_mask": FakeArray([[1] for _ in texts]),
                "token_type_ids": FakeArray([[0] for _ in texts]),
            }

    class FakeSession:
        def __init__(self, path, providers):
            assert path == str(tmp_path / "model.onnx")
            calls.providers = providers

        def get_inputs(self):
            return [
                SimpleNamespace(name="input_ids"),
                SimpleNamespace(name="attention_mask"),
                SimpleNamespace(name="token_type_ids"),
            ]

        def run(self, output_names, inputs):
            assert output_names is None
            calls.runtime_inputs = inputs
            return [
                [
                    [2.0, -2.0] + [-2.0] * (NUM_TOPIC_LABELS - 2),
                    [-2.0, 2.0] + [-2.0] * (NUM_TOPIC_LABELS - 2),
                ]
            ]

    runtime = ModuleType("onnxruntime")
    runtime.InferenceSession = FakeSession
    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = FakeTokenizer
    monkeypatch.setitem(sys.modules, "onnxruntime", runtime)
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    classifier = OnnxTopicClassifier(
        InferenceConfig(
            onnx_dir=tmp_path,
            providers=("CUDAExecutionProvider", "CPUExecutionProvider"),
        )
    )
    results = classifier.predict(("return request", "shipping status"))

    assert calls.providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert calls.texts == ["return request", "shipping status"]
    assert set(calls.runtime_inputs) == {
        "input_ids",
        "attention_mask",
        "token_type_ids",
    }
    assert all(isinstance(result, ClassificationResult) for result in results)
    assert results[0].predictions[0].label is TopicLabel.RETURN_EXCHANGE
    assert results[1].predictions[0].label is TopicLabel.LOGISTICS
    assert results[0].to_dict()["text"] == "return request"
    assert results[0].to_dict()["predictions"][0]["label"] == "退换货"


def test_onnx_classifier_uses_explicit_threshold_and_rejects_blank_text(tmp_path, monkeypatch):
    (tmp_path / "model.onnx").write_bytes(b"fake")
    (tmp_path / "threshold.json").write_text('{"threshold": 0.2}\n', encoding="utf-8")

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, path):
            return cls()

        def __call__(self, texts, **kwargs):
            return {
                "input_ids": [[1] for _ in texts],
                "attention_mask": [[1] for _ in texts],
            }

    class FakeSession:
        def __init__(self, path, providers):
            pass

        def get_inputs(self):
            return [SimpleNamespace(name="input_ids"), SimpleNamespace(name="attention_mask")]

        def run(self, output_names, inputs):
            return [[[0.0] * NUM_TOPIC_LABELS]]

    runtime = ModuleType("onnxruntime")
    runtime.InferenceSession = FakeSession
    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = FakeTokenizer
    monkeypatch.setitem(sys.modules, "onnxruntime", runtime)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    classifier = OnnxTopicClassifier(InferenceConfig(onnx_dir=tmp_path, threshold=0.8))

    with pytest.raises(InferenceConfigError, match="text"):
        classifier.classify(" ")
    assert classifier.classify("question").predictions[0].probability == pytest.approx(0.5)


def test_onnx_classifier_rejects_missing_runtime_input(tmp_path, monkeypatch):
    (tmp_path / "model.onnx").write_bytes(b"fake")
    (tmp_path / "threshold.json").write_text('{"threshold": 0.5}\n', encoding="utf-8")

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, path):
            return cls()

        def __call__(self, texts, **kwargs):
            return {"input_ids": [[1] for _ in texts]}

    class FakeSession:
        def __init__(self, path, providers):
            pass

        def get_inputs(self):
            return [SimpleNamespace(name="input_ids"), SimpleNamespace(name="attention_mask")]

    runtime = ModuleType("onnxruntime")
    runtime.InferenceSession = FakeSession
    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = FakeTokenizer
    monkeypatch.setitem(sys.modules, "onnxruntime", runtime)
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    classifier = OnnxTopicClassifier(InferenceConfig(onnx_dir=tmp_path))
    with pytest.raises(InferenceConfigError, match="attention_mask"):
        classifier.classify("question")
