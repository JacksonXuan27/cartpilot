import pytest
import json
import sys
from types import ModuleType, SimpleNamespace

from app.classifier_export import ExportConfig, ExportConfigError, export_onnx_model


def test_export_config_rejects_invalid_paths_opset_and_empty_examples(tmp_path):
    with pytest.raises(ExportConfigError, match="model_dir"):
        ExportConfig(model_dir="", output_dir=tmp_path / "onnx")
    with pytest.raises(ExportConfigError, match="output_dir"):
        ExportConfig(model_dir=tmp_path / "model", output_dir="")
    with pytest.raises(ExportConfigError, match="opset"):
        ExportConfig(model_dir=tmp_path / "model", output_dir=tmp_path / "onnx", opset=0)
    with pytest.raises(ExportConfigError, match="sample_texts"):
        ExportConfig(model_dir=tmp_path / "model", output_dir=tmp_path / "onnx", sample_texts=())


def test_export_saves_dynamic_model_tokenizer_threshold_and_manifest(tmp_path, monkeypatch):
    model_dir = tmp_path / "model"
    output_dir = tmp_path / "onnx"
    model_dir.mkdir()
    (model_dir / "threshold.json").write_text('{"threshold": 0.45}\n', encoding="utf-8")
    calls = SimpleNamespace(export=None)

    class FakeModule:
        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

        def eval(self):
            return self

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, path):
            calls.tokenizer_path = path
            return cls()

        def __call__(self, texts, **kwargs):
            assert texts == ["return request", "shipping status"]
            assert kwargs["return_tensors"] == "pt"
            return {"input_ids": "ids", "attention_mask": "mask", "token_type_ids": "types"}

        def save_pretrained(self, path):
            (path / "tokenizer.json").write_text("{}", encoding="utf-8")

    class FakeModel(FakeModule):
        @classmethod
        def from_pretrained(cls, path):
            calls.model_path = path
            return cls()

        def __call__(self, **kwargs):
            return SimpleNamespace(logits="logits")

    def fake_export(wrapper, example_inputs, output_path, **kwargs):
        calls.export = {"inputs": example_inputs, **kwargs}
        wrapper(*example_inputs)
        with open(output_path, "wb") as output:
            output.write(b"fake-onnx")

    torch = ModuleType("torch")
    torch.nn = SimpleNamespace(Module=FakeModule)
    torch.onnx = SimpleNamespace(export=fake_export)
    transformers = ModuleType("transformers")
    transformers.AutoModelForSequenceClassification = FakeModel
    transformers.AutoTokenizer = FakeTokenizer
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    result = export_onnx_model(ExportConfig(
        model_dir=model_dir,
        output_dir=output_dir,
        sample_texts=("return request", "shipping status"),
    ))

    assert calls.model_path == calls.tokenizer_path == str(model_dir)
    assert calls.export["inputs"] == ("ids", "mask", "types")
    assert calls.export["input_names"] == ["input_ids", "attention_mask", "token_type_ids"]
    assert calls.export["dynamic_axes"] == {
        "input_ids": {0: "batch", 1: "sequence"},
        "attention_mask": {0: "batch", 1: "sequence"},
        "token_type_ids": {0: "batch", 1: "sequence"},
        "logits": {0: "batch"},
    }
    assert calls.export["dynamo"] is False
    assert calls.export["opset_version"] == 17
    assert result["onnx_path"] == str(output_dir / "model.onnx")
    assert (output_dir / "model.onnx").read_bytes() == b"fake-onnx"
    assert (output_dir / "tokenizer.json").is_file()
    assert (output_dir / "threshold.json").read_text(encoding="utf-8") == '{"threshold": 0.45}\n'
    manifest = json.loads((output_dir / "export_manifest.json").read_text(encoding="utf-8"))
    assert manifest["sample_count"] == 2
    assert manifest["onnx_bytes"] == len(b"fake-onnx")
