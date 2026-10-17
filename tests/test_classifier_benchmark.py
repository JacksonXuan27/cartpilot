import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from app.classifier_benchmark import (
    BenchmarkConfigError,
    InferenceBenchmarkConfig,
    run_inference_benchmark,
)
from app.topic_taxonomy import NUM_TOPIC_LABELS


def test_benchmark_config_validates_provider_and_run_options(tmp_path):
    config = InferenceBenchmarkConfig(
        onnx_dir=tmp_path,
        providers=("CPUExecutionProvider",),
        texts=("question",),
        batch_size=2,
        warmup_runs=0,
        measured_runs=1,
    )

    assert config.batch_size == 2
    assert config.measured_runs == 1

    with pytest.raises(BenchmarkConfigError, match="provider"):
        InferenceBenchmarkConfig(onnx_dir=tmp_path, providers=())
    with pytest.raises(BenchmarkConfigError, match="batch_size"):
        InferenceBenchmarkConfig(onnx_dir=tmp_path, batch_size=0)
    with pytest.raises(BenchmarkConfigError, match="require_provider"):
        InferenceBenchmarkConfig(
            onnx_dir=tmp_path,
            providers=("CPUExecutionProvider",),
            require_provider="CUDAExecutionProvider",
        )


def test_benchmark_records_active_provider_latency_and_throughput(tmp_path, monkeypatch):
    (tmp_path / "model.onnx").write_bytes(b"fake")
    (tmp_path / "threshold.json").write_text(
        '{"threshold": 0.5}\n', encoding="utf-8"
    )

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, path):
            assert path == str(tmp_path)
            return cls()

        def __call__(self, texts, **kwargs):
            assert len(texts) == 2
            assert kwargs["return_tensors"] == "np"
            return {
                "input_ids": [[1], [1]],
                "attention_mask": [[1], [1]],
            }

    class FakeSession:
        def __init__(self, path, providers):
            assert path == str(tmp_path / "model.onnx")
            assert providers == ["CPUExecutionProvider"]

        def get_inputs(self):
            return [
                SimpleNamespace(name="input_ids"),
                SimpleNamespace(name="attention_mask"),
            ]

        def get_providers(self):
            return ["CPUExecutionProvider"]

        def run(self, output_names, inputs):
            assert output_names is None
            assert set(inputs) == {"input_ids", "attention_mask"}
            row = [2.0] + [-2.0] * (NUM_TOPIC_LABELS - 1)
            return [[row, row]]

    runtime = ModuleType("onnxruntime")
    runtime.InferenceSession = FakeSession
    runtime.get_available_providers = lambda: ["CPUExecutionProvider"]
    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = FakeTokenizer
    monkeypatch.setitem(sys.modules, "onnxruntime", runtime)
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    output_path = tmp_path / "benchmark.json"
    report = run_inference_benchmark(
        InferenceBenchmarkConfig(
            onnx_dir=tmp_path,
            providers=("CUDAExecutionProvider", "CPUExecutionProvider"),
            texts=("question",),
            batch_size=2,
            warmup_runs=1,
            measured_runs=3,
            output_path=output_path,
        )
    )

    assert report["status"] == "passed"
    assert report["smoke_passed"] is True
    assert report["unavailable_requested_providers"] == ["CUDAExecutionProvider"]
    assert report["active_providers"] == ["CPUExecutionProvider"]
    assert report["cuda_provider_active"] is False
    assert report["sample_count"] == 6
    assert report["latency_ms"]["p95"] >= report["latency_ms"]["min"]
    assert report["throughput_samples_per_second"] > 0
    assert json.loads(output_path.read_text(encoding="utf-8"))["status"] == "passed"


def test_benchmark_rejects_unavailable_required_provider(tmp_path, monkeypatch):
    runtime = ModuleType("onnxruntime")
    runtime.get_available_providers = lambda: ["CPUExecutionProvider"]
    monkeypatch.setitem(sys.modules, "onnxruntime", runtime)

    with pytest.raises(BenchmarkConfigError, match="CUDAExecutionProvider"):
        run_inference_benchmark(
            InferenceBenchmarkConfig(
                onnx_dir=tmp_path,
                providers=("CUDAExecutionProvider", "CPUExecutionProvider"),
                require_provider="CUDAExecutionProvider",
            )
        )
