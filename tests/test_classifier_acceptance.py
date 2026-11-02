import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.classifier_acceptance import (
    AcceptanceConfig,
    AcceptanceConfigError,
    run_classifier_acceptance,
)


IMPLEMENTATION_FILES = (
    "app/topic_taxonomy.py",
    "app/classifier_data.py",
    "app/classifier_dataset.py",
    "app/classifier_training.py",
    "app/classifier_export.py",
    "app/classifier_export_validation.py",
    "app/classifier_inference.py",
    "app/classifier_benchmark.py",
    "app/classifier_acceptance.py",
)


def test_acceptance_config_rejects_invalid_timeout(tmp_path):
    with pytest.raises(AcceptanceConfigError, match="timeout"):
        AcceptanceConfig(project_root=tmp_path, test_timeout_seconds=0)


def test_acceptance_reports_pending_external_artifacts(tmp_path):
    _create_implementation_files(tmp_path)
    output_path = tmp_path / "reports" / "acceptance.json"

    report = run_classifier_acceptance(
        AcceptanceConfig(
            project_root=tmp_path,
            output_path=output_path,
            run_tests=False,
        )
    )

    checks = {check["name"]: check for check in report["checks"]}
    assert report["status"] == "pending_external_validation"
    assert report["strict_ready"] is False
    assert checks["implementation"]["status"] == "passed"
    assert checks["prepared_dataset"]["status"] == "pending"
    assert checks["trained_model"]["status"] == "pending"
    assert checks["onnx_pytorch_parity"]["status"] == "pending"
    assert checks["automated_tests"]["status"] == "skipped"
    assert json.loads(output_path.read_text(encoding="utf-8"))["status"] == (
        "pending_external_validation"
    )


def test_acceptance_passes_complete_chain_when_tests_and_cuda_are_verified(
    tmp_path, monkeypatch
):
    _create_implementation_files(tmp_path)
    dataset_dir = tmp_path / "data" / "ch10" / "dataset"
    model_dir = tmp_path / "data" / "ch10" / "model"
    onnx_dir = tmp_path / "data" / "ch10" / "onnx"
    _create_dataset(dataset_dir)
    _create_model(model_dir)
    _create_onnx_artifacts(onnx_dir)

    monkeypatch.setattr(
        "app.classifier_acceptance.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="419 passed in 6.0s",
            stderr="",
        ),
    )
    report = run_classifier_acceptance(
        AcceptanceConfig(
            project_root=tmp_path,
            require_cuda=True,
            output_path=None,
        )
    )

    checks = {check["name"]: check for check in report["checks"]}
    assert report["status"] == "passed"
    assert report["strict_ready"] is True
    assert all(check["status"] == "passed" for check in checks.values())
    assert checks["onnx_inference_benchmark"]["active_providers"] == [
        "CUDAExecutionProvider"
    ]


def test_acceptance_keeps_cpu_benchmark_pending_for_cuda_requirement(tmp_path):
    _create_implementation_files(tmp_path)
    onnx_dir = tmp_path / "data" / "ch10" / "onnx"
    onnx_dir.mkdir(parents=True)
    benchmark_path = onnx_dir / "inference_benchmark.json"
    benchmark_path.write_text(
        json.dumps(
            {
                "status": "passed",
                "smoke_passed": True,
                "active_providers": ["CPUExecutionProvider"],
                "latency_ms": {
                    "min": 1.0,
                    "mean": 1.5,
                    "p50": 1.4,
                    "p95": 2.0,
                    "max": 2.1,
                },
                "throughput_samples_per_second": 100.0,
            }
        ),
        encoding="utf-8",
    )

    report = run_classifier_acceptance(
        AcceptanceConfig(
            project_root=tmp_path,
            require_cuda=True,
            run_tests=False,
            output_path=None,
        )
    )

    checks = {check["name"]: check for check in report["checks"]}
    assert report["status"] == "pending_external_validation"
    assert checks["onnx_inference_benchmark"]["status"] == "pending"


def _create_implementation_files(root: Path) -> None:
    for relative_path in IMPLEMENTATION_FILES:
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# test artifact\n", encoding="utf-8")


def _create_dataset(dataset_dir: Path) -> None:
    dataset_dir.mkdir(parents=True)
    records = (
        ("train", "train-1", "train question"),
        ("validation", "validation-1", "validation question"),
        ("test", "test-1", "test question"),
    )
    for split, example_id, text in records:
        payload = {
            "schema_version": 1,
            "example_id": example_id,
            "text": text,
            "labels": ["物流"],
            "source": "human_annotated",
            "split": split,
        }
        (dataset_dir / f"{split}.jsonl").write_text(
            json.dumps(payload, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    (dataset_dir / "manifest.json").write_text(
        '{"format_version": 1, "total_examples": 3}\n',
        encoding="utf-8",
    )


def _create_model(model_dir: Path) -> None:
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}\n", encoding="utf-8")
    (model_dir / "training_metrics.json").write_text(
        '{"device": "cuda"}\n', encoding="utf-8"
    )
    (model_dir / "validation_report.json").write_text(
        '{"micro_f1": 0.9}\n', encoding="utf-8"
    )
    (model_dir / "threshold.json").write_text(
        '{"threshold": 0.5}\n', encoding="utf-8"
    )
    (model_dir / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"weights")


def _create_onnx_artifacts(onnx_dir: Path) -> None:
    onnx_dir.mkdir(parents=True)
    (onnx_dir / "model.onnx").write_bytes(b"onnx")
    (onnx_dir / "export_manifest.json").write_text(
        '{"input_names": ["input_ids"]}\n', encoding="utf-8"
    )
    (onnx_dir / "threshold.json").write_text(
        '{"threshold": 0.5}\n', encoding="utf-8"
    )
    (onnx_dir / "export_validation.json").write_text(
        json.dumps(
            {
                "passed": True,
                "checked": 1,
                "logit_mismatch_rows": 0,
                "prediction_mismatch_rows": 0,
            }
        ),
        encoding="utf-8",
    )
    (onnx_dir / "inference_benchmark.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "smoke_passed": True,
                "active_providers": ["CUDAExecutionProvider"],
                "cuda_provider_active": True,
                "latency_ms": {
                    "min": 1.0,
                    "mean": 1.5,
                    "p50": 1.4,
                    "p95": 2.0,
                    "max": 2.1,
                },
                "throughput_samples_per_second": 100.0,
            }
        ),
        encoding="utf-8",
    )
