from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from app.classifier_data import (
    ClassifierDataError,
    TrainingDataSplit,
    load_training_dataset,
)


class AcceptanceConfigError(ValueError):
    pass


class CheckStatus(str, Enum):
    PASSED = "passed"
    PENDING = "pending"
    FAILED = "failed"
    SKIPPED = "skipped"


class OverallStatus(str, Enum):
    PASSED = "passed"
    PENDING = "pending_external_validation"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class AcceptanceConfig:
    project_root: Path = Path(".")
    dataset_dir: Path = Path("data/ch10/dataset")
    model_dir: Path = Path("data/ch10/model")
    onnx_dir: Path = Path("data/ch10/onnx")
    parity_report: Path | None = None
    benchmark_report: Path | None = None
    output_path: Path | None = Path("data/ch10/acceptance_report.json")
    run_tests: bool = True
    require_cuda: bool = False
    test_timeout_seconds: int = 900

    def __post_init__(self) -> None:
        root = Path(self.project_root).resolve()
        if not root.is_dir():
            raise AcceptanceConfigError(f"project_root does not exist: {root}")
        object.__setattr__(self, "project_root", root)
        for field_name in ("dataset_dir", "model_dir", "onnx_dir"):
            value = _resolve_path(root, getattr(self, field_name))
            object.__setattr__(self, field_name, value)
        if self.parity_report is None:
            object.__setattr__(self, "parity_report", self.onnx_dir / "export_validation.json")
        else:
            object.__setattr__(self, "parity_report", _resolve_path(root, self.parity_report))
        if self.benchmark_report is None:
            object.__setattr__(
                self,
                "benchmark_report",
                self.onnx_dir / "inference_benchmark.json",
            )
        else:
            object.__setattr__(
                self,
                "benchmark_report",
                _resolve_path(root, self.benchmark_report),
            )
        if self.output_path is not None:
            object.__setattr__(self, "output_path", _resolve_path(root, self.output_path))
        if self.test_timeout_seconds < 1:
            raise AcceptanceConfigError("test_timeout_seconds must be positive")


def run_classifier_acceptance(config: AcceptanceConfig) -> dict[str, Any]:
    checks = [
        _check_implementation(config.project_root),
        _check_dataset(config.dataset_dir),
        _check_model(config.model_dir),
        _check_onnx_export(config.onnx_dir),
        _check_parity(config.parity_report),
        _check_benchmark(config.benchmark_report, require_cuda=config.require_cuda),
    ]
    if config.run_tests:
        checks.append(_run_tests(config.project_root, config.test_timeout_seconds))
    else:
        checks.append(
            _stage(
                "automated_tests",
                CheckStatus.SKIPPED,
                "未运行测试；使用 --run-tests 或省略 --skip-tests 执行全量测试",
            )
        )

    statuses = {check["status"] for check in checks}
    if CheckStatus.FAILED.value in statuses:
        overall = OverallStatus.FAILED
    elif statuses & {CheckStatus.PENDING.value, CheckStatus.SKIPPED.value}:
        overall = OverallStatus.PENDING
    else:
        overall = OverallStatus.PASSED
    report: dict[str, Any] = {
        "status": overall.value,
        "strict_ready": overall is OverallStatus.PASSED,
        "project_root": str(config.project_root),
        "checks": checks,
        "limitations": [
            "验收脚本只根据可验证的文件、报告和运行结果判定状态，不推断训练数据或模型权重的来源。",
            "主题分类器目前是独立旁路组件，尚未接入实时对话和 Workflow 主链路。",
            "GPU 通过条件是实际 ONNX Runtime active_providers 包含 CUDAExecutionProvider。",
            "训练数据、模型权重、ONNX 文件和性能报告默认被 Git 忽略，不进入仓库。",
        ],
    }
    if config.output_path is not None:
        report["report_path"] = str(config.output_path)
        config.output_path.parent.mkdir(parents=True, exist_ok=True)
        config.output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return report


def _check_implementation(project_root: Path) -> dict[str, Any]:
    expected = (
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
    missing = [path for path in expected if not (project_root / path).is_file()]
    if missing:
        return _stage(
            "implementation",
            CheckStatus.FAILED,
            "分类器代码入口缺失",
            missing,
        )
    return _stage(
        "implementation",
        CheckStatus.PASSED,
        "分类体系、数据、训练、导出、推理、性能和验收入口均存在",
        expected,
    )


def _check_dataset(dataset_dir: Path) -> dict[str, Any]:
    expected = {
        TrainingDataSplit.TRAIN: dataset_dir / "train.jsonl",
        TrainingDataSplit.VALIDATION: dataset_dir / "validation.jsonl",
        TrainingDataSplit.TEST: dataset_dir / "test.jsonl",
    }
    manifest_path = dataset_dir / "manifest.json"
    missing = [str(path) for path in (*expected.values(), manifest_path) if not path.is_file()]
    if missing:
        return _stage(
            "prepared_dataset",
            CheckStatus.PENDING,
            "等待真实 train/validation/test 数据集和 manifest",
            missing,
        )
    counts: dict[str, int] = {}
    try:
        for split, path in expected.items():
            examples = load_training_dataset(path)
            if any(example.split is not split for example in examples):
                raise ClassifierDataError(
                    f"{path} contains records outside the {split.value} split"
                )
            counts[split.value] = len(examples)
        _read_json(manifest_path)
    except (ClassifierDataError, ValueError) as exc:
        return _stage("prepared_dataset", CheckStatus.FAILED, str(exc))
    return _stage(
        "prepared_dataset",
        CheckStatus.PASSED,
        "三份数据切分和 manifest 可读取",
        [*expected.values(), manifest_path],
        {"example_counts": counts},
    )


def _check_model(model_dir: Path) -> dict[str, Any]:
    required = (
        model_dir / "config.json",
        model_dir / "training_metrics.json",
        model_dir / "validation_report.json",
        model_dir / "threshold.json",
    )
    tokenizer_candidates = (model_dir / "tokenizer.json", model_dir / "tokenizer_config.json")
    weight_candidates = (model_dir / "model.safetensors", model_dir / "pytorch_model.bin")
    if not model_dir.is_dir():
        return _stage("trained_model", CheckStatus.PENDING, "等待真实训练模型目录", [model_dir])
    missing = [str(path) for path in required if not path.is_file()]
    if not any(path.is_file() and path.stat().st_size > 0 for path in tokenizer_candidates):
        missing.append("tokenizer.json or tokenizer_config.json")
    if not any(path.is_file() and path.stat().st_size > 0 for path in weight_candidates):
        missing.append("model.safetensors or pytorch_model.bin")
    if missing:
        return _stage("trained_model", CheckStatus.PENDING, "训练产物尚不完整", missing)
    try:
        _read_json(model_dir / "config.json")
        metrics = _read_json(model_dir / "training_metrics.json")
        _read_json(model_dir / "validation_report.json")
        _validate_threshold_file(model_dir / "threshold.json")
    except ValueError as exc:
        return _stage("trained_model", CheckStatus.FAILED, str(exc))
    return _stage(
        "trained_model",
        CheckStatus.PASSED,
        "训练模型、Tokenizer、指标和阈值文件可读取",
        [*required, *tokenizer_candidates, *weight_candidates],
        {"device": metrics.get("device")},
    )


def _check_onnx_export(onnx_dir: Path) -> dict[str, Any]:
    required = (
        onnx_dir / "model.onnx",
        onnx_dir / "export_manifest.json",
        onnx_dir / "threshold.json",
    )
    if not onnx_dir.is_dir():
        return _stage("onnx_export", CheckStatus.PENDING, "等待 ONNX 导出产物", [onnx_dir])
    missing = [str(path) for path in required if not path.is_file()]
    if required[0].is_file() and required[0].stat().st_size == 0:
        return _stage("onnx_export", CheckStatus.FAILED, "model.onnx 为空", [required[0]])
    if missing:
        return _stage("onnx_export", CheckStatus.PENDING, "ONNX 导出产物尚不完整", missing)
    try:
        manifest = _read_json(onnx_dir / "export_manifest.json")
        _validate_threshold_file(onnx_dir / "threshold.json")
    except ValueError as exc:
        return _stage("onnx_export", CheckStatus.FAILED, str(exc))
    return _stage(
        "onnx_export",
        CheckStatus.PASSED,
        "ONNX 模型、导出清单和阈值文件可读取",
        required,
        {"input_names": manifest.get("input_names")},
    )


def _check_parity(report_path: Path) -> dict[str, Any]:
    if not report_path.is_file():
        return _stage("onnx_pytorch_parity", CheckStatus.PENDING, "等待真实一致性验收报告", [report_path])
    try:
        report = _read_json(report_path)
    except ValueError as exc:
        return _stage("onnx_pytorch_parity", CheckStatus.FAILED, str(exc), [report_path])
    checked = report.get("checked")
    if (
        report.get("passed") is not True
        or not isinstance(checked, int)
        or checked < 1
        or report.get("logit_mismatch_rows") != 0
        or report.get("prediction_mismatch_rows") != 0
    ):
        return _stage(
            "onnx_pytorch_parity",
            CheckStatus.FAILED,
            "一致性验收报告缺少通过证据或存在不一致",
            [report_path],
        )
    return _stage(
        "onnx_pytorch_parity",
        CheckStatus.PASSED,
        "PyTorch 与 ONNX logits 和阈值预测一致",
        [report_path],
        {"checked": checked},
    )


def _check_benchmark(report_path: Path, *, require_cuda: bool) -> dict[str, Any]:
    if not report_path.is_file():
        return _stage("onnx_inference_benchmark", CheckStatus.PENDING, "等待推理冒烟和性能报告", [report_path])
    try:
        report = _read_json(report_path)
    except ValueError as exc:
        return _stage("onnx_inference_benchmark", CheckStatus.FAILED, str(exc), [report_path])
    latency = report.get("latency_ms")
    throughput = report.get("throughput_samples_per_second")
    if (
        report.get("status") != "passed"
        or report.get("smoke_passed") is not True
        or not isinstance(latency, dict)
        or not all(key in latency for key in ("min", "mean", "p50", "p95", "max"))
        or not isinstance(throughput, (int, float))
        or throughput <= 0
    ):
        return _stage(
            "onnx_inference_benchmark",
            CheckStatus.FAILED,
            "推理冒烟或性能报告缺少通过证据",
            [report_path],
        )
    active_providers = report.get("active_providers")
    if not isinstance(active_providers, list):
        return _stage(
            "onnx_inference_benchmark",
            CheckStatus.FAILED,
            "报告缺少 active_providers",
            [report_path],
        )
    if require_cuda and "CUDAExecutionProvider" not in active_providers:
        return _stage(
            "onnx_inference_benchmark",
            CheckStatus.PENDING,
            "报告仅证明推理可运行，尚未证明 CUDA Provider 实际生效",
            [report_path],
            {"active_providers": active_providers},
        )
    return _stage(
        "onnx_inference_benchmark",
        CheckStatus.PASSED,
        "推理冒烟、Provider 记录和性能指标可读取",
        [report_path],
        {"active_providers": active_providers},
    )


def _run_tests(project_root: Path, timeout_seconds: int) -> dict[str, Any]:
    command = (sys.executable, "-m", "pytest", "-q")
    try:
        result = subprocess.run(
            command,
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return _stage("automated_tests", CheckStatus.FAILED, "全量测试超时")
    output = "\n".join(part for part in (result.stdout, result.stderr) if part).strip()
    detail = output[-2000:] if output else f"pytest exited with code {result.returncode}"
    return _stage(
        "automated_tests",
        CheckStatus.PASSED if result.returncode == 0 else CheckStatus.FAILED,
        detail,
        extra={"command": list(command), "returncode": result.returncode},
    )


def _stage(
    name: str,
    status: CheckStatus,
    detail: str,
    paths: list[str | Path] | tuple[str | Path, ...] = (),
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "name": name,
        "status": status.value,
        "detail": detail,
        "paths": [str(path) for path in paths],
    }
    if extra:
        result.update(extra)
    return result


def _resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid JSON report: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON report must be an object: {path}")
    return value


def _validate_threshold_file(path: Path) -> None:
    data = _read_json(path)
    try:
        threshold = float(data["threshold"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"threshold metadata is invalid: {path}") from exc
    if not 0 <= threshold <= 1:
        raise ValueError(f"threshold metadata is outside [0, 1]: {path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Check classifier implementation, artifacts, parity, and GPU acceptance status"
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/ch10/dataset"))
    parser.add_argument("--model-dir", type=Path, default=Path("data/ch10/model"))
    parser.add_argument("--onnx-dir", type=Path, default=Path("data/ch10/onnx"))
    parser.add_argument("--parity-report", type=Path)
    parser.add_argument("--benchmark-report", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/ch10/acceptance_report.json"),
    )
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--test-timeout", type=int, default=900)
    arguments = parser.parse_args(argv)
    try:
        report = run_classifier_acceptance(
            AcceptanceConfig(
                project_root=arguments.root,
                dataset_dir=arguments.dataset_dir,
                model_dir=arguments.model_dir,
                onnx_dir=arguments.onnx_dir,
                parity_report=arguments.parity_report,
                benchmark_report=arguments.benchmark_report,
                output_path=arguments.output,
                run_tests=not arguments.skip_tests,
                require_cuda=arguments.require_cuda,
                test_timeout_seconds=arguments.test_timeout,
            )
        )
    except AcceptanceConfigError as exc:
        parser.error(str(exc))
        return
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if arguments.strict and not report["strict_ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
