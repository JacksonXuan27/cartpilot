import argparse
import json
from dataclasses import dataclass
from math import floor, isfinite
from pathlib import Path
from statistics import fmean
from time import perf_counter
from typing import Any, Sequence

from app.classifier_inference import (
    InferenceConfig,
    InferenceConfigError,
    OnnxTopicClassifier,
)


class BenchmarkConfigError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class InferenceBenchmarkConfig:
    onnx_dir: Path
    providers: tuple[str, ...] = ("CPUExecutionProvider",)
    texts: tuple[str, ...] = ("买大了想退", "快递到哪了")
    batch_size: int = 1
    warmup_runs: int = 2
    measured_runs: int = 10
    max_length: int = 128
    threshold: float | None = None
    require_provider: str | None = None
    output_path: Path | None = None

    def __post_init__(self) -> None:
        if not str(self.onnx_dir).strip():
            raise BenchmarkConfigError("onnx_dir cannot be empty")
        providers = (
            (self.providers,)
            if isinstance(self.providers, str)
            else tuple(self.providers)
        )
        if not providers or any(
            not isinstance(provider, str) or not provider.strip()
            for provider in providers
        ):
            raise BenchmarkConfigError("provider list cannot be empty")
        texts = (self.texts,) if isinstance(self.texts, str) else tuple(self.texts)
        if not texts or any(
            not isinstance(text, str) or not text.strip() for text in texts
        ):
            raise BenchmarkConfigError("texts must contain non-empty text")
        if self.batch_size < 1:
            raise BenchmarkConfigError("batch_size must be positive")
        if self.warmup_runs < 0:
            raise BenchmarkConfigError("warmup_runs cannot be negative")
        if self.measured_runs < 1:
            raise BenchmarkConfigError("measured_runs must be positive")
        if self.max_length < 2:
            raise BenchmarkConfigError("max_length must be at least 2")
        if self.threshold is not None:
            try:
                numeric_threshold = float(self.threshold)
            except (TypeError, ValueError) as exc:
                raise BenchmarkConfigError(
                    "threshold must be a finite number between 0 and 1"
                ) from exc
            if not isfinite(numeric_threshold) or not 0 <= numeric_threshold <= 1:
                raise BenchmarkConfigError(
                    "threshold must be a finite number between 0 and 1"
                )
        if self.require_provider is not None and not self.require_provider.strip():
            raise BenchmarkConfigError("require_provider cannot be empty")
        if self.require_provider is not None and self.require_provider not in providers:
            raise BenchmarkConfigError("require_provider must be included in providers")
        object.__setattr__(self, "onnx_dir", Path(self.onnx_dir))
        object.__setattr__(self, "providers", providers)
        object.__setattr__(self, "texts", texts)
        if self.output_path is not None:
            object.__setattr__(self, "output_path", Path(self.output_path))


def available_execution_providers() -> tuple[str, ...]:
    try:
        import onnxruntime
    except ImportError as exc:
        raise BenchmarkConfigError(
            "inference benchmarking requires the optional ML dependency group "
            "(transformers, onnxruntime)"
        ) from exc
    getter = getattr(onnxruntime, "get_available_providers", None)
    if getter is None:
        raise BenchmarkConfigError("onnxruntime does not expose available providers")
    providers = tuple(getter())
    if not providers:
        raise BenchmarkConfigError("onnxruntime reported no execution providers")
    return providers


def run_inference_benchmark(config: InferenceBenchmarkConfig) -> dict[str, Any]:
    available = available_execution_providers()
    if config.require_provider is not None and config.require_provider not in available:
        raise BenchmarkConfigError(
            f"required execution provider is unavailable: {config.require_provider}"
        )
    effective_providers = tuple(
        provider for provider in config.providers if provider in available
    )
    if not effective_providers:
        raise BenchmarkConfigError(
            "none of the requested execution providers are available"
        )
    unavailable = tuple(
        provider for provider in config.providers if provider not in available
    )
    try:
        classifier = OnnxTopicClassifier(
            InferenceConfig(
                onnx_dir=config.onnx_dir,
                providers=effective_providers,
                max_length=config.max_length,
                threshold=config.threshold,
            )
        )
    except InferenceConfigError as exc:
        raise BenchmarkConfigError(str(exc)) from exc

    batch = tuple(
        config.texts[index % len(config.texts)] for index in range(config.batch_size)
    )
    for _ in range(config.warmup_runs):
        _run_batch(classifier, batch)

    latencies_ms: list[float] = []
    for _ in range(config.measured_runs):
        started_at = perf_counter()
        _run_batch(classifier, batch)
        latency_ms = (perf_counter() - started_at) * 1000
        if not isfinite(latency_ms) or latency_ms < 0:
            raise BenchmarkConfigError("inference produced an invalid latency")
        latencies_ms.append(latency_ms)

    active_providers = tuple(classifier.execution_providers)
    total_seconds = sum(latencies_ms) / 1000
    report = {
        "status": "passed",
        "smoke_passed": True,
        "requested_providers": list(config.providers),
        "unavailable_requested_providers": list(unavailable),
        "available_providers": list(available),
        "effective_providers": list(effective_providers),
        "active_providers": list(active_providers),
        "cuda_provider_active": "CUDAExecutionProvider" in active_providers,
        "onnx_dir": str(config.onnx_dir),
        "batch_size": config.batch_size,
        "warmup_runs": config.warmup_runs,
        "measured_runs": config.measured_runs,
        "sample_count": config.batch_size * config.measured_runs,
        "latency_ms": {
            "min": min(latencies_ms),
            "mean": fmean(latencies_ms),
            "p50": _percentile(latencies_ms, 0.50),
            "p95": _percentile(latencies_ms, 0.95),
            "max": max(latencies_ms),
        },
        "throughput_samples_per_second": (
            0.0
            if total_seconds == 0
            else config.batch_size * config.measured_runs / total_seconds
        ),
    }
    if config.require_provider is not None:
        report["required_provider"] = config.require_provider
        report["required_provider_active"] = config.require_provider in active_providers
        if not report["required_provider_active"]:
            raise BenchmarkConfigError(
                f"required execution provider was not active: {config.require_provider}"
            )
    if config.output_path is not None:
        report["report_path"] = str(config.output_path)
        config.output_path.parent.mkdir(parents=True, exist_ok=True)
        config.output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return report


def _run_batch(classifier: OnnxTopicClassifier, batch: Sequence[str]) -> None:
    results = classifier.predict(batch)
    if len(results) != len(batch):
        raise BenchmarkConfigError(
            "classifier returned a different number of results than the batch size"
        )


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise BenchmarkConfigError("cannot calculate a percentile for empty values")
    if not 0 <= quantile <= 1:
        raise BenchmarkConfigError("quantile must be between 0 and 1")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower_index = floor(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = position - lower_index
    return ordered[lower_index] + (
        ordered[upper_index] - ordered[lower_index]
    ) * fraction


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an ONNX topic-classifier smoke test and latency benchmark"
    )
    parser.add_argument("--onnx-dir", type=Path, default=Path("data/ch10/onnx"))
    parser.add_argument(
        "--providers",
        nargs="+",
        default=["CPUExecutionProvider"],
        help="ordered ONNX Runtime providers; unavailable providers are skipped",
    )
    parser.add_argument("--require-provider")
    parser.add_argument("--text", dest="texts", action="append")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--threshold", type=float)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/ch10/onnx/inference_benchmark.json"),
    )
    arguments = parser.parse_args()
    try:
        report = run_inference_benchmark(
            InferenceBenchmarkConfig(
                onnx_dir=arguments.onnx_dir,
                providers=tuple(arguments.providers),
                texts=tuple(arguments.texts or ("买大了想退", "快递到哪了")),
                batch_size=arguments.batch_size,
                warmup_runs=arguments.warmup_runs,
                measured_runs=arguments.iterations,
                max_length=arguments.max_length,
                threshold=arguments.threshold,
                require_provider=arguments.require_provider,
                output_path=arguments.output,
            )
        )
    except BenchmarkConfigError as exc:
        raise SystemExit(f"inference benchmark failed: {exc}") from exc
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
