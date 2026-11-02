import argparse
from dataclasses import dataclass
import json
from math import isfinite
from pathlib import Path
from typing import Any

from app.classifier_data import (
    ClassifierDataError,
    TrainingDataSplit,
    load_training_dataset,
)
from app.topic_taxonomy import NUM_TOPIC_LABELS


class ExportValidationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ExportValidationConfig:
    model_dir: Path
    onnx_dir: Path
    test_path: Path
    report_path: Path | None = None
    batch_size: int = 32
    max_length: int = 128
    atol: float = 1e-3

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_dir", Path(self.model_dir))
        object.__setattr__(self, "onnx_dir", Path(self.onnx_dir))
        object.__setattr__(self, "test_path", Path(self.test_path))
        if self.report_path is not None:
            object.__setattr__(self, "report_path", Path(self.report_path))
        if self.batch_size < 1:
            raise ExportValidationError("batch_size must be positive")
        if self.max_length < 2:
            raise ExportValidationError("max_length must be at least 2")
        if not isfinite(self.atol) or self.atol < 0:
            raise ExportValidationError("atol must be finite and non-negative")


def compare_logits(
    reference_logits: Any,
    exported_logits: Any,
    *,
    threshold: float,
    atol: float,
) -> dict[str, Any]:
    if not isfinite(threshold) or not 0 <= threshold <= 1:
        raise ExportValidationError("threshold must be between 0 and 1")
    if not isfinite(atol) or atol < 0:
        raise ExportValidationError("atol must be finite and non-negative")
    reference = _to_rows(reference_logits)
    exported = _to_rows(exported_logits)
    if not reference or len(reference) != len(exported):
        raise ExportValidationError("logit outputs must have the same non-empty batch")

    logit_mismatch_rows = 0
    prediction_mismatch_rows = 0
    max_difference = 0.0
    for reference_row, exported_row in zip(reference, exported):
        if (
            len(reference_row) != NUM_TOPIC_LABELS
            or len(exported_row) != NUM_TOPIC_LABELS
        ):
            raise ExportValidationError(
                f"logit outputs must contain {NUM_TOPIC_LABELS} topic scores per row"
            )
        row_has_logit_mismatch = False
        row_has_prediction_mismatch = False
        for reference_value, exported_value in zip(reference_row, exported_row):
            reference_value = float(reference_value)
            exported_value = float(exported_value)
            if not isfinite(reference_value) or not isfinite(exported_value):
                raise ExportValidationError("logit outputs must be finite")
            difference = abs(reference_value - exported_value)
            max_difference = max(max_difference, difference)
            row_has_logit_mismatch |= difference > atol
            row_has_prediction_mismatch |= (
                (_sigmoid(reference_value) >= threshold)
                != (_sigmoid(exported_value) >= threshold)
            )
        logit_mismatch_rows += int(row_has_logit_mismatch)
        prediction_mismatch_rows += int(row_has_prediction_mismatch)

    return {
        "checked": len(reference),
        "logit_mismatch_rows": logit_mismatch_rows,
        "prediction_mismatch_rows": prediction_mismatch_rows,
        "max_absolute_logit_difference": max_difference,
        "passed": logit_mismatch_rows == 0 and prediction_mismatch_rows == 0,
    }


def validate_onnx_parity(config: ExportValidationConfig) -> dict[str, Any]:
    model_path = config.model_dir
    onnx_path = config.onnx_dir / "model.onnx"
    threshold_path = config.onnx_dir / "threshold.json"
    if not config.test_path.is_file():
        raise ExportValidationError(f"test dataset does not exist: {config.test_path}")
    try:
        examples = load_training_dataset(config.test_path)
    except ClassifierDataError as exc:
        raise ExportValidationError(str(exc)) from exc
    if any(example.split is not TrainingDataSplit.TEST for example in examples):
        raise ExportValidationError("parity validation requires test split records")
    for path, name in (
        (model_path, "PyTorch model directory"),
        (onnx_path, "ONNX model"),
        (threshold_path, "threshold metadata"),
    ):
        if not path.exists():
            raise ExportValidationError(f"{name} does not exist: {path}")
    try:
        threshold_data = json.loads(threshold_path.read_text(encoding="utf-8"))
        threshold = float(threshold_data["threshold"])
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise ExportValidationError("threshold metadata must contain a numeric threshold") from exc
    if not isfinite(threshold) or not 0 <= threshold <= 1:
        raise ExportValidationError("threshold must be between 0 and 1")

    try:
        import onnxruntime
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise ExportValidationError(
            "ONNX parity validation requires the optional ML dependency group "
            "(torch, transformers, onnxruntime)"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    reference_model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    reference_model.to("cpu")
    reference_model.eval()
    session = onnxruntime.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    input_names = [item.name for item in session.get_inputs()]
    if not input_names or "input_ids" not in input_names:
        raise ExportValidationError("ONNX model has no supported input_ids input")

    aggregate = {
        "checked": 0,
        "logit_mismatch_rows": 0,
        "prediction_mismatch_rows": 0,
        "max_absolute_logit_difference": 0.0,
        "passed": True,
    }
    for offset in range(0, len(examples), config.batch_size):
        batch = examples[offset : offset + config.batch_size]
        encoded = tokenizer(
            [example.normalized_text() for example in batch],
            truncation=True,
            padding=True,
            max_length=config.max_length,
            return_tensors="pt",
        )
        missing_inputs = [name for name in input_names if name not in encoded]
        if missing_inputs:
            raise ExportValidationError(
                f"tokenizer is missing ONNX inputs: {', '.join(missing_inputs)}"
            )
        model_inputs = {name: encoded[name] for name in input_names}
        with torch.no_grad():
            reference_logits = reference_model(**model_inputs).logits
        runtime_inputs = {
            name: _to_numpy(encoded[name]) for name in input_names
        }
        exported_outputs = session.run(["logits"], runtime_inputs)
        if not exported_outputs:
            raise ExportValidationError("ONNX runtime returned no logits")
        batch_report = compare_logits(
            reference_logits,
            exported_outputs[0],
            threshold=threshold,
            atol=config.atol,
        )
        for key in ("checked", "logit_mismatch_rows", "prediction_mismatch_rows"):
            aggregate[key] += batch_report[key]
        aggregate["max_absolute_logit_difference"] = max(
            aggregate["max_absolute_logit_difference"],
            batch_report["max_absolute_logit_difference"],
        )
        aggregate["passed"] = aggregate["passed"] and batch_report["passed"]

    report = {
        **aggregate,
        "threshold": threshold,
        "atol": config.atol,
        "batch_size": config.batch_size,
        "providers": ["CPUExecutionProvider"],
    }
    report_path = config.report_path or config.onnx_dir / "export_validation.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def _to_rows(values: Any) -> list[list[float]]:
    if hasattr(values, "detach"):
        values = values.detach().cpu()
    if hasattr(values, "tolist"):
        values = values.tolist()
    rows = list(values)
    return [list(row) for row in rows]


def _to_numpy(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return value


def _sigmoid(value: float) -> float:
    if value >= 0:
        exp_value = 2.718281828459045 ** (-value)
        return 1 / (1 + exp_value)
    exp_value = 2.718281828459045 ** value
    return exp_value / (1 + exp_value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare PyTorch and ONNX classifier outputs on held-out test data"
    )
    parser.add_argument("--model-dir", type=Path, default=Path("data/ch10/model"))
    parser.add_argument("--onnx-dir", type=Path, default=Path("data/ch10/onnx"))
    parser.add_argument("--test", type=Path, default=Path("data/ch10/dataset/test.jsonl"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--atol", type=float, default=1e-3)
    arguments = parser.parse_args()
    report = validate_onnx_parity(
        ExportValidationConfig(
            model_dir=arguments.model_dir,
            onnx_dir=arguments.onnx_dir,
            test_path=arguments.test,
            batch_size=arguments.batch_size,
            max_length=arguments.max_length,
            atol=arguments.atol,
        )
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit("ONNX and PyTorch classifier predictions diverged")


if __name__ == "__main__":
    main()
