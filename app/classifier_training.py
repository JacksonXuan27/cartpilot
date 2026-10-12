import json
from dataclasses import dataclass
from enum import Enum
from math import isfinite
from pathlib import Path
from typing import Any

from app.classifier_dataset import (
    load_train_validation_splits,
    tokenize_examples,
)
from app.topic_taxonomy import ID2LABEL, LABEL2ID, NUM_TOPIC_LABELS


class TrainingConfigError(ValueError):
    pass


class TrainingDevicePreference(str, Enum):
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"


@dataclass(frozen=True, slots=True)
class TrainingRunConfig:
    train_path: Path
    validation_path: Path
    output_dir: Path
    model_name_or_path: str = "hfl/chinese-roberta-wwm-ext"
    epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 2e-5
    max_length: int = 128
    seed: int = 42
    device: TrainingDevicePreference = TrainingDevicePreference.AUTO

    def __post_init__(self) -> None:
        object.__setattr__(self, "train_path", Path(self.train_path))
        object.__setattr__(self, "validation_path", Path(self.validation_path))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        try:
            preference = TrainingDevicePreference(self.device)
        except ValueError as exc:
            raise TrainingConfigError(
                f"unsupported device preference: {self.device}"
            ) from exc
        object.__setattr__(self, "device", preference)
        if not self.model_name_or_path.strip():
            raise TrainingConfigError("model_name_or_path cannot be empty")
        if self.epochs < 1:
            raise TrainingConfigError("epochs must be positive")
        if self.batch_size < 1:
            raise TrainingConfigError("batch_size must be positive")
        if not isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise TrainingConfigError("learning_rate must be a positive finite number")
        if self.max_length < 2:
            raise TrainingConfigError("max_length must be at least 2")
        if self.seed < 0:
            raise TrainingConfigError("seed cannot be negative")


def select_training_device(
    preference: TrainingDevicePreference | str,
    cuda_available: bool,
) -> str:
    try:
        selected = TrainingDevicePreference(preference)
    except ValueError as exc:
        raise TrainingConfigError(
            f"unsupported device preference: {preference}"
        ) from exc
    if selected is TrainingDevicePreference.CUDA and not cuda_available:
        raise TrainingConfigError("CUDA was requested but is unavailable")
    if selected is TrainingDevicePreference.AUTO:
        return "cuda" if cuda_available else "cpu"
    return selected.value


def compute_micro_f1(eval_prediction: Any) -> float:
    """Calculate multilabel micro-F1 without importing the ML stack."""
    predictions, labels = eval_prediction
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    return build_multilabel_report(predictions, labels)["micro_f1"]


def compute_classification_metrics(eval_prediction: Any) -> dict[str, float]:
    predictions, labels = eval_prediction
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    report = build_multilabel_report(predictions, labels)
    return {"micro_f1": report["micro_f1"], "macro_f1": report["macro_f1"]}


def build_multilabel_report(
    logits: Any,
    labels: Any,
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    if not isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite and between 0 and 1")
    if len(logits) == 0 or len(logits) != len(labels):
        raise ValueError("logits and labels must contain the same non-empty rows")

    per_class_counts = [
        {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        for _ in range(NUM_TOPIC_LABELS)
    ]
    for logit_row, label_row in zip(logits, labels):
        if len(logit_row) != NUM_TOPIC_LABELS or len(label_row) != NUM_TOPIC_LABELS:
            raise ValueError(f"each row must contain {NUM_TOPIC_LABELS} topic values")
        for index, (logit, label) in enumerate(zip(logit_row, label_row)):
            numeric_logit = float(logit)
            numeric_label = float(label)
            if not isfinite(numeric_logit) or numeric_label not in (0.0, 1.0):
                raise ValueError("logits must be finite and labels must be binary")
            predicted = _sigmoid(numeric_logit) >= threshold
            expected = numeric_label == 1.0
            counts = per_class_counts[index]
            if predicted and expected:
                counts["tp"] += 1
            elif predicted:
                counts["fp"] += 1
            elif expected:
                counts["fn"] += 1
            else:
                counts["tn"] += 1

    per_class: dict[str, dict[str, Any]] = {}
    total_tp = total_fp = total_fn = 0
    class_f1_scores: list[float] = []
    for index, label in ID2LABEL.items():
        counts = per_class_counts[index]
        true_positive = counts["tp"]
        false_positive = counts["fp"]
        false_negative = counts["fn"]
        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative
        precision = (
            true_positive / precision_denominator if precision_denominator else 0.0
        )
        recall = true_positive / recall_denominator if recall_denominator else 0.0
        f1_denominator = 2 * true_positive + false_positive + false_negative
        f1 = 2 * true_positive / f1_denominator if f1_denominator else 0.0
        per_class[label.value] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": recall_denominator,
            "confusion_matrix": [
                [counts["tn"], counts["fp"]],
                [counts["fn"], counts["tp"]],
            ],
        }
        class_f1_scores.append(f1)
        total_tp += true_positive
        total_fp += false_positive
        total_fn += false_negative

    micro_denominator = 2 * total_tp + total_fp + total_fn
    return {
        "threshold": threshold,
        "micro_f1": (
            0.0 if micro_denominator == 0 else 2 * total_tp / micro_denominator
        ),
        "macro_f1": sum(class_f1_scores) / len(class_f1_scores),
        "per_class": per_class,
    }


def run_training(config: TrainingRunConfig) -> dict[str, Any]:
    """Run one Trainer job and persist only the final model artifacts."""
    try:
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
            default_data_collator,
            set_seed,
        )
    except ImportError as exc:
        raise TrainingConfigError(
            "classifier training requires the optional ML dependency group "
            "(torch, transformers, accelerate, numpy, scikit-learn)"
        ) from exc

    device = select_training_device(config.device, bool(torch.cuda.is_available()))
    set_seed(config.seed)
    splits = load_train_validation_splits(config.train_path, config.validation_path)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
    train_dataset = tokenize_examples(
        splits.train, tokenizer, max_length=config.max_length
    )
    validation_dataset = tokenize_examples(
        splits.validation, tokenizer, max_length=config.max_length
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name_or_path,
        num_labels=NUM_TOPIC_LABELS,
        problem_type="multi_label_classification",
        id2label={index: label.value for index, label in ID2LABEL.items()},
        label2id={label.value: index for label, index in LABEL2ID.items()},
    )
    training_arguments = TrainingArguments(
        output_dir=str(config.output_dir / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="no",
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        num_train_epochs=config.epochs,
        weight_decay=0.01,
        metric_for_best_model="micro_f1",
        greater_is_better=True,
        logging_strategy="epoch",
        report_to="none",
        seed=config.seed,
        use_cpu=device == "cpu",
    )
    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        data_collator=default_data_collator,
        compute_metrics=compute_classification_metrics,
    )
    train_result = trainer.train()
    prediction_result = trainer.predict(
        validation_dataset,
        metric_key_prefix="validation",
    )
    validation_report = build_multilabel_report(
        prediction_result.predictions,
        prediction_result.label_ids,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(config.output_dir))
    tokenizer.save_pretrained(str(config.output_dir))
    metrics = {
        "device": device,
        "train_metrics": _json_ready(getattr(train_result, "metrics", {})),
        "validation_metrics": _json_ready(prediction_result.metrics),
        "validation_report": validation_report,
    }
    (config.output_dir / "training_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (config.output_dir / "validation_report.json").write_text(
        json.dumps(validation_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metrics


def _sigmoid(value: float) -> float:
    if value >= 0:
        exponent = 2.718281828459045 ** (-value)
        return 1 / (1 + exponent)
    exponent = 2.718281828459045 ** value
    return exponent / (1 + exponent)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value
