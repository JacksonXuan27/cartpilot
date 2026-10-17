import json
from dataclasses import dataclass
from math import exp, isfinite
from pathlib import Path
from typing import Any, Sequence

from app.topic_taxonomy import ID2LABEL, NUM_TOPIC_LABELS, TopicLabel


class InferenceConfigError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class InferenceConfig:
    onnx_dir: Path
    providers: tuple[str, ...] = ("CPUExecutionProvider",)
    max_length: int = 128
    threshold: float | None = None

    def __post_init__(self) -> None:
        if not str(self.onnx_dir).strip():
            raise InferenceConfigError("onnx_dir cannot be empty")
        providers = (
            (self.providers,)
            if isinstance(self.providers, str)
            else tuple(self.providers)
        )
        if not providers or any(
            not isinstance(provider, str) or not provider.strip()
            for provider in providers
        ):
            raise InferenceConfigError("provider list cannot be empty")
        if self.max_length < 2:
            raise InferenceConfigError("max_length must be at least 2")
        if self.threshold is not None:
            _validate_threshold(self.threshold)
        object.__setattr__(self, "onnx_dir", Path(self.onnx_dir))
        object.__setattr__(self, "providers", providers)


@dataclass(frozen=True, slots=True)
class TopicPrediction:
    label: TopicLabel
    probability: float

    def __post_init__(self) -> None:
        if not isfinite(self.probability) or not 0 <= self.probability <= 1:
            raise InferenceConfigError("prediction probability must be between 0 and 1")

    def to_dict(self) -> dict[str, object]:
        return {"label": self.label.value, "probability": self.probability}


@dataclass(frozen=True, slots=True)
class ClassificationResult:
    text: str
    predictions: tuple[TopicPrediction, ...]

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise InferenceConfigError("text cannot be empty")
        if not self.predictions:
            raise InferenceConfigError(
                "classification must contain at least one prediction"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "predictions": [prediction.to_dict() for prediction in self.predictions],
        }


def probabilities_from_logits(logits: Any) -> list[list[float]]:
    rows = _validated_rows(logits, "logits")
    return [[_sigmoid(value) for value in row] for row in rows]


def apply_threshold(probabilities: Any, threshold: float) -> list[list[bool]]:
    numeric_threshold = _validate_threshold(threshold)
    rows = _validated_rows(probabilities, "probabilities", probability_values=True)
    selected: list[list[bool]] = []
    for row in rows:
        flags = [value >= numeric_threshold for value in row]
        if not any(flags):
            flags[row.index(max(row))] = True
        selected.append(flags)
    return selected


class OnnxTopicClassifier:
    def __init__(self, config: InferenceConfig) -> None:
        self.config = config
        model_path = config.onnx_dir / "model.onnx"
        threshold_path = config.onnx_dir / "threshold.json"
        if not model_path.is_file():
            raise InferenceConfigError(f"model.onnx is missing from {config.onnx_dir}")
        if not threshold_path.is_file():
            raise InferenceConfigError(f"threshold.json is missing from {config.onnx_dir}")
        metadata_threshold = _load_threshold(threshold_path)
        self.threshold = (
            _validate_threshold(config.threshold)
            if config.threshold is not None
            else metadata_threshold
        )
        try:
            import onnxruntime
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise InferenceConfigError(
                "ONNX inference requires the optional ML dependency group "
                "(transformers, onnxruntime)"
            ) from exc
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(str(config.onnx_dir))
            self._session = onnxruntime.InferenceSession(
                str(model_path), providers=list(config.providers)
            )
        except Exception as exc:
            raise InferenceConfigError("failed to load ONNX classifier resources") from exc
        self._input_names = tuple(item.name for item in self._session.get_inputs())
        if not self._input_names:
            raise InferenceConfigError("ONNX model has no inputs")
        if "input_ids" not in self._input_names:
            raise InferenceConfigError("ONNX model has no supported input_ids input")

    def predict(self, texts: Sequence[str]) -> tuple[ClassificationResult, ...]:
        normalized_texts = _validate_texts(texts)
        try:
            encoded = self._tokenizer(
                normalized_texts,
                truncation=True,
                padding=True,
                max_length=self.config.max_length,
                return_tensors="np",
            )
        except Exception as exc:
            raise InferenceConfigError("tokenizer failed to encode classifier texts") from exc
        runtime_inputs: dict[str, Any] = {}
        for input_name in self._input_names:
            if input_name not in encoded:
                raise InferenceConfigError(
                    f"tokenizer is missing ONNX input: {input_name}"
                )
            runtime_inputs[input_name] = encoded[input_name]
        try:
            outputs = self._session.run(None, runtime_inputs)
        except Exception as exc:
            raise InferenceConfigError("ONNX inference failed") from exc
        if not outputs:
            raise InferenceConfigError("ONNX runtime returned no logits")
        probabilities = probabilities_from_logits(outputs[0])
        if len(probabilities) != len(normalized_texts):
            raise InferenceConfigError(
                "ONNX runtime returned a different number of rows than the input texts"
            )
        selected = apply_threshold(probabilities, self.threshold)
        results: list[ClassificationResult] = []
        for text, probability_row, selected_row in zip(
            normalized_texts, probabilities, selected
        ):
            predictions = tuple(
                TopicPrediction(label=ID2LABEL[index], probability=probability)
                for index, (probability, is_selected) in enumerate(
                    zip(probability_row, selected_row)
                )
                if is_selected
            )
            results.append(ClassificationResult(text=text, predictions=predictions))
        return tuple(results)

    def classify(self, text: str) -> ClassificationResult:
        return self.predict((text,))[0]

    @property
    def execution_providers(self) -> tuple[str, ...]:
        getter = getattr(self._session, "get_providers", None)
        if getter is None:
            return self.config.providers
        return tuple(getter())


def _load_threshold(path: Path) -> float:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        threshold = data["threshold"]
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise InferenceConfigError(
            "threshold.json must contain a numeric threshold"
        ) from exc
    return _validate_threshold(threshold)


def _validate_threshold(value: float) -> float:
    try:
        threshold = float(value)
    except (TypeError, ValueError) as exc:
        raise InferenceConfigError(
            "threshold must be a finite number between 0 and 1"
        ) from exc
    if not isfinite(threshold) or not 0 <= threshold <= 1:
        raise InferenceConfigError(
            "threshold must be a finite number between 0 and 1"
        )
    return threshold


def _validated_rows(
    values: Any,
    name: str,
    *,
    probability_values: bool = False,
) -> list[list[float]]:
    if hasattr(values, "detach"):
        values = values.detach().cpu()
    if hasattr(values, "tolist"):
        values = values.tolist()
    try:
        raw_rows = list(values)
    except TypeError as exc:
        raise InferenceConfigError(
            f"{name} must be a two-dimensional collection"
        ) from exc
    if not raw_rows:
        raise InferenceConfigError(f"{name} cannot be empty")
    rows: list[list[float]] = []
    for raw_row in raw_rows:
        if isinstance(raw_row, (str, bytes)):
            raise InferenceConfigError(
                f"{name} must be a two-dimensional collection"
            )
        try:
            raw_values = list(raw_row)
        except TypeError as exc:
            raise InferenceConfigError(
                f"{name} must be a two-dimensional collection"
            ) from exc
        if len(raw_values) != NUM_TOPIC_LABELS:
            raise InferenceConfigError(
                f"each row must contain {NUM_TOPIC_LABELS} topic scores"
            )
        row: list[float] = []
        for raw_value in raw_values:
            try:
                value = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise InferenceConfigError(
                    f"{name} must contain numeric scores"
                ) from exc
            if not isfinite(value):
                raise InferenceConfigError(f"{name} must contain finite scores")
            if probability_values and not 0 <= value <= 1:
                raise InferenceConfigError("probabilities must be between 0 and 1")
            row.append(value)
        rows.append(row)
    return rows


def _validate_texts(texts: Sequence[str]) -> list[str]:
    if isinstance(texts, (str, bytes)):
        raise InferenceConfigError(
            "predict expects a sequence of texts; use classify for one text"
        )
    try:
        normalized = list(texts)
    except TypeError as exc:
        raise InferenceConfigError("texts must be a non-empty sequence") from exc
    if not normalized or any(
        not isinstance(text, str) or not text.strip() for text in normalized
    ):
        raise InferenceConfigError("text cannot be empty")
    return normalized


def _sigmoid(value: float) -> float:
    if value >= 0:
        exponent = exp(-value)
        return 1 / (1 + exponent)
    exponent = exp(value)
    return exponent / (1 + exponent)
