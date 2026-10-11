from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from app.classifier_data import (
    ClassifierDataError,
    TrainingDataSplit,
    TrainingExample,
    load_training_dataset,
)
from app.topic_taxonomy import LABEL2ID, NUM_TOPIC_LABELS, TopicLabel


class ClassifierDatasetError(ValueError):
    pass


class TextTokenizer(Protocol):
    def __call__(
        self,
        texts: Sequence[str],
        *,
        truncation: bool,
        padding: str,
        max_length: int,
    ) -> Mapping[str, Sequence[Sequence[int]]]: ...


@dataclass(frozen=True, slots=True)
class LoadedTrainingSplits:
    train: tuple[TrainingExample, ...]
    validation: tuple[TrainingExample, ...]


def encode_labels(labels: Sequence[TopicLabel | str]) -> list[float]:
    if not labels:
        raise ClassifierDatasetError("at least one topic label is required")
    try:
        normalized = tuple(TopicLabel(label) for label in labels)
    except ValueError as exc:
        raise ClassifierDatasetError("unknown topic label") from exc
    if len(set(normalized)) != len(normalized):
        raise ClassifierDatasetError("topic labels cannot contain duplicates")
    targets = [0.0] * NUM_TOPIC_LABELS
    for label in normalized:
        targets[LABEL2ID[label]] = 1.0
    return targets


def load_train_validation_splits(
    train_path: str | Path,
    validation_path: str | Path,
) -> LoadedTrainingSplits:
    train = _load_expected_split(train_path, TrainingDataSplit.TRAIN)
    validation = _load_expected_split(validation_path, TrainingDataSplit.VALIDATION)
    train_texts = {example.normalized_text() for example in train}
    validation_texts = {example.normalized_text() for example in validation}
    if train_texts & validation_texts:
        raise ClassifierDatasetError("train and validation splits overlap by normalized text")
    return LoadedTrainingSplits(train=tuple(train), validation=tuple(validation))


def tokenize_examples(
    examples: Sequence[TrainingExample],
    tokenizer: TextTokenizer,
    *,
    max_length: int,
) -> list[dict[str, object]]:
    if not examples:
        raise ClassifierDatasetError("cannot tokenize an empty example collection")
    if max_length < 2:
        raise ClassifierDatasetError("max_length must be at least 2")
    try:
        tokenized = tokenizer(
            [example.normalized_text() for example in examples],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
    except Exception as exc:
        raise ClassifierDatasetError("tokenizer failed to encode classifier examples") from exc
    for field in ("input_ids", "attention_mask"):
        if field not in tokenized or len(tokenized[field]) != len(examples):
            raise ClassifierDatasetError(f"tokenizer returned invalid {field}")
    if "token_type_ids" in tokenized and len(tokenized["token_type_ids"]) != len(examples):
        raise ClassifierDatasetError("tokenizer returned invalid token_type_ids")

    encoded: list[dict[str, object]] = []
    for index, example in enumerate(examples):
        item: dict[str, object] = {
            "input_ids": list(tokenized["input_ids"][index]),
            "attention_mask": list(tokenized["attention_mask"][index]),
            "labels": encode_labels(example.labels),
        }
        if "token_type_ids" in tokenized:
            item["token_type_ids"] = list(tokenized["token_type_ids"][index])
        encoded.append(item)
    return encoded


def _load_expected_split(
    path: str | Path,
    expected_split: TrainingDataSplit,
) -> list[TrainingExample]:
    try:
        examples = load_training_dataset(path)
    except ClassifierDataError as exc:
        if "cannot be empty" in str(exc):
            raise ClassifierDatasetError(
                f"{expected_split.value} split cannot be empty"
            ) from exc
        raise ClassifierDatasetError(str(exc)) from exc
    if any(example.split is not expected_split for example in examples):
        raise ClassifierDatasetError(
            f"expected {expected_split.value} split records in {path}"
        )
    return examples
