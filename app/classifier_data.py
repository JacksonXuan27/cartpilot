from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.intent_routing import IntentCategory


class ClassifierDataError(ValueError):
    pass


class TrainingDataSource(str, Enum):
    HUMAN_ANNOTATED = "human_annotated"
    SYNTHETIC = "synthetic"
    REVIEWED_FEEDBACK = "reviewed_feedback"


class TrainingDataSplit(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


@dataclass(frozen=True, slots=True)
class TrainingDatasetSplits:
    train: tuple[TrainingExample, ...]
    validation: tuple[TrainingExample, ...]
    test: tuple[TrainingExample, ...]


_WHITESPACE = re.compile(r"\s+")
_PHONE_NUMBER = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
_LONG_NUMBER = re.compile(r"\d{10,}")
_SOCIAL_ACCOUNT = re.compile(r"(微信|weixin|wx|QQ)[号:: ]*[A-Za-z0-9_-]{5,}\s?", re.IGNORECASE)


TOPIC_TAXONOMY: dict[IntentCategory, str] = {
    IntentCategory.LOGISTICS: "物流轨迹、配送进度和签收问题",
    IntentCategory.ORDER: "订单查询、订单状态和订单信息变更",
    IntentCategory.PRODUCT: "商品属性、使用方式和售前咨询",
    IntentCategory.REFUND_RETURN: "退款、退货和取消订单后的退款诉求",
    IntentCategory.AFTER_SALES: "维修、换新、补发及商品质量问题",
    IntentCategory.COMPLAINT: "投诉、服务争议和负面体验升级",
    IntentCategory.HUMAN: "明确要求转接人工客服",
    IntentCategory.SMALLTALK: "问候、感谢、告别等寒暄表达",
    IntentCategory.OTHER: "不属于已定义业务主题或信息不足的请求",
}


class TrainingExample(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    example_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    )
    text: str = Field(min_length=1, max_length=2000)
    label: IntentCategory
    source: TrainingDataSource
    split: TrainingDataSplit | None = None

    def normalized_text(self) -> str:
        text = _PHONE_NUMBER.sub("[手机号]", self.text)
        text = _LONG_NUMBER.sub("[单号]", text)
        text = _SOCIAL_ACCOUNT.sub(lambda match: f"{match.group(1)}[账号]", text)
        return _WHITESPACE.sub(" ", text).strip()


def prepare_training_dataset(
    examples: list[TrainingExample], *, seed: int = 42
) -> TrainingDatasetSplits:
    if not examples:
        raise ClassifierDataError("training dataset cannot be empty")
    labels_by_text: dict[str, IntentCategory] = {}
    seen_ids: set[str] = set()
    explicit_split_by_text: dict[str, TrainingDataSplit] = {}
    first_by_text: dict[str, TrainingExample] = {}
    for example in examples:
        if example.example_id in seen_ids:
            raise ClassifierDataError(
                f"duplicate example_id: {example.example_id}"
            )
        seen_ids.add(example.example_id)
        text = example.normalized_text()
        if not text:
            raise ClassifierDataError(
                f"training text cannot be blank: {example.example_id}"
            )
        previous_label = labels_by_text.get(text)
        if previous_label is not None and previous_label is not example.label:
            raise ClassifierDataError(
                f"conflicting labels for normalized text: {text}"
            )
        if example.split is not None:
            previous_split = explicit_split_by_text.get(text)
            if previous_split is not None and previous_split is not example.split:
                raise ClassifierDataError(
                    f"text appears in multiple splits: {text}"
                )
            explicit_split_by_text[text] = example.split
        labels_by_text[text] = example.label
        if text in first_by_text:
            continue
        normalized = example.model_copy(update={"text": text})
        first_by_text[text] = normalized

    preassigned: dict[TrainingDataSplit, list[TrainingExample]] = {
        split: [] for split in TrainingDataSplit
    }
    unseen: dict[IntentCategory, list[TrainingExample]] = defaultdict(list)
    for example in first_by_text.values():
        assigned_split = explicit_split_by_text.get(example.text, example.split)
        if assigned_split is None:
            unseen[example.label].append(example)
            continue
        preassigned[assigned_split].append(
            example.model_copy(update={"split": assigned_split})
        )

    generator = random.Random(seed)
    for label in sorted(unseen, key=lambda item: item.value):
        items = unseen[label]
        generator.shuffle(items)
        count = len(items)
        if count < 3:
            preassigned[TrainingDataSplit.TRAIN].extend(items)
            continue
        test_count = max(1, round(count * 0.1))
        validation_count = max(1, round(count * 0.1))
        preassigned[TrainingDataSplit.TEST].extend(items[:test_count])
        preassigned[TrainingDataSplit.VALIDATION].extend(
            items[test_count:test_count + validation_count]
        )
        preassigned[TrainingDataSplit.TRAIN].extend(
            items[test_count + validation_count:]
        )

    prepared: dict[TrainingDataSplit, tuple[TrainingExample, ...]] = {}
    for split, items in preassigned.items():
        prepared[split] = tuple(
            sorted(
                (
                    item.model_copy(update={"split": split})
                    for item in items
                ),
                key=lambda item: item.example_id,
            )
        )
    return TrainingDatasetSplits(
        train=prepared[TrainingDataSplit.TRAIN],
        validation=prepared[TrainingDataSplit.VALIDATION],
        test=prepared[TrainingDataSplit.TEST],
    )


def write_training_dataset_splits(
    splits: TrainingDatasetSplits,
    output_dir: str | Path,
    *,
    seed: int,
    input_sha256: str,
) -> dict[str, Path]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    split_examples = {
        TrainingDataSplit.TRAIN: splits.train,
        TrainingDataSplit.VALIDATION: splits.validation,
        TrainingDataSplit.TEST: splits.test,
    }
    for split in TrainingDataSplit:
        examples = split_examples[split]
        destination = directory / f"{split.value}.jsonl"
        content = "\n".join(
            example.model_dump_json()
            for example in examples
        )
        destination.write_text(content + ("\n" if content else ""), encoding="utf-8")
        written[split.value] = destination

    manifest = {
        "format_version": 1,
        "seed": seed,
        "input_sha256": input_sha256,
        "total_examples": sum(len(items) for items in split_examples.values()),
        "splits": {
            split.value: {
                "examples": len(items),
                "labels": {
                    label.value: sum(example.label is label for example in items)
                    for label in IntentCategory
                },
            }
            for split, items in split_examples.items()
        },
    }
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    written["manifest"] = manifest_path
    return written


def prepare_dataset(
    input_path: str | Path, output_dir: str | Path, *, seed: int = 42
) -> TrainingDatasetSplits:
    examples = load_training_dataset(input_path)
    splits = prepare_training_dataset(examples, seed=seed)
    input_sha256 = hashlib.sha256(Path(input_path).read_bytes()).hexdigest()
    write_training_dataset_splits(
        splits,
        output_dir,
        seed=seed,
        input_sha256=input_sha256,
    )
    return splits


def load_training_dataset(path: str | Path) -> list[TrainingExample]:
    examples: list[TrainingExample] = []
    seen_ids: set[str] = set()
    try:
        with Path(path).open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                    if not isinstance(payload, dict):
                        raise TypeError("each JSONL record must be an object")
                    example = TrainingExample.model_validate(payload)
                except (json.JSONDecodeError, TypeError, ValidationError) as exc:
                    raise ClassifierDataError(
                        f"invalid training example at line {line_number}"
                    ) from exc
                if example.example_id in seen_ids:
                    raise ClassifierDataError(
                        f"duplicate example_id at line {line_number}: {example.example_id}"
                    )
                seen_ids.add(example.example_id)
                if not example.normalized_text():
                    raise ClassifierDataError(
                        f"training text cannot be blank at line {line_number}"
                    )
                examples.append(example)
    except OSError as exc:
        raise ClassifierDataError(f"cannot read training dataset: {path}") from exc
    if not examples:
        raise ClassifierDataError("training dataset cannot be empty")
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize and stratify classifier JSONL training data."
    )
    parser.add_argument("--input", required=True, help="Input JSONL dataset")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()
    splits = prepare_dataset(
        arguments.input,
        arguments.output_dir,
        seed=arguments.seed,
    )
    print(
        "Prepared classifier dataset: "
        f"train={len(splits.train)}, "
        f"validation={len(splits.validation)}, "
        f"test={len(splits.test)}, "
        f"seed={arguments.seed}"
    )


if __name__ == "__main__":
    main()
