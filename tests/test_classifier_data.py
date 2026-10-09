import json

import pytest
from pydantic import ValidationError

from app.classifier_data import (
    TOPIC_TAXONOMY,
    ClassifierDataError,
    TrainingDataSource,
    TrainingDataSplit,
    TrainingExample,
    load_training_dataset,
)
from app.intent_routing import IntentCategory


def test_topic_taxonomy_covers_every_existing_intent_label():
    assert set(TOPIC_TAXONOMY) == set(IntentCategory)
    assert all(description.strip() for description in TOPIC_TAXONOMY.values())


def test_training_example_accepts_versioned_provenance_and_optional_split():
    example = TrainingExample(
        example_id="sample-1",
        text="  我的快递 到哪了  ",
        label=IntentCategory.LOGISTICS,
        source=TrainingDataSource.HUMAN_ANNOTATED,
        split=TrainingDataSplit.TRAIN,
    )

    assert example.schema_version == 1
    assert example.normalized_text() == "我的快递 到哪了"
    assert example.label is IntentCategory.LOGISTICS
    assert example.split is TrainingDataSplit.TRAIN


def test_training_example_rejects_unknown_labels_and_extra_fields():
    with pytest.raises(ValidationError):
        TrainingExample(
            example_id="sample-1",
            text="物流查询",
            label="shipping",
            source="human_annotated",
        )
    with pytest.raises(ValidationError):
        TrainingExample(
            example_id="sample-2",
            text="物流查询",
            label="logistics",
            source="human_annotated",
            customer_phone="123456",
        )


def test_training_dataset_loader_reads_jsonl_and_skips_blank_lines(tmp_path):
    dataset = tmp_path / "samples.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "example_id": "sample-1",
                "text": "查一下物流",
                "label": "logistics",
                "source": "reviewed_feedback",
                "split": "validation",
            },
            ensure_ascii=False,
        )
        + "\n\n",
        encoding="utf-8",
    )

    examples = load_training_dataset(dataset)

    assert len(examples) == 1
    assert examples[0].source is TrainingDataSource.REVIEWED_FEEDBACK
    assert examples[0].split is TrainingDataSplit.VALIDATION


@pytest.mark.parametrize(
    "content",
    [
        "not-json\n",
        json.dumps(
            {
                "example_id": "sample-1",
                "text": "退款",
                "label": "unknown-topic",
                "source": "synthetic",
            }
        )
        + "\n",
    ],
)
def test_training_dataset_loader_rejects_malformed_or_blank_records(
    tmp_path, content
):
    dataset = tmp_path / "invalid.jsonl"
    dataset.write_text(content, encoding="utf-8")

    with pytest.raises(ClassifierDataError, match="invalid training example"):
        load_training_dataset(dataset)


def test_training_dataset_loader_rejects_whitespace_only_text(tmp_path):
    dataset = tmp_path / "blank.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "example_id": "sample-blank",
                "text": "   ",
                "label": "other",
                "source": "synthetic",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ClassifierDataError, match="text cannot be blank"):
        load_training_dataset(dataset)


def test_training_dataset_loader_rejects_duplicate_ids_and_empty_files(tmp_path):
    dataset = tmp_path / "duplicate.jsonl"
    sample = {
        "example_id": "same-id",
        "text": "退款什么时候到账",
        "label": "refund_return",
        "source": "human_annotated",
    }
    dataset.write_text(
        json.dumps(sample, ensure_ascii=False)
        + "\n"
        + json.dumps(sample, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ClassifierDataError, match="duplicate example_id"):
        load_training_dataset(dataset)

    dataset.write_text("\n", encoding="utf-8")
    with pytest.raises(ClassifierDataError, match="cannot be empty"):
        load_training_dataset(dataset)
