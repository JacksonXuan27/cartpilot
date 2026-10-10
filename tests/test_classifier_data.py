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
    prepare_dataset,
    prepare_training_dataset,
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


def test_training_example_normalization_masks_personal_and_order_identifiers():
    example = TrainingExample(
        example_id="private-sample",
        text="电话 13800138000，订单 123456789012，微信号 alice_12345",
        label=IntentCategory.ORDER,
        source=TrainingDataSource.HUMAN_ANNOTATED,
    )

    assert example.normalized_text() == "电话 [手机号]，订单 [单号]，微信[账号]"


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


def test_dataset_preparation_normalizes_deduplicates_and_stratifies_reproducibly():
    examples = [
        TrainingExample(
            example_id=f"logistics-{index}",
            text=f"  查询物流进度 {index}  ",
            label=IntentCategory.LOGISTICS,
            source=TrainingDataSource.HUMAN_ANNOTATED,
        )
        for index in range(10)
    ] + [
        TrainingExample(
            example_id=f"order-{index}",
            text=f"  查询订单状态 {index}  ",
            label=IntentCategory.ORDER,
            source=TrainingDataSource.REVIEWED_FEEDBACK,
        )
        for index in range(10)
    ]
    examples.append(
        TrainingExample(
            example_id="logistics-duplicate",
            text="查询物流进度 0",
            label=IntentCategory.LOGISTICS,
            source=TrainingDataSource.HUMAN_ANNOTATED,
        )
    )

    first = prepare_training_dataset(examples, seed=17)
    second = prepare_training_dataset(examples, seed=17)

    assert len(first.train) == 16
    assert len(first.validation) == 2
    assert len(first.test) == 2
    assert [item.example_id for item in first.train] == [
        item.example_id for item in second.train
    ]
    assert all(item.text == item.normalized_text() for item in first.train)
    assert all(item.split is TrainingDataSplit.TRAIN for item in first.train)
    assert all(item.split is TrainingDataSplit.VALIDATION for item in first.validation)
    assert all(item.split is TrainingDataSplit.TEST for item in first.test)
    assert {
        item.label for item in first.train
    } == {IntentCategory.LOGISTICS, IntentCategory.ORDER}


def test_dataset_preparation_keeps_tiny_classes_in_training():
    examples = [
        TrainingExample(
            example_id=f"sample-{index}",
            text=f"咨询内容 {index}",
            label=IntentCategory.COMPLAINT,
            source=TrainingDataSource.SYNTHETIC,
        )
        for index in range(2)
    ]

    splits = prepare_training_dataset(examples)

    assert len(splits.train) == 2
    assert not splits.validation
    assert not splits.test


def test_dataset_preparation_rejects_conflicting_duplicate_text_labels():
    examples = [
        TrainingExample(
            example_id="sample-1",
            text="订单到了吗",
            label=IntentCategory.ORDER,
            source=TrainingDataSource.HUMAN_ANNOTATED,
        ),
        TrainingExample(
            example_id="sample-2",
            text=" 订单到了吗 ",
            label=IntentCategory.LOGISTICS,
            source=TrainingDataSource.HUMAN_ANNOTATED,
        ),
    ]

    with pytest.raises(ClassifierDataError, match="conflicting labels"):
        prepare_training_dataset(examples)


def test_dataset_preparation_rejects_cross_split_text_leakage():
    examples = [
        TrainingExample(
            example_id="train-sample",
            text="查一下订单",
            label=IntentCategory.ORDER,
            source=TrainingDataSource.HUMAN_ANNOTATED,
            split=TrainingDataSplit.TRAIN,
        ),
        TrainingExample(
            example_id="test-sample",
            text=" 查一下订单 ",
            label=IntentCategory.ORDER,
            source=TrainingDataSource.HUMAN_ANNOTATED,
            split=TrainingDataSplit.TEST,
        ),
    ]

    with pytest.raises(ClassifierDataError, match="text appears in multiple splits"):
        prepare_training_dataset(examples)


def test_prepare_dataset_writes_reproducible_utf8_jsonl_outputs(tmp_path):
    source = tmp_path / "input.jsonl"
    output_dir = tmp_path / "prepared"
    source.write_text(
        "\n".join(
            json.dumps(
                {
                    "example_id": f"sample-{index}",
                    "text": f"  查询物流进度 {index}  ",
                    "label": "logistics",
                    "source": "human_annotated",
                },
                ensure_ascii=False,
            )
            for index in range(10)
        )
        + "\n",
        encoding="utf-8",
    )

    splits = prepare_dataset(source, output_dir, seed=9)

    assert len(splits.train) + len(splits.validation) + len(splits.test) == 10
    assert all(
        (output_dir / f"{split}.jsonl").exists()
        for split in ("train", "validation", "test")
    )
    test_examples = load_training_dataset(output_dir / "test.jsonl")
    assert all(item.split is TrainingDataSplit.TEST for item in test_examples)
    assert "查询物流进度" in (output_dir / "train.jsonl").read_text(encoding="utf-8")


def test_dataset_preparation_rejects_duplicate_ids_and_empty_input():
    example = TrainingExample(
        example_id="same-id",
        text="咨询物流",
        label=IntentCategory.LOGISTICS,
        source=TrainingDataSource.HUMAN_ANNOTATED,
    )

    with pytest.raises(ClassifierDataError, match="duplicate example_id"):
        prepare_training_dataset(
            [example, example.model_copy(update={"text": "咨询订单"})]
        )

    with pytest.raises(ClassifierDataError, match="cannot be empty"):
        prepare_training_dataset([])


def test_prepare_dataset_writes_reproducible_utf8_jsonl_outputs(tmp_path):
    source = tmp_path / "input.jsonl"
    output_dir = tmp_path / "prepared"
    source.write_text(
        "\n".join(
            json.dumps(
                {
                    "example_id": f"sample-{index}",
                    "text": f"  查询物流进度 {index}  ",
                    "label": "logistics",
                    "source": "human_annotated",
                },
                ensure_ascii=False,
            )
            for index in range(10)
        )
        + "\n",
        encoding="utf-8",
    )

    splits = prepare_dataset(source, output_dir, seed=9)

    assert len(splits.train) + len(splits.validation) + len(splits.test) == 10
    assert all(
        (output_dir / f"{split}.jsonl").exists()
        for split in ("train", "validation", "test")
    )
    test_examples = load_training_dataset(output_dir / "test.jsonl")
    assert all(item.split is TrainingDataSplit.TEST for item in test_examples)
    assert "查询物流进度" in (output_dir / "train.jsonl").read_text(encoding="utf-8")


def test_dataset_preparation_rejects_duplicate_ids_and_empty_input():
    example = TrainingExample(
        example_id="same-id",
        text="咨询物流",
        label=IntentCategory.LOGISTICS,
        source=TrainingDataSource.HUMAN_ANNOTATED,
    )

    with pytest.raises(ClassifierDataError, match="duplicate example_id"):
        prepare_training_dataset(
            [example, example.model_copy(update={"text": "咨询订单"})]
        )

    with pytest.raises(ClassifierDataError, match="cannot be empty"):
        prepare_training_dataset([])
