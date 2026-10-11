import hashlib
import json

import pytest
from pydantic import ValidationError

from app.classifier_data import (
    ClassifierDataError,
    TrainingDataSource,
    TrainingDataSplit,
    TrainingExample,
    load_training_dataset,
    prepare_dataset,
    prepare_training_dataset,
)
from app.topic_taxonomy import TopicLabel


def make_example(example_id, text, labels, split=None):
    return TrainingExample(
        example_id=example_id,
        text=text,
        labels=labels,
        source=TrainingDataSource.HUMAN_ANNOTATED,
        split=split,
    )


def test_training_example_accepts_versioned_multilabel_provenance():
    example = make_example(
        "sample-1",
        "猫窝买大了想退",
        [TopicLabel.SIZE, TopicLabel.RETURN_EXCHANGE],
        TrainingDataSplit.TRAIN,
    )

    assert example.schema_version == 1
    assert example.labels == (TopicLabel.RETURN_EXCHANGE, TopicLabel.SIZE)
    assert example.model_dump(mode="json")["labels"] == ["退换货", "尺码"]
    assert example.split is TrainingDataSplit.TRAIN


def test_training_example_normalization_masks_personal_and_order_identifiers():
    example = make_example(
        "private-sample",
        "电话 13800138000，订单 123456789012，微信号 alice_12345",
        [TopicLabel.ORDER_MODIFICATION],
    )

    assert example.normalized_text() == "电话 [手机号]，订单 [单号]，微信[账号]"


@pytest.mark.parametrize("labels", [[], ["物流", "物流"], ["logistics"]])
def test_training_example_rejects_empty_duplicate_or_unknown_labels(labels):
    with pytest.raises(ValidationError):
        make_example("invalid", "查询物流", labels)


def test_training_example_rejects_extra_fields():
    with pytest.raises(ValidationError):
        TrainingExample(
            example_id="sample-2",
            text="物流查询",
            labels=[TopicLabel.LOGISTICS],
            source=TrainingDataSource.HUMAN_ANNOTATED,
            customer_phone="123456",
        )


def test_training_dataset_loader_reads_multilabel_jsonl_and_skips_blank_lines(tmp_path):
    dataset = tmp_path / "samples.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "example_id": "sample-1",
                "text": "猫窝买大了想退",
                "labels": ["尺码", "退换货"],
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
    assert examples[0].labels == (TopicLabel.RETURN_EXCHANGE, TopicLabel.SIZE)
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
                "labels": ["unknown-topic"],
                "source": "synthetic",
            }
        )
        + "\n",
    ],
)
def test_training_dataset_loader_rejects_malformed_records(tmp_path, content):
    dataset = tmp_path / "invalid.jsonl"
    dataset.write_text(content, encoding="utf-8")

    with pytest.raises(ClassifierDataError, match="invalid training example"):
        load_training_dataset(dataset)


def test_training_dataset_loader_rejects_blank_text_duplicate_ids_and_empty_files(tmp_path):
    dataset = tmp_path / "samples.jsonl"
    blank = {
        "example_id": "blank",
        "text": "   ",
        "labels": ["其他"],
        "source": "synthetic",
    }
    dataset.write_text(json.dumps(blank) + "\n", encoding="utf-8")
    with pytest.raises(ClassifierDataError, match="text cannot be blank"):
        load_training_dataset(dataset)

    duplicate = {
        "example_id": "same-id",
        "text": "退款什么时候到账",
        "labels": ["退换货"],
        "source": "human_annotated",
    }
    dataset.write_text(
        json.dumps(duplicate, ensure_ascii=False)
        + "\n"
        + json.dumps(duplicate, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ClassifierDataError, match="duplicate example_id"):
        load_training_dataset(dataset)

    dataset.write_text("\n", encoding="utf-8")
    with pytest.raises(ClassifierDataError, match="cannot be empty"):
        load_training_dataset(dataset)


def test_dataset_preparation_deduplicates_and_stratifies_label_combinations():
    examples = [
        make_example(f"size-return-{index}", f"尺寸不合适要求退换 {index}", [TopicLabel.SIZE, TopicLabel.RETURN_EXCHANGE])
        for index in range(10)
    ] + [
        make_example(f"logistics-{index}", f"查询配送进度 {index}", [TopicLabel.LOGISTICS])
        for index in range(10)
    ]
    examples.append(
        make_example("duplicate", "尺寸不合适要求退换 0", [TopicLabel.SIZE, TopicLabel.RETURN_EXCHANGE])
    )

    first = prepare_training_dataset(examples, seed=17)
    second = prepare_training_dataset(examples, seed=17)

    assert tuple(map(len, (first.train, first.validation, first.test))) == (16, 2, 2)
    assert [item.example_id for item in first.train] == [item.example_id for item in second.train]
    assert all(item.split is TrainingDataSplit.TRAIN for item in first.train)
    assert all(item.split is TrainingDataSplit.VALIDATION for item in first.validation)
    assert all(item.split is TrainingDataSplit.TEST for item in first.test)
    assert any(len(item.labels) == 2 for item in first.train)
    assert all(item.text == item.normalized_text() for item in first.train)


def test_dataset_preparation_keeps_tiny_label_combinations_in_training():
    examples = [
        make_example(f"tiny-{index}", f"少量咨询 {index}", [TopicLabel.OTHER])
        for index in range(2)
    ]

    splits = prepare_training_dataset(examples)

    assert len(splits.train) == 2
    assert not splits.validation
    assert not splits.test


def test_dataset_preparation_rejects_conflicting_duplicate_text_labels():
    examples = [
        make_example("sample-1", "订单到了吗", [TopicLabel.ORDER_MODIFICATION]),
        make_example("sample-2", " 订单到了吗 ", [TopicLabel.LOGISTICS]),
    ]

    with pytest.raises(ClassifierDataError, match="conflicting labels"):
        prepare_training_dataset(examples)


def test_dataset_preparation_rejects_cross_split_normalized_text_leakage():
    examples = [
        make_example("train-sample", "查一下订单", [TopicLabel.ORDER_MODIFICATION], TrainingDataSplit.TRAIN),
        make_example("test-sample", " 查一下订单 ", [TopicLabel.ORDER_MODIFICATION], TrainingDataSplit.TEST),
    ]

    with pytest.raises(ClassifierDataError, match="text appears in multiple splits"):
        prepare_training_dataset(examples)


def test_dataset_preparation_rejects_duplicate_ids_and_empty_input():
    example = make_example("same-id", "咨询物流", [TopicLabel.LOGISTICS])
    with pytest.raises(ClassifierDataError, match="duplicate example_id"):
        prepare_training_dataset([example, example.model_copy(update={"text": "另一个问法"})])
    with pytest.raises(ClassifierDataError, match="cannot be empty"):
        prepare_training_dataset([])


def test_prepare_dataset_writes_manifest_with_hash_and_multilabel_counts(tmp_path):
    source = tmp_path / "samples.jsonl"
    output_dir = tmp_path / "prepared"
    source.write_text(
        "\n".join(
            json.dumps(
                {
                    "example_id": f"manifest-{index}",
                    "text": f"分类样例 {index}",
                    "labels": ["尺码", "退换货"] if index < 5 else ["物流"],
                    "source": "human_annotated",
                },
                ensure_ascii=False,
            )
            for index in range(10)
        )
        + "\n",
        encoding="utf-8",
    )

    prepare_dataset(source, output_dir, seed=23)
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["input_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert manifest["total_examples"] == 10
    assert set(manifest["splits"]) == {"train", "validation", "test"}
    assert all(split["labels"]["物流"] > 0 for split in manifest["splits"].values())
    assert all(split["labels"]["尺码"] > 0 for split in manifest["splits"].values())
    assert "分类样例" not in manifest_path.read_text(encoding="utf-8")


def test_dataset_preparation_rejects_empty_input_file(tmp_path):
    source = tmp_path / "empty.jsonl"
    source.write_text("\n", encoding="utf-8")
    with pytest.raises(ClassifierDataError, match="cannot be empty"):
        prepare_dataset(source, tmp_path / "out")
