import json

import pytest

from app.classifier_data import TrainingDataSource, TrainingDataSplit, TrainingExample
from app.classifier_dataset import (
    ClassifierDatasetError,
    encode_labels,
    load_train_validation_splits,
    tokenize_examples,
)
from app.topic_taxonomy import NUM_TOPIC_LABELS, TopicLabel


class FixedTokenizer:
    def __call__(self, texts, *, truncation, padding, max_length):
        assert truncation is True
        assert padding == "max_length"
        return {
            "input_ids": [[index, 0] for index, _ in enumerate(texts)],
            "attention_mask": [[1, 0] for _ in texts],
            "token_type_ids": [[0, 0] for _ in texts],
        }


def write_split(path, split, examples):
    records = [
        {
            "example_id": example_id,
            "text": text,
            "labels": labels,
            "source": "human_annotated",
            "split": split,
        }
        for example_id, text, labels in examples
    ]
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records)
        + ("\n" if records else ""),
        encoding="utf-8",
    )


def test_multilabel_encoder_uses_taxonomy_order_and_binary_float_targets():
    encoded = encode_labels([TopicLabel.SIZE, TopicLabel.RETURN_EXCHANGE])

    assert len(encoded) == NUM_TOPIC_LABELS == 17
    assert sum(encoded) == 2.0
    assert encoded[0] == 1.0
    assert encoded[2] == 1.0
    assert all(value in (0.0, 1.0) for value in encoded)


@pytest.mark.parametrize(
    "labels",
    [[], ["not-a-topic"], [TopicLabel.LOGISTICS, TopicLabel.LOGISTICS]],
)
def test_multilabel_encoder_rejects_empty_unknown_and_duplicate_labels(labels):
    with pytest.raises(ClassifierDatasetError):
        encode_labels(labels)


def test_split_loader_requires_expected_split_and_rejects_text_leakage(tmp_path):
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    write_split(train_path, "train", [("train-1", "相同问题", ["物流"])])
    write_split(validation_path, "validation", [("validation-1", " 相同问题 ", ["物流"])])

    with pytest.raises(ClassifierDatasetError, match="overlap"):
        load_train_validation_splits(train_path, validation_path)


def test_split_loader_rejects_records_marked_for_another_split(tmp_path):
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    write_split(train_path, "validation", [("train-1", "查询物流", ["物流"])])
    write_split(validation_path, "validation", [("validation-1", "查询订单", ["订单修改"])])

    with pytest.raises(ClassifierDatasetError, match="expected train split"):
        load_train_validation_splits(train_path, validation_path)


def test_split_loader_rejects_empty_training_or_validation_files(tmp_path):
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    write_split(train_path, "train", [])
    write_split(validation_path, "validation", [("validation-1", "问物流", ["物流"])])

    with pytest.raises(ClassifierDatasetError, match="train split cannot be empty"):
        load_train_validation_splits(train_path, validation_path)


def test_tokenization_builds_model_inputs_and_multihot_targets():
    examples = [
        TrainingExample(
            example_id="sample-1",
            text="猫窝买大了想退",
            labels=[TopicLabel.SIZE, TopicLabel.RETURN_EXCHANGE],
            source=TrainingDataSource.HUMAN_ANNOTATED,
            split=TrainingDataSplit.TRAIN,
        )
    ]

    encoded = tokenize_examples(examples, FixedTokenizer(), max_length=32)

    assert encoded == [
        {
            "input_ids": [0, 0],
            "attention_mask": [1, 0],
            "token_type_ids": [0, 0],
            "labels": [
                1.0 if index in (0, 2) else 0.0
                for index in range(NUM_TOPIC_LABELS)
            ],
        }
    ]


def test_tokenization_accepts_tokenizers_without_token_type_ids():
    class TokenizerWithoutSegments:
        def __call__(self, texts, **kwargs):
            return {"input_ids": [[1] for _ in texts], "attention_mask": [[1] for _ in texts]}

    example = TrainingExample(
        example_id="sample-no-segments",
        text="物流在哪",
        labels=[TopicLabel.LOGISTICS],
        source=TrainingDataSource.HUMAN_ANNOTATED,
    )

    encoded = tokenize_examples([example], TokenizerWithoutSegments(), max_length=16)

    assert set(encoded[0]) == {"input_ids", "attention_mask", "labels"}


def test_tokenization_rejects_incomplete_tokenizer_output():
    class IncompleteTokenizer:
        def __call__(self, texts, **kwargs):
            return {"input_ids": []}

    example = TrainingExample(
        example_id="sample-invalid-tokenizer",
        text="物流在哪",
        labels=[TopicLabel.LOGISTICS],
        source=TrainingDataSource.HUMAN_ANNOTATED,
    )

    with pytest.raises(ClassifierDatasetError, match="invalid input_ids"):
        tokenize_examples([example], IncompleteTokenizer(), max_length=16)
