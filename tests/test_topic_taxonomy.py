import pytest
from pydantic import ValidationError

from app.classifier_data import TrainingDataSource, TrainingExample
from app.topic_taxonomy import ID2LABEL, LABEL2ID, TOPIC_CLASSES, TopicLabel


def test_topic_taxonomy_matches_the_seventeen_ordered_business_topics():
    expected = (
        "退换货", "物流", "尺码", "发票", "质量问题", "运费", "优惠活动",
        "价保", "支付", "订单修改", "库存补货", "商品信息", "保修维修",
        "账号", "会员积分", "评价", "其他",
    )
    assert tuple(item.label.value for item in TOPIC_CLASSES) == expected
    assert len({item.boundary for item in TOPIC_CLASSES}) == len(expected)
    assert all(item.examples for item in TOPIC_CLASSES)
    assert LABEL2ID == {label: index for index, label in enumerate(expected)}
    assert ID2LABEL == {index: label for index, label in enumerate(expected)}


def test_training_example_accepts_multiple_topic_labels():
    example = TrainingExample(
        example_id="multi-topic-1",
        text="猫窝买大了想退",
        labels=[TopicLabel.SIZE, TopicLabel.RETURN_EXCHANGE],
        source=TrainingDataSource.HUMAN_ANNOTATED,
    )
    assert example.model_dump(mode="json")["labels"] == ["退换货", "尺码"]


@pytest.mark.parametrize("labels", [[], ["物流", "物流"], ["logistics"]])
def test_training_example_rejects_empty_duplicate_or_non_topic_labels(labels):
    with pytest.raises(ValidationError):
        TrainingExample(
            example_id="invalid-labels",
            text="查询物流",
            labels=labels,
            source=TrainingDataSource.HUMAN_ANNOTATED,
        )
