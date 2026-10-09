import json
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
        return " ".join(self.text.split())


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
