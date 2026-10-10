from dataclasses import dataclass
from enum import Enum
from math import isfinite
from pathlib import Path


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
