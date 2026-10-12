import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


class ExportConfigError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ExportConfig:
    model_dir: Path
    output_dir: Path
    opset: int = 17
    sample_texts: tuple[str, ...] = ("买大了想退", "快递到哪了")
    max_length: int = 128

    def __post_init__(self) -> None:
        if not str(self.model_dir).strip():
            raise ExportConfigError("model_dir cannot be empty")
        if not str(self.output_dir).strip():
            raise ExportConfigError("output_dir cannot be empty")
        if self.opset < 1:
            raise ExportConfigError("opset must be positive")
        if not self.sample_texts or any(not text.strip() for text in self.sample_texts):
            raise ExportConfigError("sample_texts must contain non-empty text")
        if self.max_length < 2:
            raise ExportConfigError("max_length must be at least 2")
        object.__setattr__(self, "model_dir", Path(self.model_dir))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "sample_texts", tuple(self.sample_texts))


def export_onnx_model(config: ExportConfig) -> dict[str, Any]:
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise ExportConfigError(
            "ONNX export requires the optional ML dependency group "
            "(torch, transformers, onnx)"
        ) from exc

    if not config.model_dir.is_dir():
        raise ExportConfigError(f"model_dir does not exist: {config.model_dir}")
    threshold_path = config.model_dir / "threshold.json"
    if not threshold_path.is_file():
        raise ExportConfigError(f"threshold.json is missing from {config.model_dir}")
    try:
        threshold_data = json.loads(threshold_path.read_text(encoding="utf-8"))
        threshold = float(threshold_data["threshold"])
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise ExportConfigError("threshold.json must contain a numeric threshold") from exc
    if not 0 <= threshold <= 1:
        raise ExportConfigError("threshold must be between 0 and 1")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(str(config.model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(config.model_dir))
    model.eval()
    encoded = tokenizer(
        list(config.sample_texts),
        truncation=True,
        padding=True,
        max_length=config.max_length,
        return_tensors="pt",
    )
    input_names = [name for name in ("input_ids", "attention_mask", "token_type_ids") if name in encoded]
    if "input_ids" not in input_names:
        raise ExportConfigError("tokenizer did not produce input_ids")
    example_inputs = tuple(encoded[name] for name in input_names)

    class LogitsWrapper(torch.nn.Module):
        def __init__(self, wrapped_model: Any, names: list[str]) -> None:
            super().__init__()
            self.wrapped_model = wrapped_model
            self.names = names

        def forward(self, *inputs: Any) -> Any:
            return self.wrapped_model(**dict(zip(self.names, inputs))).logits

    wrapper = LogitsWrapper(model, input_names)
    onnx_path = config.output_dir / "model.onnx"
    dynamic_axes = {
        name: {0: "batch", 1: "sequence"} for name in input_names
    }
    dynamic_axes["logits"] = {0: "batch"}
    torch.onnx.export(
        wrapper,
        example_inputs,
        str(onnx_path),
        input_names=input_names,
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        dynamo=False,
        opset_version=config.opset,
    )
    tokenizer.save_pretrained(config.output_dir)
    (config.output_dir / "threshold.json").write_bytes(threshold_path.read_bytes())
    manifest = {
        "onnx_path": str(onnx_path),
        "input_names": input_names,
        "output_names": ["logits"],
        "opset": config.opset,
        "sample_count": len(config.sample_texts),
        "max_length": config.max_length,
        "onnx_bytes": onnx_path.stat().st_size,
    }
    (config.output_dir / "export_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a trained topic classifier to ONNX")
    parser.add_argument("--model-dir", type=Path, default=Path("data/ch10/model"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/ch10/onnx"))
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--max-length", type=int, default=128)
    arguments = parser.parse_args()
    result = export_onnx_model(
        ExportConfig(
            model_dir=arguments.model_dir,
            output_dir=arguments.output_dir,
            opset=arguments.opset,
            max_length=arguments.max_length,
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
