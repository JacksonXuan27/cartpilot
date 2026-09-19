from collections.abc import Mapping
from dataclasses import dataclass
from string import Formatter

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PromptError(ValueError):
    pass


class PromptNotFoundError(PromptError):
    pass


class PromptAlreadyExistsError(PromptError):
    pass


class PromptRenderError(PromptError):
    pass


class PromptTemplate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    version: str = Field(pattern=r"^v?\d+\.\d+\.\d+$")
    content: str = Field(min_length=1)
    variables: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_variables(self) -> "PromptTemplate":
        if len(set(self.variables)) != len(self.variables):
            raise ValueError("prompt variables must be unique")

        placeholders = {
            field_name
            for _, field_name, _, _ in Formatter().parse(self.content)
            if field_name
        }
        declared = set(self.variables)
        if placeholders != declared:
            raise ValueError("prompt variables must match template placeholders")
        return self

    def render(self, values: Mapping[str, object]) -> str:
        missing = set(self.variables) - values.keys()
        if missing:
            missing_names = ", ".join(sorted(missing))
            raise PromptRenderError(f"missing prompt variables: {missing_names}")
        return self.content.format(**values)


@dataclass(slots=True)
class PromptRegistry:
    _templates: dict[tuple[str, str], PromptTemplate]

    @classmethod
    def with_defaults(cls) -> "PromptRegistry":
        registry = cls(_templates={})
        registry.register(
            PromptTemplate(
                name="customer-support-system",
                version="v1.0.0",
                content=(
                    "You are a helpful e-commerce customer-support assistant. "
                    "Answer in {language} and do not invent order information."
                ),
                variables=("language",),
            )
        )
        return registry

    def register(self, template: PromptTemplate) -> None:
        key = (template.name, template.version)
        if key in self._templates:
            raise PromptAlreadyExistsError(
                f"prompt already exists: {template.name}@{template.version}"
            )
        self._templates[key] = template

    def get(self, name: str, version: str | None = None) -> PromptTemplate:
        if version is not None:
            template = self._templates.get((name, version))
            if template is None:
                raise PromptNotFoundError(f"prompt not found: {name}@{version}")
            return template

        candidates = [
            template for (template_name, _), template in self._templates.items()
            if template_name == name
        ]
        if not candidates:
            raise PromptNotFoundError(f"prompt not found: {name}")
        return max(candidates, key=lambda template: _version_key(template.version))

    def render(
        self,
        name: str,
        values: Mapping[str, object],
        version: str | None = None,
    ) -> str:
        return self.get(name, version).render(values)


def _version_key(version: str) -> tuple[int, int, int]:
    return tuple(int(part) for part in version.lstrip("v").split("."))
