from collections.abc import Iterable, Mapping
from copy import deepcopy
from typing import Any, Protocol

from pydantic import BaseModel, ValidationError


class ToolNotFoundError(LookupError):
    pass


class ToolAlreadyRegisteredError(ValueError):
    pass


class ToolArgumentError(ValueError):
    def __init__(self, tool_name: str, error: ValidationError) -> None:
        self.tool_name = tool_name
        self.errors = error.errors()
        super().__init__(f"invalid arguments for tool: {tool_name}")


class ToolLike(Protocol):
    name: str
    description: str
    input_model: type[BaseModel]

    async def execute(self, arguments: Mapping[str, object]) -> Any:
        """Execute the tool with validated arguments."""


class ToolDefinition(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any]


class ToolRegistry:
    def __init__(self, tools: list[ToolLike] | None = None) -> None:
        self._tools: dict[str, ToolLike] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: ToolLike) -> None:
        self.register_many([tool])

    def register_many(self, tools: Iterable[ToolLike]) -> None:
        pending = list(tools)
        names = set(self._tools)
        additions: dict[str, ToolLike] = {}
        for tool in pending:
            if not isinstance(tool.name, str) or not tool.name.strip():
                raise ValueError("tool name cannot be empty")
            if tool.name in names:
                raise ToolAlreadyRegisteredError(tool.name)
            names.add(tool.name)
            additions[tool.name] = tool
        self._tools.update(additions)

    def get(self, name: str) -> ToolLike:
        tool = self._tools.get(name)
        if tool is None:
            raise ToolNotFoundError(name)
        return tool

    def definitions(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name=tool.name,
                description=tool.description,
                input_schema=deepcopy(tool.input_model.model_json_schema()),
            )
            for tool in sorted(self._tools.values(), key=lambda item: item.name)
        ]

    async def execute(self, name: str, arguments: Mapping[str, object]) -> Any:
        tool = self.get(name)
        try:
            validated = tool.input_model.model_validate(arguments)
        except ValidationError as exc:
            raise ToolArgumentError(name, exc) from exc
        return await tool.execute(validated.model_dump())
