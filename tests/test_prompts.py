import pytest
from pydantic import ValidationError

from app.prompts import (
    PromptAlreadyExistsError,
    PromptNotFoundError,
    PromptRegistry,
    PromptRenderError,
    PromptTemplate,
)


def test_prompt_template_renders_declared_variables():
    prompt = PromptTemplate(
        name="greeting",
        version="v1.0.0",
        content="Say hello in {language}.",
        variables=("language",),
    )

    assert prompt.render({"language": "Chinese"}) == "Say hello in Chinese."


def test_prompt_template_requires_placeholders_and_declared_variables_to_match():
    with pytest.raises(ValidationError):
        PromptTemplate(
            name="greeting",
            version="v1.0.0",
            content="Say hello in {language}.",
            variables=(),
        )

    with pytest.raises(ValidationError):
        PromptTemplate(
            name="greeting",
            version="v1.0.0",
            content="Say hello.",
            variables=("language",),
        )


def test_registry_returns_an_explicit_version_or_the_latest_version():
    registry = PromptRegistry(_templates={})
    registry.register(
        PromptTemplate(name="greeting", version="v1.0.0", content="hello")
    )
    registry.register(
        PromptTemplate(name="greeting", version="v1.2.0", content="hello again")
    )

    assert registry.get("greeting").version == "v1.2.0"
    assert registry.get("greeting", "v1.0.0").content == "hello"


def test_registry_rejects_duplicate_versions_and_missing_prompts():
    registry = PromptRegistry(_templates={})
    prompt = PromptTemplate(name="greeting", version="v1.0.0", content="hello")
    registry.register(prompt)

    with pytest.raises(PromptAlreadyExistsError):
        registry.register(prompt)

    with pytest.raises(PromptNotFoundError):
        registry.get("unknown")


def test_render_reports_missing_values_and_default_prompt_is_available():
    registry = PromptRegistry.with_defaults()

    with pytest.raises(PromptRenderError, match="language"):
        registry.render("customer-support-system", {})

    assert "Chinese" in registry.render(
        "customer-support-system", {"language": "Chinese"}
    )
