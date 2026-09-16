# CartPilot

CartPilot is an e-commerce customer-support agent that combines conversational
answers, order assistance, policy knowledge, and guided after-sales workflows.
The system is being built as a sequence of independently testable milestones.

## Development status

The repository is currently in the foundation milestone. The first milestone
defines configuration loading and the service health contract before model
streaming, tools, retrieval, and workflow orchestration are added.

## Local setup

```powershell
uv sync --dev
Copy-Item .env.example .env
uv run uvicorn app.main:app --reload
```

The health endpoint is available at `http://127.0.0.1:8000/healthz`.

## Quality gates

```powershell
uv run pytest
```

Secrets stay in `.env` and are never committed. Runtime integrations are
introduced only after their contracts have a deterministic test.

