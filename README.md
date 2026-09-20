# CartPilot —— 电商客服智能助手

CartPilot 是一个面向电商客服场景的 AI Agent 服务，逐步提供多轮对话、订单协助、售后信息提取、知识库问答和业务工具编排能力。
项目采用可独立验证的功能切片开发，每个切片都有对应的代码、测试和可解释的 Git 提交。

## 当前进度

第一章“对话基础设施”已完成：

- 配置加载与服务健康检查；
- 对话请求、响应、错误和 Token 使用量契约；
- 内存会话创建、查询、消息追加和删除；
- 模型供应商协议与确定性的本地测试替身；
- Prompt 模板、变量渲染和版本注册；
- 非流式对话接口 `POST /chat`；
- SSE 流式对话接口 `POST /chat/stream`；
- 售后意图、订单号、原因和处理请求的结构化提取接口。

后续章节将继续实现 Function Calling 工具链、RAG 检索、Workflow + Agent、上下文管理、MCP 工具系统、可观测性和分类器推理。

## 本地启动

项目使用 Python 3.12+、`uv` 和 FastAPI。

```powershell
uv sync --dev
Copy-Item .env.example .env
uv run uvicorn app.main:app --reload
```

服务启动后可以访问：

- 健康检查：`GET http://127.0.0.1:8000/healthz`；
- 非流式对话：`POST http://127.0.0.1:8000/chat`；
- SSE 流式对话：`POST http://127.0.0.1:8000/chat/stream`；
- 售后信息提取：`POST http://127.0.0.1:8000/after-sales/extract`。

当前接口默认使用本地确定性模型替身，便于在没有外部模型密钥时运行测试。真实模型配置通过本地 `.env` 提供，不提交密钥。

## 测试

运行全部测试：

```powershell
uv run pytest
```

测试覆盖配置校验、会话隔离、Prompt 版本、模型供应商、普通对话、SSE 流式输出、售后结构化提取和第一章跨接口集成流程。

## 代码结构

| 路径 | 职责 |
| --- | --- |
| `app/config.py` | 环境变量和运行配置 |
| `app/contracts.py` | 请求、响应和领域数据模型 |
| `app/sessions.py` | 会话存储与生命周期管理 |
| `app/providers.py` | 模型供应商协议和本地替身 |
| `app/prompts.py` | Prompt 模板与版本注册 |
| `app/chat.py` | 对话服务和流式上下文 |
| `app/after_sales.py` | 售后信息结构化提取 |
| `tests/` | 单元测试、接口测试和集成测试 |

## 开发约定

- 每章使用独立 Git worktree 和章节分支；
- 每次有效对话最多产生一个真实 commit；
- 每个 commit 都必须对应实际实现、测试或已复现并修复的问题；
- 每次 commit 通过验证后立即推送到当前章节分支；
- 不创建空提交，不伪造日期，不虚构 bugfix，不提交 `.env`、密钥、缓存和训练产物；
- 只有实际运行并验证过的外部服务能力，才会被标记为已完成。

## 许可证

本项目代码和文档采用 MIT 许可证，详见 `LICENSE`。第三方依赖、模型、镜像和数据集遵循各自的许可证与使用条款。
