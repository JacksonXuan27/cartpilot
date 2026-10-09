# CartPilot —— 电商客服智能助手

CartPilot 是一个面向电商客服场景的 AI Agent 服务，逐步提供多轮对话、订单协助、售后信息提取、知识库问答和业务工具编排能力。
项目采用可独立验证的功能切片开发，每个切片都有对应的代码、测试和可解释的 Git 提交。

## 当前进度

当前已具备以下能力：

- 配置加载与服务健康检查；
- 对话请求、响应、错误和 Token 使用量契约；
- 内存会话创建、查询、消息追加和删除；
- 模型供应商协议与确定性的本地测试替身；
- Prompt 模板、变量渲染和版本注册；
- 非流式对话接口 `POST /chat`；
- SSE 流式对话接口 `POST /chat/stream`；
- 售后意图、订单号、原因和处理请求的结构化提取接口。
- 订单、知识文档和检索记录的数据模型与 SQLite Repository；
- 文档清洗、切块、Embedding 抽象和本地确定性向量替身；
- Milvus 向量存储适配器与知识库问答接口 `POST /knowledge/query`；
- Docker Compose 下的应用、Milvus、etcd 和 MinIO 服务编排。
- Workflow Trace/Span、模型 Token、耗时和估算成本统计；
- 可选的后台 HTTP 观测导出，默认只发送观测元数据，不发送对话正文；
- 用户对回答的有用/无用反馈通过 SQLite 持久化，并关联请求与 Trace 标识。

主题分类采用与 Workflow 意图路由一致的九类标签：物流、订单、商品、退款退货、售后、投诉、人工、寒暄和其他。训练样本使用 UTF-8 JSONL，每行包含版本、样本 ID、文本、标签、来源及可选数据集拆分；当前仅定义格式和校验，不代表已构建或训练真实数据集。

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
- 知识库问答：`POST http://127.0.0.1:8000/knowledge/query`；
- 用户反馈：`POST http://127.0.0.1:8000/feedback`，请求体示例：

```json
{
  "request_id": "知识问答返回的 request_id",
  "trace_id": "可选的 Trace 标识",
  "rating": "unhelpful",
  "reason": "检索内容不相关"
}
```

审核反馈可通过 POST /feedback/{feedback_id}/review 写回审核状态、备注和审核人；GET /feedback/export 按 JSONL 导出反馈及审核元数据，可使用 rating、review_status 和 limit 筛选。导出不包含对话正文。

默认使用本地确定性模型替身。配置 `.env` 中的 `CHAT_PROVIDER=openai-compatible`、`CHAT_BASE_URL`、`CHAT_MODEL` 和 `CHAT_API_KEY` 后，普通对话、知识问答及售后信息提取会调用配置的 OpenAI API 规范兼容接口；不支持非标准扩展响应字段。真实模型可在 Workflow ReAct 循环中请求已注册的工具，并将调用结果回传模型；默认运行时尚未装配业务工具，实际可调用范围由运行时注册表决定。密钥不提交到 Git。

如需接入自建观测服务或 Langfuse 兼容网关，可在 `.env` 中配置 `OBSERVABILITY_EXPORT_URL`，并按需设置 `OBSERVABILITY_API_KEY`。导出在后台队列中执行，观测服务不可用时不会阻塞对话和 Workflow 主链路；未配置端点时继续使用内存记录器。

## Docker 服务

Docker Compose 会启动应用、Milvus Standalone、etcd 和 MinIO：

```powershell
Copy-Item .env.example .env
docker compose up --build
```

应用健康检查：`GET http://127.0.0.1:8000/healthz`；Milvus 健康检查由 Compose 自动执行。首次启动需要下载镜像，实际容器验收应在 Docker Desktop 引擎运行时进行。

## 测试

运行全部测试：

```powershell
uv run pytest
```

测试覆盖配置校验、会话隔离、Prompt 版本、模型供应商、普通对话、SSE 流式输出、售后结构化提取、数据层、向量检索和 RAG HTTP 冒烟流程。

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
| `app/data_models.py` | 持久化与检索数据模型 |
| `app/repositories.py` | SQLite Repository 与表初始化 |
| `app/document_processing.py` | 文档清洗和文本切块 |
| `app/embeddings.py` | Embedding 接口和本地替身 |
| `app/vector_store.py` | 内存向量库和 Milvus 适配器 |
| `app/knowledge_base.py` | 知识库检索与问答服务 |
| `app/observability.py` | Trace、模型指标和可选 HTTP 导出 |
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
