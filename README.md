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
- 知识库问答：`POST http://127.0.0.1:8000/knowledge/query`。

当前接口默认使用本地确定性模型替身，便于在没有外部模型密钥时运行测试。真实模型配置通过本地 `.env` 提供，不提交密钥。`/knowledge/query` 使用本地规则改写查询，分别检索并通过 RRF 合并候选，再由本地 `LexicalReranker` 排序；配置 `RERANK_PROVIDER=siliconflow`、`RERANK_MODEL`、`RERANK_BASE_URL` 和 `RERANK_API_KEY` 后，会改用 SiliconFlow 重排，并按重排分数执行置信度判断。没有启用重排时，置信度判断仍使用向量相似度。重排配置独立于聊天模型凭据。

设置 `CHAT_PROVIDER=openai-compatible` 后，`/chat`、`/chat/stream`、售后信息提取和知识库问答使用 `CHAT_MODEL`、`CHAT_BASE_URL`、`CHAT_API_KEY` 指定的模型。未选择远程供应商时保持 stub，不会因填写了密钥而自动切换。选择远程供应商但缺少凭据时，应用直接报配置错误，不悄悄回退到本地替身。

语义向量设置为 `EMBEDDING_PROVIDER=openai-compatible`，使用 `EMBEDDING_MODEL`、`EMBEDDING_BASE_URL` 和 `EMBEDDING_API_KEY`（也兼容 `EMBED_API_KEY`）。默认模型 `BAAI/bge-m3` 对应 `EMBEDDING_DIMENSION=1024`；本地 Hash 替身默认 64 维。向量库维度必须与嵌入一致，不能复用不同维度的集合。未指定 `MILVUS_COLLECTION` 时按维度生成集合名。

设置 `MILVUS_URI` 后，知识问答连接 Milvus；未设置时使用内存向量库。应用直接在 Windows 运行时使用 `http://127.0.0.1:19530`，在 Compose 网络内使用 `http://milvus:19530`。Milvus 客户端在应用关闭时释放。启动并不等于已有知识数据：先向相同集合导入业务文档，再验收带来源的回答。

## Docker 服务

Docker Compose 会启动应用、Milvus Standalone、etcd 和 MinIO：

```powershell
Copy-Item .env.example .env
docker compose up --build
```

应用健康检查：`GET http://127.0.0.1:8000/healthz`；Milvus 健康检查由 Compose 自动执行。首次启动需要下载镜像，实际容器验收应在 Docker Desktop 引擎运行时进行。

在章节 worktree 开发且凭据保存在仓库根目录时，不需要复制密钥文件。从 `D:\code\cartpilot\.worktrees\ch04-hybrid-retrieval` 运行 `docker compose --env-file ..\..\.env up --build`，Compose 会将聊天、Embedding、Milvus 和重排配置传给该章节应用。不要从尚未集成这些改动的 `main` 目录启动并据此判断章节功能。

本地测试通过 HTTP 测试传输与 Milvus 边界替身验证装配、普通对话、SSE、查询改写、重排和客户端释放；这些测试不代表硅基流动、Docker 或真实 Milvus 已通过服务验收。

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
| `app/reranking.py` | 本地词法重排与 SiliconFlow 重排适配器 |
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
