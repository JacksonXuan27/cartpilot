# 检索离线评估

`data/evaluation/ch04/retrieval_qrels.jsonl` 是用于验证评估流程的少量人工示例标注。它不是完整业务语料，也不能单独用于宣称检索效果。

## 输入格式

标注集采用 JSONL，每行包含 `query_id`、`query` 和相关片段 ID 列表 `relevant_record_ids`：

```json
{"query_id":"q01","query":"退款多久到账","relevant_record_ids":["refund-policy-01"]}
```

排名文件也是 JSONL，每行包含 `query_id` 和按相关度降序排列的 `ranked_record_ids`：

```json
{"query_id":"q01","ranked_record_ids":["refund-policy-01","refund-policy-02"]}
```

排名文件可以缺少某些 query；缺失项按空结果处理。重复 ID、未知 query ID 和不完整标注会报错。

## 运行

先由待测检索器生成排名文件，再运行：

```powershell
uv run python -m app.retrieval_evaluation `
  --dataset data/evaluation/ch04/retrieval_qrels.jsonl `
  --rankings data/evaluation/ch04/rankings.jsonl `
  --output data/evaluation/ch04/reports/retrieval-report.md `
  --cutoffs 1,3,5
```

报告输出 Recall@K、Precision@K、MRR@K、nDCG@K 的逐查询宏平均以及逐查询 Recall/MRR。报告仅描述输入排名在当前标注集上的离线结果，不代表在线 SLA、回答正确率或更大业务数据集上的表现。
