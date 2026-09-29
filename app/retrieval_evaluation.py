import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


class RetrievalEvaluationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    query_id: str
    query: str
    relevant_record_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.query_id, str)
            or not isinstance(self.query, str)
            or not self.query_id.strip()
            or not self.query.strip()
        ):
            raise RetrievalEvaluationError("query_id and query cannot be empty")
        if not self.relevant_record_ids:
            raise RetrievalEvaluationError("each query must have relevant records")
        if any(not isinstance(record_id, str) for record_id in self.relevant_record_ids):
            raise RetrievalEvaluationError("relevant record IDs must be strings")
        if len(set(self.relevant_record_ids)) != len(self.relevant_record_ids):
            raise RetrievalEvaluationError("relevant record IDs must be unique")
        if any(not record_id.strip() for record_id in self.relevant_record_ids):
            raise RetrievalEvaluationError("relevant record IDs cannot be empty")


@dataclass(frozen=True, slots=True)
class QueryEvaluation:
    query_id: str
    query: str
    metrics: dict[int, dict[str, float]]


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    query_count: int
    cutoffs: tuple[int, ...]
    aggregate: dict[int, dict[str, float]]
    per_query: tuple[QueryEvaluation, ...]


METRIC_NAMES = ("recall", "precision", "mrr", "ndcg")


def load_evaluation_dataset(path: str | Path) -> list[EvaluationCase]:
    cases: list[EvaluationCase] = []
    seen_query_ids: set[str] = set()
    for line_number, item in _read_jsonl(path):
        try:
            relevant_record_ids = item["relevant_record_ids"]
            if not isinstance(relevant_record_ids, list):
                raise TypeError("relevant_record_ids must be an array")
            case = EvaluationCase(
                query_id=item["query_id"],
                query=item["query"],
                relevant_record_ids=tuple(relevant_record_ids),
            )
        except (KeyError, TypeError) as exc:
            raise RetrievalEvaluationError(
                f"invalid evaluation case at line {line_number}"
            ) from exc
        if case.query_id in seen_query_ids:
            raise RetrievalEvaluationError(
                f"duplicate query_id at line {line_number}: {case.query_id}"
            )
        seen_query_ids.add(case.query_id)
        cases.append(case)
    if not cases:
        raise RetrievalEvaluationError("evaluation dataset cannot be empty")
    return cases


def load_rankings(path: str | Path) -> dict[str, tuple[str, ...]]:
    rankings: dict[str, tuple[str, ...]] = {}
    for line_number, item in _read_jsonl(path):
        try:
            query_id = item["query_id"]
            raw_record_ids = item["ranked_record_ids"]
            if not isinstance(raw_record_ids, list):
                raise TypeError("ranked_record_ids must be an array")
            record_ids = tuple(raw_record_ids)
        except (KeyError, TypeError) as exc:
            raise RetrievalEvaluationError(
                f"invalid ranking at line {line_number}"
            ) from exc
        if not isinstance(query_id, str) or not query_id.strip():
            raise RetrievalEvaluationError(
                f"query_id cannot be empty at line {line_number}"
            )
        if any(not isinstance(record_id, str) or not record_id.strip() for record_id in record_ids):
            raise RetrievalEvaluationError(
                f"ranked record IDs cannot be empty at line {line_number}"
            )
        if len(set(record_ids)) != len(record_ids):
            raise RetrievalEvaluationError(
                f"ranked record IDs must be unique at line {line_number}"
            )
        if query_id in rankings:
            raise RetrievalEvaluationError(
                f"duplicate query_id at line {line_number}: {query_id}"
            )
        rankings[query_id] = record_ids
    return rankings


def evaluate_retrieval(
    cases: Sequence[EvaluationCase],
    rankings: dict[str, Sequence[str]],
    cutoffs: Sequence[int] = (1, 3, 5),
) -> EvaluationReport:
    if not cases:
        raise RetrievalEvaluationError("evaluation dataset cannot be empty")
    if not cutoffs or any(cutoff < 1 for cutoff in cutoffs):
        raise RetrievalEvaluationError("cutoffs must contain positive integers")
    if len(set(cutoffs)) != len(cutoffs):
        raise RetrievalEvaluationError("cutoffs must be unique")

    query_ids = [case.query_id for case in cases]
    if len(set(query_ids)) != len(query_ids):
        raise RetrievalEvaluationError("evaluation query IDs must be unique")
    unknown_query_ids = set(rankings) - set(query_ids)
    if unknown_query_ids:
        raise RetrievalEvaluationError(
            f"rankings contain unknown query IDs: {sorted(unknown_query_ids)}"
        )

    per_query: list[QueryEvaluation] = []
    totals = {cutoff: {name: 0.0 for name in METRIC_NAMES} for cutoff in cutoffs}
    for case in cases:
        ranked_ids = tuple(rankings.get(case.query_id, ()))
        if len(set(ranked_ids)) != len(ranked_ids):
            raise RetrievalEvaluationError(
                f"ranked record IDs must be unique for query {case.query_id}"
            )
        relevant_ids = set(case.relevant_record_ids)
        metrics_by_cutoff: dict[int, dict[str, float]] = {}
        for cutoff in cutoffs:
            metrics = _query_metrics(ranked_ids, relevant_ids, cutoff)
            metrics_by_cutoff[cutoff] = metrics
            for name, value in metrics.items():
                totals[cutoff][name] += value
        per_query.append(
            QueryEvaluation(case.query_id, case.query, metrics_by_cutoff)
        )

    query_count = len(cases)
    aggregate = {
        cutoff: {
            name: total / query_count for name, total in metric_totals.items()
        }
        for cutoff, metric_totals in totals.items()
    }
    return EvaluationReport(
        query_count=query_count,
        cutoffs=tuple(cutoffs),
        aggregate=aggregate,
        per_query=tuple(per_query),
    )


def render_markdown_report(
    report: EvaluationReport, dataset_path: str | Path, rankings_path: str | Path
) -> str:
    lines = [
        "# 检索离线评估报告",
        "",
        f"- 评估查询数：{report.query_count}",
        f"- 标注集：`{dataset_path}`",
        f"- 排名输入：`{rankings_path}`",
        "- 统计口径：逐查询计算后进行宏平均；缺失排名按空结果处理。",
        "- 适用范围：仅代表本次输入的离线排名，不代表在线 SLA 或端到端回答质量。",
        "",
        "## 汇总指标",
        "",
        "| K | Recall@K | Precision@K | MRR@K | nDCG@K |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for cutoff in report.cutoffs:
        metrics = report.aggregate[cutoff]
        lines.append(
            f"| {cutoff} | {metrics['recall']:.4f} | {metrics['precision']:.4f} "
            f"| {metrics['mrr']:.4f} | {metrics['ndcg']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 逐查询结果",
            "",
            "| Query ID | Query | "
            + " | ".join(
                f"Recall@{cutoff} / MRR@{cutoff}" for cutoff in report.cutoffs
            )
            + " |",
            "| --- | --- | " + " | ".join("---" for _ in report.cutoffs) + " |",
        ]
    )
    for result in report.per_query:
        values = " | ".join(
            f"{result.metrics[cutoff]['recall']:.4f} / "
            f"{result.metrics[cutoff]['mrr']:.4f}"
            for cutoff in report.cutoffs
        )
        safe_query = result.query.replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {result.query_id} | {safe_query} | {values} |")
    return "\n".join(lines) + "\n"


def _query_metrics(
    ranked_ids: Sequence[str], relevant_ids: set[str], cutoff: int
) -> dict[str, float]:
    top_ids = ranked_ids[:cutoff]
    relevant_ranks = [rank for rank, record_id in enumerate(top_ids, 1) if record_id in relevant_ids]
    recall = len(relevant_ranks) / len(relevant_ids)
    precision = len(relevant_ranks) / cutoff
    mrr = 1 / relevant_ranks[0] if relevant_ranks else 0.0
    dcg = sum(1 / math.log2(rank + 1) for rank in relevant_ranks)
    ideal_relevant_count = min(len(relevant_ids), cutoff)
    ideal_dcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_relevant_count + 1))
    ndcg = dcg / ideal_dcg if ideal_dcg else 0.0
    return {"recall": recall, "precision": precision, "mrr": mrr, "ndcg": ndcg}


def _read_jsonl(path: str | Path) -> list[tuple[int, dict[str, object]]]:
    records: list[tuple[int, dict[str, object]]] = []
    try:
        with Path(path).open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, 1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RetrievalEvaluationError(
                        f"invalid JSON at line {line_number} in {path}"
                    ) from exc
                if not isinstance(item, dict):
                    raise RetrievalEvaluationError(
                        f"each JSONL record must be an object at line {line_number}"
                    )
                records.append((line_number, item))
    except OSError as exc:
        raise RetrievalEvaluationError(f"cannot read {path}: {exc}") from exc
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate offline retrieval rankings")
    parser.add_argument("--dataset", required=True, help="JSONL relevance labels")
    parser.add_argument("--rankings", required=True, help="JSONL ranked record IDs")
    parser.add_argument("--output", required=True, help="Markdown report path")
    parser.add_argument("--cutoffs", default="1,3,5", help="comma-separated K values")
    arguments = parser.parse_args()
    try:
        cutoffs = tuple(int(value.strip()) for value in arguments.cutoffs.split(","))
        cases = load_evaluation_dataset(arguments.dataset)
        rankings = load_rankings(arguments.rankings)
        report = evaluate_retrieval(cases, rankings, cutoffs)
        rendered = render_markdown_report(report, arguments.dataset, arguments.rankings)
        output_path = Path(arguments.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")
    except (RetrievalEvaluationError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Wrote {arguments.output} ({report.query_count} queries)")


if __name__ == "__main__":
    main()
