import json

import pytest

from app.retrieval_evaluation import (
    EvaluationCase,
    RetrievalEvaluationError,
    evaluate_retrieval,
    load_evaluation_dataset,
    load_rankings,
    render_markdown_report,
)


def test_evaluator_calculates_macro_recall_precision_mrr_and_ndcg():
    cases = [
        EvaluationCase("q1", "退款政策", ("relevant-a", "relevant-b")),
        EvaluationCase("q2", "物流", ("relevant-c",)),
    ]
    rankings = {
        "q1": ("irrelevant", "relevant-a", "relevant-b"),
        "q2": ("relevant-c", "other"),
    }

    report = evaluate_retrieval(cases, rankings, cutoffs=(1, 3))

    assert report.query_count == 2
    assert report.aggregate[1]["recall"] == pytest.approx(0.5)
    assert report.aggregate[1]["precision"] == pytest.approx(0.5)
    assert report.aggregate[1]["mrr"] == pytest.approx(0.5)
    assert report.aggregate[3]["recall"] == pytest.approx(1.0)
    assert report.aggregate[3]["precision"] == pytest.approx(0.5)
    assert report.aggregate[3]["mrr"] == pytest.approx(0.75)
    assert report.aggregate[3]["ndcg"] < 1.0


def test_missing_query_ranking_counts_as_empty_results():
    case = EvaluationCase("q1", "退款政策", ("refund",))

    report = evaluate_retrieval([case], {}, cutoffs=(1,))

    assert report.aggregate[1] == {
        "recall": 0.0,
        "precision": 0.0,
        "mrr": 0.0,
        "ndcg": 0.0,
    }


def test_evaluator_rejects_unknown_queries_duplicate_records_and_bad_cutoffs():
    case = EvaluationCase("q1", "退款政策", ("refund",))

    with pytest.raises(RetrievalEvaluationError, match="unknown query"):
        evaluate_retrieval([case], {"q2": ("refund",)})
    with pytest.raises(RetrievalEvaluationError, match="unique"):
        evaluate_retrieval([case], {"q1": ("refund", "refund")})
    with pytest.raises(RetrievalEvaluationError, match="positive"):
        evaluate_retrieval([case], {}, cutoffs=(0,))


def test_dataset_and_ranking_jsonl_loaders_validate_records(tmp_path):
    dataset_path = tmp_path / "qrels.jsonl"
    rankings_path = tmp_path / "rankings.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "query_id": "q1",
                "query": "退款政策",
                "relevant_record_ids": ["refund"],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    rankings_path.write_text(
        json.dumps({"query_id": "q1", "ranked_record_ids": ["refund"]}) + "\n",
        encoding="utf-8",
    )

    cases = load_evaluation_dataset(dataset_path)
    rankings = load_rankings(rankings_path)

    assert cases[0].relevant_record_ids == ("refund",)
    assert rankings["q1"] == ("refund",)


def test_jsonl_loaders_reject_non_array_record_ids(tmp_path):
    dataset_path = tmp_path / "bad-qrels.jsonl"
    ranking_path = tmp_path / "bad-rankings.jsonl"
    dataset_path.write_text(
        '{"query_id":"q1","query":"query","relevant_record_ids":"doc"}\n',
        encoding="utf-8",
    )
    ranking_path.write_text(
        '{"query_id":"q1","ranked_record_ids":"doc"}\n',
        encoding="utf-8",
    )

    with pytest.raises(RetrievalEvaluationError, match="invalid evaluation case"):
        load_evaluation_dataset(dataset_path)
    with pytest.raises(RetrievalEvaluationError, match="invalid ranking"):
        load_rankings(ranking_path)


def test_report_renders_metrics_scope_and_per_query_rows():
    case = EvaluationCase("q1", "退款政策", ("refund",))
    report = evaluate_retrieval([case], {"q1": ("refund",)}, cutoffs=(1,))

    markdown = render_markdown_report(report, "qrels.jsonl", "rankings.jsonl")

    assert "Recall@K" in markdown
    assert "MRR@1" in markdown
    assert "仅代表本次输入的离线排名" in markdown
    assert "| q1 | 退款政策 | 1.0000 / 1.0000 |" in markdown
    assert "| --- | --- | --- |" in markdown


def test_repository_evaluation_dataset_is_loadable():
    cases = load_evaluation_dataset("data/evaluation/ch04/retrieval_qrels.jsonl")

    assert len(cases) == 8
    assert cases[1].relevant_record_ids == ("return-policy-01", "return-policy-02")
