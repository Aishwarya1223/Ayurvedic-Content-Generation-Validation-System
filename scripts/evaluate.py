# src/eval/evaluator.py
import json
import statistics
import time
import csv
from pathlib import Path

def compute_run_metrics(fact_check):
    per = fact_check.get("per_sentence", [])
    n = max(1, len(per))
    matched_count = sum(1 for r in per if r.get("matched"))
    grounding_score = matched_count / n
    avg_overlap = statistics.mean([r.get("overlap_score", 0.0) for r in per]) if per else 0.0
    retrieval_coverage = sum(1 for r in per if r.get("top_matches")) / n
    unsupported_rate = 1.0 - grounding_score

    overlaps = [r.get("overlap_score", 0.0) for r in per]
    p50 = statistics.median(overlaps) if overlaps else 0.0
    p90 = sorted(overlaps)[int(0.9 * len(overlaps))-1] if overlaps else 0.0

    return {
        "n_sentences": n,
        "grounding_score": grounding_score,
        "unsupported_rate": unsupported_rate,
        "avg_overlap": avg_overlap,
        "p50_overlap": p50,
        "p90_overlap": p90,
        "retrieval_coverage": retrieval_coverage,
    }

def persist_run_csv(out_dir, run_meta, metrics, fact_check):
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    csvf = p / "eval_runs.csv"
    headers = [
        "timestamp","run_id","brief","grounding_score","unsupported_rate",
        "avg_overlap","p50_overlap","p90_overlap","retrieval_coverage","n_sentences","notes"
    ]
    row = {
        "timestamp": run_meta.get("timestamp", time.time()),
        "run_id": run_meta.get("run_id",""),
        "brief": run_meta.get("brief",""),
        "grounding_score": metrics["grounding_score"],
        "unsupported_rate": metrics["unsupported_rate"],
        "avg_overlap": metrics["avg_overlap"],
        "p50_overlap": metrics["p50_overlap"],
        "p90_overlap": metrics["p90_overlap"],
        "retrieval_coverage": metrics["retrieval_coverage"],
        "n_sentences": metrics["n_sentences"],
        "notes": run_meta.get("notes",""),
    }
    write_header = not csvf.exists()
    with open(csvf, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

def evaluate_and_persist(fact_check, run_meta, out_dir = "./eval_out"):
    metrics = compute_run_metrics(fact_check)
    persist_run_csv(out_dir, run_meta, metrics, fact_check)
    return metrics
