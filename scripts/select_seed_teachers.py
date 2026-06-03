#!/usr/bin/env python3
"""Select per-opponent teacher checkpoints from SEED RL matrix rows."""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics
from pathlib import Path
from typing import Iterable


def _checkpoint_base(path: str) -> str:
    path = str(path)
    if path.endswith(".actor.weights.h5"):
        return path[: -len(".actor.weights.h5")]
    if path.endswith(".critic.weights.h5"):
        return path[: -len(".critic.weights.h5")]
    return path


def _basename(path: str) -> str:
    return os.path.basename(_checkpoint_base(str(path)))


def _expand(paths: Iterable[str]) -> list[Path]:
    out: list[Path] = []
    for raw in paths:
        matches = [Path(p) for p in glob.glob(raw)]
        if not matches:
            matches = [Path(raw)]
        for path in matches:
            if path.exists() and path not in out:
                out.append(path)
    return out


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-file", action="append", default=[])
    parser.add_argument("--matrix-glob", action="append", default=[])
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-games", type=int, default=500)
    parser.add_argument("--min-score", type=float, default=-10_000.0)
    parser.add_argument(
        "--aggregate-repeats",
        action="store_true",
        help="Aggregate repeated rows by (teacher, opponent) before selecting teachers.",
    )
    parser.add_argument(
        "--rank-metric",
        choices=["score", "mean", "lcb", "min"],
        default="score",
        help="Metric used to choose the teacher per opponent. With --aggregate-repeats, score aliases mean.",
    )
    parser.add_argument(
        "--min-repeats",
        type=int,
        default=1,
        help="Minimum repeated rows required when --aggregate-repeats is enabled.",
    )
    parser.add_argument(
        "--max-std",
        type=float,
        default=None,
        help="Optional maximum repeat stddev for aggregate teacher candidates.",
    )
    parser.add_argument(
        "--min-lcb",
        type=float,
        default=None,
        help="Optional minimum 95%% lower confidence bound for aggregate teacher candidates.",
    )
    parser.add_argument("--opponent", action="append", default=[])
    parser.add_argument(
        "--allow-missing-weights",
        action="store_true",
        help="Keep selected rows even when teacher/opponent weight files are not present on this machine.",
    )
    return parser.parse_args()


def _score_rows(values: list[float]) -> dict:
    mean = float(sum(values) / len(values))
    std = float(statistics.stdev(values)) if len(values) > 1 else 0.0
    lcb95 = float(mean - 1.96 * std / math.sqrt(len(values)))
    return {
        "mean": mean,
        "std": std,
        "min": float(min(values)),
        "max": float(max(values)),
        "lcb": lcb95,
        "repeats": len(values),
    }


def _rank_value(candidate: dict, metric: str) -> float:
    if metric == "score":
        return float(candidate["teacher_score"])
    if metric == "mean":
        return float(candidate.get("teacher_mean_score", candidate["teacher_score"]))
    if metric == "lcb":
        return float(candidate.get("teacher_lcb95", candidate["teacher_score"]))
    if metric == "min":
        return float(candidate.get("teacher_min_score", candidate["teacher_score"]))
    raise ValueError(metric)


def main() -> int:
    args = parse_args()
    paths = _expand(list(args.matrix_file) + list(args.matrix_glob))
    if not paths:
        raise SystemExit("No matrix files found")

    wanted = {_basename(name) for name in args.opponent}
    best: dict[str, dict] = {}
    grouped: dict[tuple[str, str], list[dict]] = {}
    considered = 0
    for path in paths:
        for row in _read_jsonl(path):
            model = row.get("model")
            opponent = row.get("opponent")
            if not model or not opponent:
                continue
            opponent_name = row.get("opponent_name") or _basename(str(opponent))
            if wanted and opponent_name not in wanted:
                continue
            games = int(row.get("total_games", 0) or 0)
            if games < args.min_games:
                continue
            score = float(row.get("bb_per_100", 0.0))
            if score < args.min_score:
                continue
            teacher = _checkpoint_base(str(model))
            opponent_base = _checkpoint_base(str(opponent))
            if not args.allow_missing_weights:
                if not Path(teacher + ".actor.weights.h5").exists():
                    continue
                if not Path(opponent_base + ".actor.weights.h5").exists():
                    continue
            considered += 1
            if args.aggregate_repeats:
                grouped.setdefault((opponent_name, teacher), []).append(
                    {
                        "score": score,
                        "games": games,
                        "teacher": teacher,
                        "teacher_name": _basename(teacher),
                        "opponent": opponent_base,
                        "opponent_name": opponent_name,
                        "source": str(path),
                    }
                )
                continue
            prev = best.get(opponent_name)
            candidate = {
                "row": opponent_name,
                "teacher": teacher,
                "teacher_name": _basename(teacher),
                "teacher_score": score,
                "opponent": opponent_base,
                "opponent_name": opponent_name,
                "source": str(path),
                "total_games": games,
            }
            if prev is None or _rank_value(candidate, args.rank_metric) > _rank_value(prev, args.rank_metric):
                best[opponent_name] = {
                    "row": opponent_name,
                    "teacher": teacher,
                    "teacher_name": _basename(teacher),
                    "teacher_score": score,
                    "opponent": opponent_base,
                    "opponent_name": opponent_name,
                    "source": str(path),
                    "total_games": games,
                }

    if args.aggregate_repeats:
        for (opponent_name, teacher), items in grouped.items():
            if len(items) < max(1, int(args.min_repeats)):
                continue
            scores = [float(item["score"]) for item in items]
            stats = _score_rows(scores)
            if args.max_std is not None and stats["std"] > float(args.max_std):
                continue
            if args.min_lcb is not None and stats["lcb"] < float(args.min_lcb):
                continue
            first = items[0]
            candidate = {
                "row": opponent_name,
                "teacher": teacher,
                "teacher_name": first["teacher_name"],
                "teacher_score": stats["mean"],
                "teacher_mean_score": stats["mean"],
                "teacher_lcb95": stats["lcb"],
                "teacher_min_score": stats["min"],
                "teacher_max_score": stats["max"],
                "teacher_std": stats["std"],
                "teacher_repeats": stats["repeats"],
                "opponent": first["opponent"],
                "opponent_name": opponent_name,
                "source": items[-1]["source"],
                "sources": sorted({str(item["source"]) for item in items}),
                "scores": scores,
                "total_games": int(sum(int(item["games"]) for item in items)),
                "rank_metric": args.rank_metric,
            }
            prev = best.get(opponent_name)
            if prev is None or _rank_value(candidate, args.rank_metric) > _rank_value(prev, args.rank_metric):
                best[opponent_name] = candidate

    selected = [best[name] for name in sorted(best)]
    out = {
        "selected": selected,
        "selected_count": len(selected),
        "considered_rows": considered,
        "sources": [str(path) for path in paths],
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"out": str(out_path), "selected_count": len(selected)}, ensure_ascii=False))
    return 0 if selected else 1


if __name__ == "__main__":
    raise SystemExit(main())
