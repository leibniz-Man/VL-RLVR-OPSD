#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


MATH_WORDS = {
    "angle", "triangle", "radius", "area", "perimeter", "equation", "solve",
    "therefore", "hence", "degree", "degrees", "parallel", "line", "point",
    "circle", "length", "width", "height", "sum", "product", "ratio",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute token-level CEPO/GRPO conflict energy.")
    parser.add_argument("--input", required=True, help="Token record JSONL or CSV.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--lam", type=float, default=0.5)
    parser.add_argument("--eps-w", type=float, default=0.2)
    parser.add_argument("--decisive-quantile", type=float, default=0.8)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            return [dict(row) for row in csv.DictReader(f)]
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def as_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_intish(value: Any) -> str:
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return str(value)


def is_math_token(text: str) -> bool:
    s = str(text or "").strip()
    if not s:
        return False
    low = s.lower()
    if re.search(r"[0-9]", s):
        return True
    if re.search(r"[+\-*/=<>^√≤≥]", s):
        return True
    if any(ch in s for ch in "()[]{}"):
        return True
    if re.fullmatch(r"[xyzabcnmXYZABCNM]", s):
        return True
    if low in MATH_WORDS:
        return True
    if "\\" in s:
        return True
    return False


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(values: list[float]) -> dict[str, Any]:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    if not clean:
        return {"n": 0, "mean": None, "median": None}
    return {"n": len(clean), "mean": mean(clean), "median": median(clean)}


def percentile(values: list[float], q: float) -> float:
    clean = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not clean:
        return float("inf")
    q = min(max(q, 0.0), 1.0)
    idx = int(round((len(clean) - 1) * q))
    return clean[idx]


def enrich_rows(rows: list[dict[str, Any]], lam: float, eps_w: float) -> list[dict[str, Any]]:
    enriched = []
    for row in rows:
        prompt_id = str(row.get("prompt_id") or row.get("id") or "")
        rollout_id = str(row.get("rollout_id") or "")
        token_id = as_intish(row.get("token_id"))
        token_text = str(row.get("token_text", ""))
        reward = as_float(row, "reward")
        advantage = as_float(row, "advantage")
        delta = as_float(row, "delta_cepo")
        eff_grpo = as_float(row, "effective_adv_grpo", advantage)
        if "token_weight_cepo" in row and row.get("token_weight_cepo") not in (None, ""):
            token_weight = as_float(row, "token_weight_cepo", 1.0)
        else:
            sign = 1.0 if advantage > 0 else -1.0 if advantage < 0 else 0.0
            token_weight = math.exp(sign * delta) if sign else 1.0
            token_weight = min(max(token_weight, 1.0 - eps_w), 1.0 + eps_w)
        if "effective_adv_cepo" in row and row.get("effective_adv_cepo") not in (None, ""):
            eff_cepo = as_float(row, "effective_adv_cepo")
        else:
            eff_cepo = advantage * ((1.0 - lam) + lam * token_weight)
        enriched.append({
            **row,
            "prompt_id": prompt_id,
            "rollout_id": rollout_id,
            "token_id": token_id,
            "token_text": token_text,
            "reward": reward,
            "advantage": advantage,
            "delta_cepo": delta,
            "effective_adv_grpo": eff_grpo,
            "token_weight_cepo": token_weight,
            "effective_adv_cepo": eff_cepo,
            "is_math_token": is_math_token(token_text),
        })
    return enriched


def mark_decisive(rows: list[dict[str, Any]], q: float) -> None:
    by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_prompt[row["prompt_id"]].append(row)
    for prompt_rows in by_prompt.values():
        threshold = percentile([abs(float(r["delta_cepo"])) for r in prompt_rows], q)
        for row in prompt_rows:
            row["is_decisive"] = abs(float(row["delta_cepo"])) >= threshold


def compute_conflicts(rows: list[dict[str, Any]], epsilon: float) -> list[dict[str, Any]]:
    by_prompt_token: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    prompt_sides: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        key = (row["prompt_id"], row["token_id"])
        by_prompt_token[key].append(row)
        if float(row["advantage"]) > 0:
            prompt_sides[key].add("pos")
        elif float(row["advantage"]) < 0:
            prompt_sides[key].add("neg")

    out = []
    for (prompt_id, token_id), token_rows in by_prompt_token.items():
        shared = prompt_sides[(prompt_id, token_id)] == {"pos", "neg"}
        p_pos_grpo = sum(abs(float(r["effective_adv_grpo"])) for r in token_rows if float(r["effective_adv_grpo"]) > 0)
        p_neg_grpo = sum(abs(float(r["effective_adv_grpo"])) for r in token_rows if float(r["effective_adv_grpo"]) < 0)
        p_pos_cepo = sum(abs(float(r["effective_adv_cepo"])) for r in token_rows if float(r["effective_adv_cepo"]) > 0)
        p_neg_cepo = sum(abs(float(r["effective_adv_cepo"])) for r in token_rows if float(r["effective_adv_cepo"]) < 0)
        e_grpo = p_pos_grpo * p_neg_grpo
        e_cepo = p_pos_cepo * p_neg_cepo
        log_rho = math.log(e_cepo + epsilon) - math.log(e_grpo + epsilon)
        any_math = any(bool(r["is_math_token"]) for r in token_rows)
        any_decisive = any(bool(r.get("is_decisive")) for r in token_rows)
        token_texts = [str(r.get("token_text", "")) for r in token_rows if str(r.get("token_text", ""))]
        token_text = max(set(token_texts), key=token_texts.count) if token_texts else ""
        if shared and any_math and any_decisive:
            group = "shared_decisive_math"
        elif shared and any_math and not any_decisive:
            group = "shared_non_decisive_math"
        elif (not shared) and any_decisive:
            group = "non_shared_decisive"
        elif not any_math:
            group = "filler"
        else:
            group = "other"
        out.append({
            "prompt_id": prompt_id,
            "token_id": token_id,
            "token_text": token_text,
            "count": len(token_rows),
            "shared": shared,
            "is_math_token": any_math,
            "is_decisive": any_decisive,
            "group": group,
            "p_pos_grpo": p_pos_grpo,
            "p_neg_grpo": p_neg_grpo,
            "e_grpo": e_grpo,
            "p_pos_cepo": p_pos_cepo,
            "p_neg_cepo": p_neg_cepo,
            "e_cepo": e_cepo,
            "rho": (e_cepo + epsilon) / (e_grpo + epsilon),
            "log_rho": log_rho,
            "mean_abs_delta": mean([abs(float(r["delta_cepo"])) for r in token_rows]),
        })
    out.sort(key=lambda r: (r["group"], -float(r["log_rho"])))
    return out


def group_summary(conflicts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in conflicts:
        by_group[str(row["group"])].append(row)
    rows = []
    for group, items in sorted(by_group.items()):
        log_rhos = [float(r["log_rho"]) for r in items]
        e_grpo = [float(r["e_grpo"]) for r in items]
        e_cepo = [float(r["e_cepo"]) for r in items]
        rows.append({
            "group": group,
            "n": len(items),
            "mean_e_grpo": summarize(e_grpo)["mean"],
            "mean_e_cepo": summarize(e_cepo)["mean"],
            "median_log_rho": summarize(log_rhos)["median"],
            "mean_log_rho": summarize(log_rhos)["mean"],
            "pct_rho_gt_1": sum(1 for v in log_rhos if v > 0) / len(log_rhos) if log_rhos else None,
            "pct_rho_gt_1_5": sum(1 for v in log_rhos if v > math.log(1.5)) / len(log_rhos) if log_rhos else None,
            "pct_rho_gt_2": sum(1 for v in log_rhos if v > math.log(2.0)) / len(log_rhos) if log_rhos else None,
        })
    return rows


def prompt_summary(conflicts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_prompt_group: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in conflicts:
        by_prompt_group[(str(row["prompt_id"]), str(row["group"]))].append(float(row["log_rho"]))
    rows = []
    for (prompt_id, group), values in sorted(by_prompt_group.items()):
        rows.append({
            "prompt_id": prompt_id,
            "group": group,
            "n_tokens": len(values),
            "mean_log_rho": mean(values),
            "median_log_rho": median(values),
        })
    return rows


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = enrich_rows(load_rows(Path(args.input)), lam=args.lam, eps_w=args.eps_w)
    mark_decisive(rows, args.decisive_quantile)
    conflicts = compute_conflicts(rows, epsilon=args.epsilon)
    write_csv(out_dir / "token_conflicts.csv", conflicts)
    write_csv(out_dir / "group_summary.csv", group_summary(conflicts))
    write_csv(out_dir / "prompt_summary.csv", prompt_summary(conflicts))
    top = sorted(conflicts, key=lambda r: float(r["log_rho"]), reverse=True)[:200]
    write_csv(out_dir / "top_conflicts.csv", top)
    metadata = {
        "input": args.input,
        "num_token_rows": len(rows),
        "num_prompt_token_conflicts": len(conflicts),
        "lam": args.lam,
        "eps_w": args.eps_w,
        "decisive_quantile": args.decisive_quantile,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote token conflict analysis to {out_dir}")


if __name__ == "__main__":
    main()
