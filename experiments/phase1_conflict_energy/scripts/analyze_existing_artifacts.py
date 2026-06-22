#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import mean, median
from typing import Any


DEFAULT_GRPO_CKPT = "checkpoints/cepo/qwen3_vl_2b_geo_grpo"
DEFAULT_GRPO_EVAL = "logs/lmms_eval_outputs_retry2_20260615_014949/actor__huggingface/20260615_015007_results.json"
DEFAULT_GRPO_LOG = "logs/grpo_promptfix_20260614_230930.nohup.log"
DEFAULT_CEPO_LOG = "logs/cepo_launcher_20260615_084307.nohup.log"
DEFAULT_CEPO_HIST = "checkpoints/cepo_archived/qwen3_vl_2b_geo_cepo_incomplete_20260615_084249/cepo_histograms"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze existing CEPO/GRPO artifacts for Phase 1 diagnostics.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-dir", default="experiments/phase1_conflict_energy/outputs/existing_artifacts")
    parser.add_argument("--grpo-checkpoint-dir", default=DEFAULT_GRPO_CKPT)
    parser.add_argument("--grpo-eval-json", default=DEFAULT_GRPO_EVAL)
    parser.add_argument("--grpo-log", default=DEFAULT_GRPO_LOG)
    parser.add_argument("--cepo-log", default=DEFAULT_CEPO_LOG)
    parser.add_argument("--cepo-hist-dir", default=DEFAULT_CEPO_HIST)
    return parser.parse_args()


def read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def safe_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.endswith("%"):
            try:
                return float(stripped[:-1]) / 100.0
            except ValueError:
                return None
        try:
            return float(stripped)
        except ValueError:
            return None
    return None


def summarize_values(values: list[float]) -> dict[str, float | int | None]:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    if not clean:
        return {"n": 0, "mean": None, "median": None, "std": None, "min": None, "max": None}
    mu = mean(clean)
    var = mean([(v - mu) ** 2 for v in clean])
    return {
        "n": len(clean),
        "mean": mu,
        "median": median(clean),
        "std": math.sqrt(var),
        "min": min(clean),
        "max": max(clean),
    }


def parse_step_metrics(log_path: Path) -> list[dict[str, Any]]:
    if not log_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    section: str | None = None
    ansi_re = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
    ray_prefix_re = re.compile(r"^\([^)]*pid=[0-9]+\)\s*")
    step_re = re.compile(r"\bStep\s+(\d+)\s*$")
    kv_re = re.compile(r"^\s*([A-Za-z0-9_./-]+):\s+([-+0-9.eE]+)\s*$")
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = ansi_re.sub("", line).replace("\r", "")
        line = ray_prefix_re.sub("", line.strip())
        m_step = step_re.search(line)
        if m_step:
            if current:
                rows.append(current)
            current = {"step": int(m_step.group(1))}
            section = None
            continue
        if current is None:
            continue
        stripped = line.strip()
        if stripped.endswith(":") and re.match(r"^[A-Za-z0-9_/-]+:$", stripped):
            section = stripped[:-1]
            continue
        m_kv = kv_re.match(line)
        if m_kv and section:
            key = f"{section}/{m_kv.group(1)}"
            current[key] = float(m_kv.group(2))
    if current:
        rows.append(current)
    return rows


def parse_cepo_prints(log_path: Path) -> list[dict[str, Any]]:
    if not log_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    pattern = re.compile(
        r"\[CEPO\]\s+step=(?P<step>\d+),\s+lambda=(?P<lam>[-+0-9.eE]+),\s+"
        r"delta_mean=(?P<delta_mean>[-+0-9.eE]+),\s+delta_std=(?P<delta_std>[-+0-9.eE]+),\s+"
        r"mu\+=(?P<mu_pos>[-+0-9.eE]+),\s+mu-=(?P<mu_neg>[-+0-9.eE]+),\s+"
        r"fwd=(?P<fwd>[-+0-9.eE]+)\s+\(pos=(?P<fwd_pos>[-+0-9.eE]+),\s+neg=(?P<fwd_neg>[-+0-9.eE]+)\),\s+"
        r"adv_pre=(?P<adv_pre>[-+0-9.eE]+),\s+adv_post=(?P<adv_post>[-+0-9.eE]+)"
    )
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = pattern.search(line)
        if not m:
            continue
        row = {k: float(v) for k, v in m.groupdict().items()}
        row["step"] = int(row["step"])
        rows.append(row)
    return rows


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


def extract_eval_metrics(eval_json: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not eval_json:
        return []
    rows = []
    for task, metrics in (eval_json.get("results") or {}).items():
        row: dict[str, Any] = {"task": task}
        for key, value in metrics.items():
            if key == "alias" or "stderr" in key or key.startswith("submission"):
                continue
            fval = safe_float(value)
            if fval is not None:
                row[key] = fval
        rows.append(row)
    return rows


def summarize_histograms(hist_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not hist_dir.exists():
        return rows
    for path in sorted(hist_dir.glob("step_*.json")):
        obj = read_json(path) or {}
        delta_pos = [float(x) for x in obj.get("delta_pos", []) if math.isfinite(float(x))]
        delta_neg = [float(x) for x in obj.get("delta_neg", []) if math.isfinite(float(x))]
        all_delta = delta_pos + delta_neg
        row: dict[str, Any] = {"file": str(path), "step": obj.get("step")}
        for prefix, values in [("all", all_delta), ("pos", delta_pos), ("neg", delta_neg)]:
            stats = summarize_values(values)
            for k, v in stats.items():
                row[f"delta_{prefix}_{k}"] = v
        row["frac_wrong_dir_pos"] = sum(1 for v in delta_pos if v < 0) / len(delta_pos) if delta_pos else None
        row["frac_wrong_dir_neg"] = sum(1 for v in delta_neg if v > 0) / len(delta_neg) if delta_neg else None
        signed = delta_pos + [-v for v in delta_neg]
        row["frac_wrong_dir_signed"] = sum(1 for v in signed if v < 0) / len(signed) if signed else None
        rows.append(row)
    return rows


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Phase 1 Existing-Artifact Analysis")
    lines.append("")
    lines.append("## Inputs")
    for key, value in payload["inputs"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    lines.append("## GRPO Eval Summary")
    eval_rows = payload["eval_metrics"]
    if eval_rows:
        for row in eval_rows:
            metrics = ", ".join(f"{k}={v:.4g}" for k, v in row.items() if k != "task" and isinstance(v, float))
            lines.append(f"- `{row['task']}`: {metrics}")
    else:
        lines.append("- No eval JSON found.")
    lines.append("")
    lines.append("## CEPO Delta Histogram Summary")
    hist_rows = payload["histogram_summary"]
    if hist_rows:
        for row in hist_rows:
            lines.append(
                "- step `{}`: n_pos={}, n_neg={}, mean_pos={:.6g}, mean_neg={:.6g}, "
                "wrong_pos={:.3g}, wrong_neg={:.3g}, signed_wrong={:.3g}".format(
                    row.get("step"),
                    row.get("delta_pos_n"),
                    row.get("delta_neg_n"),
                    row.get("delta_pos_mean") or 0.0,
                    row.get("delta_neg_mean") or 0.0,
                    row.get("frac_wrong_dir_pos") or 0.0,
                    row.get("frac_wrong_dir_neg") or 0.0,
                    row.get("frac_wrong_dir_signed") or 0.0,
                )
            )
    else:
        lines.append("- No CEPO histogram JSON found.")
    lines.append("")
    lines.append("## Interpretation")
    lines.append(
        "The histogram-only diagnostic can verify whether CEPO evidence is directionally aligned "
        "on positive and negative rollout tokens. It cannot prove shared-token conflict by itself, "
        "because the saved histogram omits prompt ids, rollout ids, token ids, and token text."
    )
    lines.append("")
    lines.append("## Next Required Dump")
    lines.append(
        "To run the full conflict-energy test, dump one row per response token with: "
        "`prompt_id`, `rollout_id`, `token_position`, `token_id`, `token_text`, `reward`, "
        "`advantage`, `delta_cepo`, and optionally `pos_teacher_logprob`, `neg_teacher_logprob`."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo = Path(args.repo_root).resolve()
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = repo / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    grpo_ckpt = repo / args.grpo_checkpoint_dir
    tracker = read_json(grpo_ckpt / "checkpoint_tracker.json")
    eval_json = read_json(repo / args.grpo_eval_json)
    eval_rows = extract_eval_metrics(eval_json)
    grpo_steps = parse_step_metrics(repo / args.grpo_log)
    cepo_steps = parse_step_metrics(repo / args.cepo_log)
    cepo_prints = parse_cepo_prints(repo / args.cepo_log)
    hist_rows = summarize_histograms(repo / args.cepo_hist_dir)

    write_csv(out_dir / "grpo_eval_metrics.csv", eval_rows)
    write_csv(out_dir / "grpo_step_metrics.csv", grpo_steps)
    write_csv(out_dir / "cepo_step_metrics.csv", cepo_steps)
    write_csv(out_dir / "cepo_print_metrics.csv", cepo_prints)
    write_csv(out_dir / "cepo_histogram_summary.csv", hist_rows)

    summary = {
        "inputs": {
            "grpo_checkpoint_dir": str(grpo_ckpt),
            "grpo_eval_json": str(repo / args.grpo_eval_json),
            "grpo_log": str(repo / args.grpo_log),
            "cepo_log": str(repo / args.cepo_log),
            "cepo_hist_dir": str(repo / args.cepo_hist_dir),
        },
        "checkpoint_tracker": tracker,
        "eval_metrics": eval_rows,
        "num_grpo_step_rows": len(grpo_steps),
        "num_cepo_step_rows": len(cepo_steps),
        "num_cepo_print_rows": len(cepo_prints),
        "histogram_summary": hist_rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(out_dir / "REPORT.md", summary)
    print(f"Wrote existing-artifact analysis to {out_dir}")


if __name__ == "__main__":
    main()
