#!/usr/bin/env python3
"""Entropy probe for original-image vs full-mask-image token distributions.

This is an isolated analysis script for the custom image-mask CEPO experiment. It
leaves the original CEPO source tree unchanged.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from datasets import load_dataset
from jinja2 import Template
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


@dataclass
class TokenRecord:
    sample_index: int
    token_index: int
    token_id: int
    token: str
    decoded_token: str
    h_vis: float
    h_mask: float
    gap: float
    logp_vis: float
    logp_mask: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", default="hiyouga/geometry3k")
    parser.add_argument("--split", default="test")
    parser.add_argument("--prompt-template", required=True)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--mask-color", default="black")
    parser.add_argument("--output", required=True)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--significant-gap", type=float, default=0.05)
    parser.add_argument("--topk-preview", type=int, default=8)
    return parser.parse_args()


def build_messages(prompt: str) -> List[Dict[str, Any]]:
    content: List[Dict[str, str]] = []
    parts = prompt.split("<image>")
    for idx, part in enumerate(parts):
        if idx > 0:
            content.append({"type": "image"})
        if part:
            content.append({"type": "text", "text": part})
    if not any(item.get("type") == "image" for item in content):
        content.insert(0, {"type": "image"})
    return [{"role": "user", "content": content}]


def first_image(example: Dict[str, Any]) -> Image.Image:
    imgs = example.get("images")
    if isinstance(imgs, (list, tuple)):
        img = imgs[0]
    else:
        img = imgs
    if not isinstance(img, Image.Image):
        img = Image.open(img)
    return img.convert("RGB")


def mask_image(img: Image.Image, mode: str = "black") -> Image.Image:
    if mode != "black":
        raise ValueError(f"Unsupported mask mode: {mode}")
    return Image.new("RGB", img.size, (0, 0, 0))


def to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def prepare_inputs(processor, prompt_text: str, img: Image.Image, device: torch.device) -> Dict[str, Any]:
    messages = build_messages(prompt_text)
    chat_text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(images=[img], text=[chat_text], add_special_tokens=False, return_tensors="pt")
    inputs = to_device(dict(inputs), device)
    inputs["_chat_text"] = chat_text
    return inputs


def entropy_and_logp(logits: torch.Tensor, token_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    if token_ids.device != log_probs.device:
        token_ids = token_ids.to(log_probs.device)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)
    picked = log_probs.gather(-1, token_ids[:, None]).squeeze(-1)
    return entropy.cpu(), picked.cpu(), log_probs.cpu()


def append_response(prompt_inputs: Dict[str, Any], response_ids: torch.Tensor, device: torch.device) -> Dict[str, Any]:
    full: Dict[str, Any] = {}
    prompt_ids = prompt_inputs["input_ids"]
    if response_ids.ndim == 1:
        response_ids = response_ids.unsqueeze(0)
    full["input_ids"] = torch.cat([prompt_ids, response_ids.to(device)], dim=1)
    prompt_mask = prompt_inputs.get("attention_mask", torch.ones_like(prompt_ids))
    response_mask = torch.ones((1, response_ids.shape[1]), dtype=prompt_mask.dtype, device=device)
    full["attention_mask"] = torch.cat([prompt_mask, response_mask], dim=1)
    for key, value in prompt_inputs.items():
        if key in {"input_ids", "attention_mask", "_chat_text"}:
            continue
        full[key] = value
    return full


def topk_for_position(log_probs: torch.Tensor, tokenizer, k: int) -> List[Dict[str, Any]]:
    values, ids = torch.topk(log_probs, k=k, dim=-1)
    out = []
    for score, tok_id in zip(values.tolist(), ids.tolist()):
        out.append({
            "token_id": int(tok_id),
            "token": tokenizer.convert_ids_to_tokens(int(tok_id)),
            "decoded": tokenizer.decode([int(tok_id)], skip_special_tokens=False),
            "prob": float(math.exp(score)),
            "logp": float(score),
        })
    return out


def summarize_records(records: List[TokenRecord], significant_gap: float) -> Dict[str, Any]:
    gaps = [r.gap for r in records]
    h_vis = [r.h_vis for r in records]
    h_mask = [r.h_mask for r in records]
    if not records:
        return {}
    total = len(records)
    neg = sum(g < 0 for g in gaps)
    pos = sum(g > 0 for g in gaps)
    close = sum(abs(g) <= 0.01 for g in gaps)
    sig_neg = sum(g < -significant_gap for g in gaps)
    sig_pos = sum(g > significant_gap for g in gaps)
    return {
        "total_tokens": total,
        "mean_h_vis": sum(h_vis) / total,
        "mean_h_mask": sum(h_mask) / total,
        "mean_gap_h_mask_minus_h_vis": sum(gaps) / total,
        "fraction_gap_positive": pos / total,
        "fraction_gap_negative": neg / total,
        "fraction_gap_near_zero_abs_le_0_01": close / total,
        "fraction_gap_negative_lt_minus_significant_gap": sig_neg / total,
        "fraction_gap_positive_gt_significant_gap": sig_pos / total,
        "min_gap": min(gaps),
        "max_gap": max(gaps),
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    template = Template(Path(args.prompt_template).read_text())
    print(f"Loading processor/model from {args.model}", flush=True)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        device_map={"": 0} if device.type == "cuda" else None,
        trust_remote_code=True,
    )
    model.eval()
    tokenizer = processor.tokenizer

    print(f"Loading dataset {args.dataset}:{args.split}", flush=True)
    ds = load_dataset(args.dataset, split=args.split)
    sample_indices = list(range(args.start_index, min(len(ds), args.start_index + args.num_samples)))

    all_records: List[TokenRecord] = []
    sample_summaries: List[Dict[str, Any]] = []
    worst_record: TokenRecord | None = None
    worst_extra: Dict[str, Any] | None = None

    with torch.inference_mode():
        for n, idx in enumerate(sample_indices, start=1):
            ex = ds[idx]
            img = first_image(ex)
            masked = mask_image(img, args.mask_color)
            rendered = template.render(content=ex["problem"])
            prompt_vis = prepare_inputs(processor, rendered, img, device)
            prompt_mask = prepare_inputs(processor, rendered, masked, device)
            prompt_len = prompt_vis["input_ids"].shape[1]
            print(f"[{n}/{len(sample_indices)}] sample={idx} prompt_len={prompt_len}", flush=True)

            gen_kwargs = dict(max_new_tokens=args.max_new_tokens)
            if args.do_sample:
                gen_kwargs.update(dict(do_sample=True, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k))
            else:
                gen_kwargs.update(dict(do_sample=False))
            generated = model.generate(
                **{k: v for k, v in prompt_vis.items() if not k.startswith("_")},
                **gen_kwargs,
            )
            response_ids = generated[0, prompt_len:]
            if response_ids.numel() == 0:
                print(f"sample={idx} produced empty response; skipping", flush=True)
                continue
            eos_id = tokenizer.eos_token_id
            if eos_id is not None:
                eos_positions = (response_ids == eos_id).nonzero(as_tuple=False)
                if eos_positions.numel() > 0:
                    response_ids = response_ids[: int(eos_positions[0].item())]
            if response_ids.numel() == 0:
                continue

            full_vis = append_response(prompt_vis, response_ids, device)
            full_mask = append_response(prompt_mask, response_ids, device)
            out_vis = model(**full_vis)
            out_mask = model(**full_mask)
            T = response_ids.shape[0]
            pred_start = prompt_len - 1
            pred_end = pred_start + T
            logits_vis = out_vis.logits[0, pred_start:pred_end, :]
            logits_mask = out_mask.logits[0, pred_start:pred_end, :]
            h_vis, logp_vis, log_probs_vis = entropy_and_logp(logits_vis, response_ids.detach().cpu())
            h_mask, logp_mask, log_probs_mask = entropy_and_logp(logits_mask, response_ids.detach().cpu())
            gaps = h_mask - h_vis
            sample_records: List[TokenRecord] = []
            for t in range(T):
                tok_id = int(response_ids[t].item())
                rec = TokenRecord(
                    sample_index=idx,
                    token_index=t,
                    token_id=tok_id,
                    token=tokenizer.convert_ids_to_tokens(tok_id),
                    decoded_token=tokenizer.decode([tok_id], skip_special_tokens=False),
                    h_vis=float(h_vis[t].item()),
                    h_mask=float(h_mask[t].item()),
                    gap=float(gaps[t].item()),
                    logp_vis=float(logp_vis[t].item()),
                    logp_mask=float(logp_mask[t].item()),
                )
                sample_records.append(rec)
                all_records.append(rec)
                if worst_record is None or rec.gap < worst_record.gap:
                    lo = max(0, t - 25)
                    hi = min(T, t + 26)
                    context_ids = response_ids[lo:hi].tolist()
                    worst_record = rec
                    worst_extra = {
                        "sample_index": idx,
                        "problem": ex["problem"],
                        "answer": ex.get("answer"),
                        "generated_response": tokenizer.decode(response_ids.tolist(), skip_special_tokens=True),
                        "context_window_token_start": lo,
                        "context_window_token_end": hi,
                        "context_window_text": tokenizer.decode(context_ids, skip_special_tokens=False),
                        "topk_vis": topk_for_position(log_probs_vis[t], tokenizer, args.topk_preview),
                        "topk_mask": topk_for_position(log_probs_mask[t], tokenizer, args.topk_preview),
                    }
            sample_summary = summarize_records(sample_records, args.significant_gap)
            sample_summary.update({
                "sample_index": idx,
                "num_response_tokens": len(sample_records),
                "problem": ex["problem"],
                "answer": ex.get("answer"),
                "generated_response": tokenizer.decode(response_ids.tolist(), skip_special_tokens=True),
            })
            sample_summaries.append(sample_summary)
            print(
                "  tokens={tokens} mean_gap={gap:.4f} frac_neg={neg:.3f} frac_pos={pos:.3f}".format(
                    tokens=sample_summary["num_response_tokens"],
                    gap=sample_summary["mean_gap_h_mask_minus_h_vis"],
                    neg=sample_summary["fraction_gap_negative"],
                    pos=sample_summary["fraction_gap_positive"],
                ),
                flush=True,
            )
            del out_vis, out_mask, logits_vis, logits_mask, log_probs_vis, log_probs_mask
            if device.type == "cuda":
                torch.cuda.empty_cache()

    overall = summarize_records(all_records, args.significant_gap)
    negative_sorted = sorted(all_records, key=lambda r: r.gap)[:50]
    positive_sorted = sorted(all_records, key=lambda r: r.gap, reverse=True)[:20]
    payload = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": args.model,
        "dataset": args.dataset,
        "split": args.split,
        "num_samples_requested": args.num_samples,
        "sample_indices": sample_indices,
        "max_new_tokens": args.max_new_tokens,
        "generation": {
            "do_sample": args.do_sample,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
        },
        "definition": "gap = H_mask - H_vis; negative gap means full-mask condition has lower next-token entropy than original-image condition for the same generated prefix.",
        "overall": overall,
        "sample_summaries": sample_summaries,
        "all_token_records": [asdict(r) for r in all_records],
        "most_negative_tokens": [asdict(r) for r in negative_sorted],
        "most_positive_tokens": [asdict(r) for r in positive_sorted],
        "selected_example": {
            "record": asdict(worst_record) if worst_record else None,
            "details": worst_extra,
        },
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Wrote {output_path}", flush=True)
    print(json.dumps(overall, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
