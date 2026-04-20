#!/usr/bin/env python3
"""
Compute tool_call_f1 (ToolCall-Trigger Similarity) between a ground-truth
JSONL and a test JSONL.

Per the README:
  TP: Both ground-truth and test have finish_reason == "tool_calls"
  FP: Test has "tool_calls" but ground-truth has "stop" or others
  FN: Test has "stop" or others but ground-truth has "tool_calls"
  TN: Both have "stop" or others

  precision = TP / (TP + FP)
  recall    = TP / (TP + FN)
  f1        = 2 * precision * recall / (precision + recall)

Requests are matched by ``data_index``.

Usage:
    python f1_score.py ground_truth.jsonl test_results.jsonl
"""

import argparse
import json
import sys
from typing import Dict


def load_finish_reasons(path: str) -> Dict[int, str]:
    """Return {data_index: finish_reason} from a results JSONL."""
    reasons: Dict[int, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            reasons[record["data_index"]] = record["finish_reason"]
    return reasons


def is_tool_calls(finish_reason: str) -> bool:
    return finish_reason == "tool_calls"


def compute_f1(gt_path: str, test_path: str) -> None:
    gt = load_finish_reasons(gt_path)
    test = load_finish_reasons(test_path)

    common_indices = sorted(set(gt) & set(test))
    if not common_indices:
        print("Error: no common data_index values between the two files.", file=sys.stderr)
        sys.exit(1)

    gt_only = set(gt) - set(test)
    test_only = set(test) - set(gt)
    if gt_only:
        print(f"Warning: {len(gt_only)} indices in ground-truth but not in test (ignored).", file=sys.stderr)
    if test_only:
        print(f"Warning: {len(test_only)} indices in test but not in ground-truth (ignored).", file=sys.stderr)

    tp = fp = fn = tn = 0
    for idx in common_indices:
        gt_tc = is_tool_calls(gt[idx])
        test_tc = is_tool_calls(test[idx])
        if gt_tc and test_tc:
            tp += 1
        elif (not gt_tc) and test_tc:
            fp += 1
        elif gt_tc and (not test_tc):
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"Matched requests: {len(common_indices)}")
    print(f"TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"Precision: {precision:.4f} ({precision * 100:.2f}%)")
    print(f"Recall:    {recall:.4f} ({recall * 100:.2f}%)")
    print(f"F1:        {f1:.4f} ({f1 * 100:.2f}%)")


def main():
    parser = argparse.ArgumentParser(description="Compute tool_call_f1 (ToolCall-Trigger Similarity)")
    parser.add_argument("ground_truth", help="Path to ground-truth results JSONL")
    parser.add_argument("test_results", help="Path to test results JSONL")
    args = parser.parse_args()
    compute_f1(args.ground_truth, args.test_results)


if __name__ == "__main__":
    main()
