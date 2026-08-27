"""
summary_to_csv.py — transfer_eval.py の *.summary.json を
カテゴリ × held-out ターゲット の Transfer ASR 行列(CSV)に変換する。

使い方:
    python summary_to_csv.py data/transfer/llama32_1b_source.summary.json \
        --output data/transfer/llama32_1b_source_matrix.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("summary_json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with open(args.summary_json, "r", encoding="utf-8") as f:
        summary = json.load(f)

    by_cat_target = summary["by_category_and_target"]
    target_names = summary["config"]["held_out_targets"]

    # カテゴリ一覧を抽出
    categories = sorted({key.split("__")[0] for key in by_cat_target})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["vuln_category"] + [f"{t}_transfer_asr" for t in target_names] + ["category_overall_asr"])
        for cat in categories:
            row = [cat]
            for target in target_names:
                key = f"{cat}__{target}"
                asr = by_cat_target.get(key, {}).get("transfer_asr", "")
                row.append(asr)
            cat_overall = summary["by_category_overall"].get(cat, {}).get("transfer_asr", "")
            row.append(cat_overall)
            writer.writerow(row)

        # 全体行
        writer.writerow([])
        writer.writerow(["overall"] + [
            summary["by_target_overall"].get(t, {}).get("transfer_asr", "")
            for t in target_names
        ] + [""])

    print(f"書き出し完了: {args.output}")


if __name__ == "__main__":
    main()
