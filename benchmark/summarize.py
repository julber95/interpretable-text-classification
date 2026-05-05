"""
Build a summary (Markdown) from existing JSON result files.

How to use:
    uv run summarize.py configs/fasttext.yaml
"""

import argparse
import json

import yaml

from run import RESULTS_DIR

SUMMARY_COLS = ["dataset_name", "test_accuracy", "test_f1_macro", "train_time_s", "train_size", "test_size"]


def build_summary(model_name: str):
    """Read all JSON results for a model and write summary.md + summary.csv."""
    result_dir = RESULTS_DIR / model_name
    rows = []
    for path in sorted(result_dir.glob("*.json")):
        if path.stem == "summary":
            continue
        with open(path) as f:
            data = json.load(f)
        rows.append({col: data.get(col, "") for col in SUMMARY_COLS})

    if not rows:
        print("No results found.")
        return

    # Markdown
    def fmt(v):
        return f"{v:.4f}" if isinstance(v, float) else str(v)

    col_widths = {col: max(len(col), max(len(fmt(r[col])) for r in rows)) for col in SUMMARY_COLS}
    sep = "| " + " | ".join("-" * col_widths[c] for c in SUMMARY_COLS) + " |"
    header = "| " + " | ".join(c.ljust(col_widths[c]) for c in SUMMARY_COLS) + " |"
    lines = [header, sep]
    for r in rows:
        lines.append("| " + " | ".join(fmt(r[c]).ljust(col_widths[c]) for c in SUMMARY_COLS) + " |")

    md_path = result_dir / "summary.md"
    md_path.write_text(f"# {model_name} — benchmark results\n\n" + "\n".join(lines) + "\n")

    print(f"Summary saved → {md_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to the model YAML config file")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    build_summary(cfg["model_name"])
