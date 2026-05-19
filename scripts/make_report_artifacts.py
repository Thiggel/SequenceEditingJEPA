from __future__ import annotations

import json
import os
import re
from pathlib import Path
from xml.sax.saxutils import escape


RUN_LABELS = {
    "igsm_official_med_causal_lm_200k": "causal",
    "igsm_official_med_mask_x0_action_conditioned_jepa_200k": "x0 JEPA",
    "igsm_official_med_mask_x0_denoising_lm_200k": "x0 DLM",
    "igsm_official_med_step_mask_jepa_T20_200k": "step JEPA T20",
    "igsm_official_med_step_mask_jepa_T50_200k": "step JEPA T50",
    "igsm_official_med_step_mask_jepa_T64_200k": "step JEPA T64",
}


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    work_root = Path(os.environ.get("SEQ_EDIT_JEPA_WORK_ROOT", "/home/atuin/c107fa/c107fa12/sequence-editing"))
    runs_root = work_root / "runs"
    output_dir = repo / "docs" / "assets"
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_train_loss_chart(runs_root, output_dir / "igsm_train_loss.svg")
    _write_periodic_chart(
        runs_root,
        output_dir / "igsm_full_denoise_answer_accuracy.svg",
        "periodic/full_denoise/igsm/answer_accuracy",
        "Full-denoise answer accuracy",
    )
    _write_periodic_chart(
        runs_root,
        output_dir / "igsm_full_denoise_token_accuracy.svg",
        "periodic/full_denoise/task/token_accuracy",
        "Full-denoise token accuracy",
    )
    _write_posthoc_summary(work_root / "posthoc" / "igsm_ood", output_dir / "posthoc_summary.md")
    _write_sample_excerpts(runs_root, output_dir / "igsm_generation_samples.md")
    print(json.dumps({"artifacts": str(output_dir)}, sort_keys=True))


def _write_train_loss_chart(runs_root: Path, output: Path) -> None:
    series = {}
    for run_name, label in RUN_LABELS.items():
        points = _read_train_loss(runs_root / run_name)
        if points:
            series[label] = points
    _write_svg_chart(output, "Training loss", "step", "loss", series)
    _write_png_chart(output.with_suffix(".png"), "Training loss", "step", "loss", series)


def _write_periodic_chart(runs_root: Path, output: Path, metric: str, title: str) -> None:
    series = {}
    for run_name, label in RUN_LABELS.items():
        points = _read_periodic_metric(runs_root / run_name, metric)
        if points:
            series[label] = points
    _write_svg_chart(output, title, "step", metric.rsplit("/", 1)[-1], series)
    _write_png_chart(output.with_suffix(".png"), title, "step", metric.rsplit("/", 1)[-1], series)


def _read_train_loss(run_dir: Path) -> list[tuple[float, float]]:
    states = sorted(run_dir.glob("checkpoint-*/trainer_state.json"), key=_checkpoint_step)
    if not states:
        return []
    data = json.loads(states[-1].read_text(encoding="utf-8"))
    points = []
    seen = set()
    for row in data.get("log_history", []):
        if "step" not in row or "loss" not in row:
            continue
        step = int(row["step"])
        if step in seen:
            continue
        seen.add(step)
        points.append((float(step), float(row["loss"])))
    return points


def _read_periodic_metric(run_dir: Path, metric: str) -> list[tuple[float, float]]:
    path = run_dir / "periodic_eval_metrics.jsonl"
    if not path.exists():
        return []
    points = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "periodic/step" in row and metric in row:
            points.append((float(row["periodic/step"]), float(row[metric])))
    return points


def _write_svg_chart(output: Path, title: str, xlabel: str, ylabel: str, series: dict[str, list[tuple[float, float]]]) -> None:
    width, height = 900, 520
    margin_left, margin_right, margin_top, margin_bottom = 70, 210, 45, 65
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    all_points = [point for points in series.values() for point in points]
    if not all_points:
        output.write_text(f"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\"><text x=\"20\" y=\"40\">No data</text></svg>\n")
        return
    x_min = min(x for x, _ in all_points)
    x_max = max(x for x, _ in all_points)
    y_min = min(y for _, y in all_points)
    y_max = max(y for _, y in all_points)
    if x_min == x_max:
        x_max += 1.0
    if y_min == y_max:
        y_max += 1.0
    y_pad = 0.06 * (y_max - y_min)
    y_min -= y_pad
    y_max += y_pad
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b"]

    def sx(x: float) -> float:
        return margin_left + (x - x_min) / (x_max - x_min) * plot_w

    def sy(y: float) -> float:
        return margin_top + (y_max - y) / (y_max - y_min) * plot_h

    parts = [
        f"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\">",
        "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>",
        f"<text x=\"{margin_left}\" y=\"28\" font-family=\"sans-serif\" font-size=\"22\" font-weight=\"700\">{escape(title)}</text>",
        f"<line x1=\"{margin_left}\" y1=\"{margin_top + plot_h}\" x2=\"{margin_left + plot_w}\" y2=\"{margin_top + plot_h}\" stroke=\"#333\"/>",
        f"<line x1=\"{margin_left}\" y1=\"{margin_top}\" x2=\"{margin_left}\" y2=\"{margin_top + plot_h}\" stroke=\"#333\"/>",
        f"<text x=\"{margin_left + plot_w / 2}\" y=\"{height - 18}\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"13\">{escape(xlabel)}</text>",
        f"<text x=\"18\" y=\"{margin_top + plot_h / 2}\" transform=\"rotate(-90 18 {margin_top + plot_h / 2})\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"13\">{escape(ylabel)}</text>",
    ]
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        x_val = x_min + frac * (x_max - x_min)
        x = sx(x_val)
        parts.append(f"<line x1=\"{x:.1f}\" y1=\"{margin_top}\" x2=\"{x:.1f}\" y2=\"{margin_top + plot_h}\" stroke=\"#eee\"/>")
        parts.append(f"<text x=\"{x:.1f}\" y=\"{margin_top + plot_h + 20}\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"11\">{x_val/1000:.0f}k</text>")
        y_val = y_min + frac * (y_max - y_min)
        y = sy(y_val)
        parts.append(f"<line x1=\"{margin_left}\" y1=\"{y:.1f}\" x2=\"{margin_left + plot_w}\" y2=\"{y:.1f}\" stroke=\"#eee\"/>")
        parts.append(f"<text x=\"{margin_left - 10}\" y=\"{y + 4:.1f}\" text-anchor=\"end\" font-family=\"sans-serif\" font-size=\"11\">{y_val:.2f}</text>")
    for index, (label, points) in enumerate(series.items()):
        color = colors[index % len(colors)]
        polyline = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in points)
        parts.append(f"<polyline points=\"{polyline}\" fill=\"none\" stroke=\"{color}\" stroke-width=\"2.2\"/>")
        legend_y = margin_top + 22 * index
        legend_x = margin_left + plot_w + 28
        parts.append(f"<line x1=\"{legend_x}\" y1=\"{legend_y}\" x2=\"{legend_x + 24}\" y2=\"{legend_y}\" stroke=\"{color}\" stroke-width=\"3\"/>")
        parts.append(f"<text x=\"{legend_x + 32}\" y=\"{legend_y + 4}\" font-family=\"sans-serif\" font-size=\"12\">{escape(label)}</text>")
    parts.append("</svg>")
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _write_png_chart(output: Path, title: str, xlabel: str, ylabel: str, series: dict[str, list[tuple[float, float]]]) -> None:
    if not series:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=160)
    for label, points in series.items():
        xs = [x for x, _ in points]
        ys = [y for _, y in points]
        ax.plot(xs, ys, linewidth=1.8, label=label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, color="#dddddd", linewidth=0.6)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def _write_posthoc_summary(posthoc_dir: Path, output: Path) -> None:
    rows = []
    for path in sorted(posthoc_dir.glob("*summary.json")):
        if not any(token in path.name for token in ("latest_full_denoise", "x0_commit", "stepwise_partial_commit", "stepwise_oracle_mpc")):
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        rows.append((path.name, _extract_answer_metrics(data)))
    lines = ["# Posthoc Summary", ""]
    for name, metrics in rows:
        lines.append(f"## {name}")
        lines.append("")
        if not metrics:
            lines.append("_No answer-accuracy fields found._")
            lines.append("")
            continue
        lines.append("| Split | Answer accuracy | Token accuracy |")
        lines.append("| --- | ---: | ---: |")
        for split, answer, token in metrics:
            answer_text = "" if answer is None else f"{100 * answer:.1f}%"
            token_text = "" if token is None else f"{100 * token:.1f}%"
            lines.append(f"| {split} | {answer_text} | {token_text} |")
        lines.append("")
    output.write_text("\n".join(lines), encoding="utf-8")


def _extract_answer_metrics(data: dict) -> list[tuple[str, float | None, float | None]]:
    rows = []
    for key, value in sorted(data.items()):
        if not isinstance(value, dict):
            continue
        answer = _find_metric(value, "answer_accuracy")
        token = _find_metric(value, "token_accuracy")
        if answer is not None or token is not None:
            rows.append((key, answer, token))
    return rows


def _find_metric(row: dict, suffix: str) -> float | None:
    for key, value in row.items():
        if key.endswith(suffix) and isinstance(value, (int, float)):
            return float(value)
    return None


def _write_sample_excerpts(runs_root: Path, output: Path) -> None:
    lines = ["# Generation Sample Excerpts", ""]
    for run_name, label in RUN_LABELS.items():
        path = runs_root / run_name / "periodic_eval_samples.txt"
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        excerpt = _last_sample_block(text)
        lines.append(f"## {label}")
        lines.append("")
        lines.append("```text")
        lines.append(excerpt.strip()[:3500])
        lines.append("```")
        lines.append("")
    output.write_text("\n".join(lines), encoding="utf-8")


def _last_sample_block(text: str) -> str:
    matches = list(re.finditer(r"^=== step .*$", text, flags=re.MULTILINE))
    if matches:
        return text[matches[-1].start() :]
    return text[-3500:]


def _checkpoint_step(path: Path) -> int:
    match = re.search(r"checkpoint-(\d+)", str(path))
    return int(match.group(1)) if match else -1


if __name__ == "__main__":
    main()
