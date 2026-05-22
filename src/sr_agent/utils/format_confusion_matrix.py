from __future__ import annotations

from typing import Mapping, Optional

__all__ = ["format_confusion_matrix"]


def color_by_error_rate(text: str, error_rate: float) -> str:
    if error_rate >= 0.30:
        return f"[red]{text}[reset]"
    if error_rate >= 0.10:
        return f"[orange]{text}[reset]"
    if error_rate > 0.00:
        return f"[yellow]{text}[reset]"
    return f"[green]{text}[reset]"


def format_confusion_matrix(
    matrix: Mapping[str, Mapping[str, int]] | None,
    *,
    max_size: int = 12,
    topk: int = 10,
) -> Optional[str]:
    if not matrix:
        return None

    labels = list(matrix)
    seen_labels = set(labels)
    for row in matrix.values():
        for label in row:
            if label not in seen_labels:
                labels.append(label)
                seen_labels.add(label)

    row_totals = {
        gt_label: sum(matrix.get(gt_label, {}).values())
        for gt_label in labels
    }

    if len(labels) <= max_size:
        short_labels = [label[:8] for label in labels]
        width = max(5, min(8, max(len(label) for label in short_labels)))
        lines = ["Confusion Matrix (GT rows x Pred cols)"]
        lines.append(" " * (width + 1) + " ".join(f"{label:>{width}}" for label in short_labels))
        for gt_label, short_label in zip(labels, short_labels):
            row = matrix.get(gt_label, {})
            row_total = row_totals[gt_label]
            cells = []
            for pred_label in labels:
                count = row.get(pred_label, 0)
                if gt_label == pred_label:
                    cells.append(f"[green]{count:>{width}}[reset]")
                else:
                    error_rate = 0.0 if row_total == 0 else count / row_total
                    cells.append(color_by_error_rate(f"{count:>{width}}", error_rate))
            lines.append(f"{short_label:>{width}} " + " ".join(cells))
        return "\n".join(lines)

    errors = []
    for gt_label, row in matrix.items():
        row_total = row_totals[gt_label]
        if row_total == 0:
            continue
        for pred_label, count in row.items():
            if gt_label != pred_label and count > 0:
                errors.append((count / row_total, count, gt_label, pred_label, row_total))
    errors.sort(key=lambda item: (item[0], item[1]), reverse=True)
    if not errors:
        return "Top Confusion Errors: none"

    lines = [f"Top {min(topk, len(errors))} Confusion Errors"]
    for error_rate, count, gt_label, pred_label, row_total in errors[:topk]:
        text = f"{gt_label} -> {pred_label}: {count}/{row_total} ({error_rate:.1%})"
        lines.append(color_by_error_rate(text, error_rate))
    return "\n".join(lines)
