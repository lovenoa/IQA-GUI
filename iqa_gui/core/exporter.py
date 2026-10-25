"""Export evaluation results to CSV and JSON formats."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from iqa_gui.core.batch_evaluator import BatchItem, BatchSummary


class ResultExporter:
    """Handles formatted export of IQA evaluation results."""

    @staticmethod
    def export_to_csv(
        file_path: str,
        items: List[BatchItem],
        metrics: List[str],
        summary: Optional[BatchSummary] = None,
    ) -> None:
        """Export batch items and optional summary to CSV."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)

            # Metadata header
            writer.writerow(["# IQA-PyTorch 图像质量评估导出报表"])
            writer.writerow(["# 导出时间", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
            if summary:
                writer.writerow(["# 总图像数", summary.total_images, "成功", summary.completed_images, "失败", summary.failed_images])
                writer.writerow(["# 总耗时(秒)", f"{summary.total_time_seconds:.2f}"])
            writer.writerow([])

            # Table Header
            headers = ["序号", "目标图像", "参考图像", "状态", "耗时(秒)"] + [f"{m} 得分" for m in metrics] + ["错误信息"]
            writer.writerow(headers)

            # Rows
            for item in items:
                row = [
                    item.index,
                    item.target_filename,
                    item.ref_filename or "-",
                    item.status,
                    f"{item.time_taken:.3f}",
                ]
                for m in metrics:
                    val = item.scores.get(m)
                    row.append(f"{val:.4f}" if val is not None else "-")
                row.append(item.error_msg or "")
                writer.writerow(row)

            # Summary statistical row if present
            if summary and summary.stats:
                writer.writerow([])
                writer.writerow(["# 统计摘要 (Summary Statistics)"])
                for stat_name, label in [("mean", "平均值 (Mean)"), ("std", "标准差 (Std)"), ("min", "最小值 (Min)"), ("max", "最大值 (Max)")]:
                    row = ["-", label, "-", "-", "-"]
                    for m in metrics:
                        val = summary.stats.get(m, {}).get(stat_name)
                        row.append(f"{val:.4f}" if val is not None else "-")
                    writer.writerow(row)

    @staticmethod
    def export_to_json(
        file_path: str,
        items: List[BatchItem],
        metrics: List[str],
        summary: Optional[BatchSummary] = None,
    ) -> None:
        """Export batch evaluation data to structured JSON."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "metrics": metrics,
            },
            "summary": {
                "total": summary.total_images if summary else len(items),
                "completed": summary.completed_images if summary else sum(1 for i in items if i.status == "completed"),
                "failed": summary.failed_images if summary else sum(1 for i in items if i.status == "failed"),
                "total_time_seconds": summary.total_time_seconds if summary else 0.0,
                "stats": summary.stats if summary else {},
            },
            "items": [
                {
                    "index": item.index,
                    "target_path": item.target_path,
                    "target_filename": item.target_filename,
                    "ref_path": item.ref_path,
                    "ref_filename": item.ref_filename,
                    "status": item.status,
                    "time_taken": item.time_taken,
                    "scores": item.scores,
                    "error_msg": item.error_msg,
                }
                for item in items
            ],
        }

        # Validate before opening/truncating an existing report.
        payload = json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)
        with open(path, "w", encoding="utf-8") as f:
            f.write(payload)
