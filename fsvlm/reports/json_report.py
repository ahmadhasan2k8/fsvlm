"""JSON report generator."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from fsvlm.interfaces import ReportGenerator
from fsvlm.registry import report_generators
from fsvlm.types import ValidationReport


@report_generators.register("json")
class JSONReportGenerator(ReportGenerator):
    """Generates a JSON validation report."""

    def file_extension(self) -> str:
        return ".json"

    def generate(self, report: ValidationReport, output_path: Path) -> Path:
        """Write the validation report as JSON.

        Args:
            report: ValidationReport to serialize.
            output_path: Path for the output file.

        Returns:
            Path to the generated JSON file.
        """
        out = output_path.with_suffix(".json") if output_path.suffix != ".json" else output_path
        out.parent.mkdir(parents=True, exist_ok=True)

        data = asdict(report)
        # Convert Path objects to strings
        for fc in data.get("failure_cases", []):
            fc["image_path"] = str(fc["image_path"])

        out.write_text(json.dumps(data, indent=2, default=str))
        return out
