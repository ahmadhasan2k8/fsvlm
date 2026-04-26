"""HTML report generator using Jinja2."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from fsvlm.interfaces import ReportGenerator
from fsvlm.registry import report_generators
from fsvlm.types import ValidationReport

TEMPLATE_DIR = Path(__file__).parent / "templates"


@report_generators.register("html")
class HTMLReportGenerator(ReportGenerator):
    """Generates an HTML validation report with Jinja2."""

    def file_extension(self) -> str:
        return ".html"

    def generate(self, report: ValidationReport, output_path: Path) -> Path:
        """Render the validation report as HTML.

        Args:
            report: ValidationReport to render.
            output_path: Path for the output file.

        Returns:
            Path to the generated HTML file.
        """
        from jinja2 import Environment, FileSystemLoader

        out = output_path.with_suffix(".html") if output_path.suffix != ".html" else output_path
        out.parent.mkdir(parents=True, exist_ok=True)

        env = Environment(
            loader=FileSystemLoader(str(TEMPLATE_DIR)),
            autoescape=True,
        )
        env.filters["basename"] = lambda p: Path(str(p)).name

        template = env.get_template("validation.html")

        data = asdict(report)
        # Convert Path objects to strings for Jinja
        for fc in data.get("failure_cases", []):
            fc["image_path"] = str(fc["image_path"])

        html = template.render(**data)
        out.write_text(html)
        return out
