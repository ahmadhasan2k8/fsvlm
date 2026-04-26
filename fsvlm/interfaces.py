"""Abstract base classes for extension points.

ABCs introduced in Phase 2 because we now have 3 reader implementations
(folder, CSV, JSON) and 2 report generators (HTML, JSON) — the abstraction
is earned by need, not speculation.

ModelBackend and TrainingBackend ABCs deferred to Phase 4 (still only one impl each).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from fsvlm.types import LabeledSample, ValidationReport


class LabelReader(ABC):
    """Abstraction for reading labeled image data from various formats."""

    @abstractmethod
    def read(self, path: Path) -> list[LabeledSample]:
        """Read labeled samples from the given path.

        Args:
            path: File or directory containing labeled data.

        Returns:
            List of LabeledSample objects.
        """
        ...

    @abstractmethod
    def supports(self, path: Path) -> bool:
        """Check if this reader can handle the given path.

        Args:
            path: File or directory to check.

        Returns:
            True if this reader can read the given path.
        """
        ...


class ReportGenerator(ABC):
    """Abstraction for generating validation reports in different formats."""

    @abstractmethod
    def generate(self, report: ValidationReport, output_path: Path) -> Path:
        """Generate a report file from validation results.

        Args:
            report: The validation report to render.
            output_path: Where to write the report file.

        Returns:
            Path to the generated report file.
        """
        ...

    @abstractmethod
    def file_extension(self) -> str:
        """Return the file extension for this report format (e.g., '.html')."""
        ...
