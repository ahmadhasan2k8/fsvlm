"""Feedback Agent — logs user corrections and triggers retraining.

Corrections are stored as JSONL files in ~/.fsvlm/corrections/.
When enough corrections accumulate, the agent suggests retraining.
On retrain, corrections merge with original training data to produce
a new adapter version.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from fsvlm.config import FSVLMConfig
from fsvlm.types import Correction, LabeledSample

CORRECTIONS_FILENAME = "corrections.jsonl"


class FeedbackAgent:
    """Manages user corrections and retrain triggers.

    Args:
        config: FSVLM configuration.
    """

    def __init__(self, config: FSVLMConfig) -> None:
        self._config = config

    @property
    def _corrections_file(self) -> Path:
        return self._config.corrections_dir / CORRECTIONS_FILENAME

    def log_correction(self, correction: Correction) -> int:
        """Log a user correction.

        Args:
            correction: The correction to log.

        Returns:
            Total number of pending corrections after logging.
        """
        self._config.corrections_dir.mkdir(parents=True, exist_ok=True)

        if not correction.timestamp:
            correction.timestamp = datetime.now(timezone.utc).isoformat()

        record = {
            "image_path": str(correction.image_path),
            "predicted_label": correction.predicted_label,
            "actual_label": correction.actual_label,
            "confidence": correction.confidence,
            "timestamp": correction.timestamp,
            "adapter_name": correction.adapter_name,
            "adapter_version": correction.adapter_version,
            "notes": correction.notes,
        }

        with self._corrections_file.open("a") as f:
            f.write(json.dumps(record) + "\n")

        count = self.pending_count()
        logger.info(f"Correction logged ({count} total pending)")

        if count >= self._config.correction_retrain_threshold:
            logger.info(
                f"Retrain threshold reached ({count} >= "
                f"{self._config.correction_retrain_threshold}). "
                f"Run: fsvlm retrain"
            )

        return count

    def load_corrections(self) -> list[Correction]:
        """Load all pending corrections.

        Returns:
            List of Correction objects.
        """
        if not self._corrections_file.exists():
            return []

        corrections: list[Correction] = []
        for line in self._corrections_file.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                corrections.append(
                    Correction(
                        image_path=Path(data["image_path"]),
                        predicted_label=data.get("predicted_label", ""),
                        actual_label=data.get("actual_label", ""),
                        confidence=data.get("confidence", 0.0),
                        timestamp=data.get("timestamp", ""),
                        adapter_name=data.get("adapter_name", ""),
                        adapter_version=data.get("adapter_version", 1),
                        notes=data.get("notes", ""),
                    )
                )
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Skipping corrupt correction line: {e}")

        return corrections

    def pending_count(self) -> int:
        """Count pending corrections without loading all of them."""
        if not self._corrections_file.exists():
            return 0
        return sum(1 for line in self._corrections_file.read_text().splitlines() if line.strip())

    def should_retrain(self) -> bool:
        """Check if enough corrections have accumulated to trigger retrain."""
        return self.pending_count() >= self._config.correction_retrain_threshold

    def corrections_as_samples(self) -> list[LabeledSample]:
        """Convert corrections to LabeledSamples for retraining.

        Returns:
            List of LabeledSample with corrected labels.
        """
        corrections = self.load_corrections()
        samples: list[LabeledSample] = []

        for c in corrections:
            if not c.image_path.exists():
                logger.warning(f"Correction image missing, skipping: {c.image_path}")
                continue
            samples.append(
                LabeledSample(
                    image_path=c.image_path,
                    label=c.actual_label,
                    description=f"Corrected from {c.predicted_label} to {c.actual_label}",
                )
            )

        return samples

    def clear_corrections(self) -> None:
        """Archive corrections after a successful retrain."""
        if not self._corrections_file.exists():
            return

        # Rename to timestamped archive
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        archive = self._corrections_file.with_name(f"corrections_{ts}.jsonl")
        self._corrections_file.rename(archive)
        logger.info(f"Corrections archived to {archive.name}")

    def summarize(self) -> dict[str, int]:
        """Summarize correction patterns.

        Returns:
            Dict mapping pattern descriptions to counts.
        """
        corrections = self.load_corrections()
        if not corrections:
            return {}

        patterns: dict[str, int] = {}
        for c in corrections:
            key = f"{c.predicted_label} -> {c.actual_label}"
            patterns[key] = patterns.get(key, 0) + 1

        return patterns
