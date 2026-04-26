"""Tests for the DeepPCB benchmark reader."""

from __future__ import annotations

from pathlib import Path

import pytest

from fsvlm.readers.deeppcb_reader import DeepPCBReader, read_deeppcb


def _make_deeppcb_fixture(tmp_path: Path) -> Path:
    """Build a minimal DeepPCB-shaped directory with one pair."""
    root = tmp_path / "deeppcb"
    pcb_data = root / "PCBData"
    group = pcb_data / "group20085"
    (group / "20085").mkdir(parents=True)
    (group / "20085_not").mkdir(parents=True)

    test_img = group / "20085" / "20085000_test.jpg"
    temp_img = group / "20085" / "20085000_temp.jpg"
    ann_file = group / "20085_not" / "20085000.txt"

    test_img.write_bytes(b"fake-jpeg")
    temp_img.write_bytes(b"fake-jpeg")
    ann_file.write_text("144 477 178 508 1\n188 229 227 322 2\n353 395 390 427 5\n")

    (pcb_data / "trainval.txt").write_text(
        "group20085/20085/20085000.jpg group20085/20085_not/20085000.txt\n"
    )
    (pcb_data / "test.txt").write_text("")
    return root


def test_deeppcb_reader_supports(tmp_path: Path) -> None:
    root = _make_deeppcb_fixture(tmp_path)
    assert DeepPCBReader().supports(root) is True


def test_deeppcb_reader_rejects_random_dir(tmp_path: Path) -> None:
    assert DeepPCBReader().supports(tmp_path) is False


def test_read_deeppcb_produces_pair(tmp_path: Path) -> None:
    root = _make_deeppcb_fixture(tmp_path)
    samples = read_deeppcb(root, split="trainval")
    labels = sorted(s.label for s in samples)
    assert labels == ["defect", "good"]
    assert len(samples) == 2


def test_read_deeppcb_description_lists_defect_names(tmp_path: Path) -> None:
    root = _make_deeppcb_fixture(tmp_path)
    samples = read_deeppcb(root, split="trainval")
    defect_samples = [s for s in samples if s.label == "defect"]
    assert len(defect_samples) == 1
    desc = defect_samples[0].description.lower()
    # Annotations have classes 1 (open), 2 (short), 5 (copper) in our fixture
    assert "open" in desc
    assert "short" in desc
    assert "copper" in desc


def test_read_deeppcb_unknown_split(tmp_path: Path) -> None:
    root = _make_deeppcb_fixture(tmp_path)
    from fsvlm.exceptions import DatasetError

    with pytest.raises(DatasetError):
        read_deeppcb(root, split="bogus")


def test_read_deeppcb_missing_root(tmp_path: Path) -> None:
    from fsvlm.exceptions import DatasetError

    with pytest.raises(DatasetError):
        read_deeppcb(tmp_path / "nope")
