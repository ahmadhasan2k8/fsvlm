"""Tests for fsvlm.registry."""

from __future__ import annotations

import pytest

from fsvlm.registry import Registry


def test_register_and_get():
    reg = Registry()

    @reg.register("test")
    class TestClass:
        pass

    assert reg.get("test") is TestClass


def test_get_unknown_raises():
    reg = Registry()
    with pytest.raises(KeyError, match="Unknown"):
        reg.get("nonexistent")


def test_list_available():
    reg = Registry()

    @reg.register("a")
    class A:
        pass

    @reg.register("b")
    class B:
        pass

    assert sorted(reg.list_available()) == ["a", "b"]


def test_all_classes():
    reg = Registry()

    @reg.register("x")
    class X:
        pass

    assert reg.all_classes() == [X]


def test_global_registries():
    from fsvlm.readers.csv_reader import CSVLabelReader  # noqa: F401

    # Import readers/reports to populate
    from fsvlm.readers.folder_reader import FolderLabelReader  # noqa: F401
    from fsvlm.readers.json_reader import JSONLabelReader  # noqa: F401
    from fsvlm.registry import label_readers, report_generators
    from fsvlm.reports.html_report import HTMLReportGenerator  # noqa: F401
    from fsvlm.reports.json_report import JSONReportGenerator  # noqa: F401

    assert "folder" in label_readers.list_available()
    assert "csv" in label_readers.list_available()
    assert "json" in label_readers.list_available()
    assert "html" in report_generators.list_available()
    assert "json" in report_generators.list_available()
