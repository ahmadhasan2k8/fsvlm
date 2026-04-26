"""Tests for fsvlm.interfaces."""

from __future__ import annotations

import pytest

from fsvlm.interfaces import LabelReader, ReportGenerator


def test_label_reader_is_abstract():
    with pytest.raises(TypeError):
        LabelReader()  # type: ignore[abstract]


def test_report_generator_is_abstract():
    with pytest.raises(TypeError):
        ReportGenerator()  # type: ignore[abstract]


def test_folder_reader_implements_abc():
    from fsvlm.readers.folder_reader import FolderLabelReader

    reader = FolderLabelReader()
    assert isinstance(reader, LabelReader)


def test_csv_reader_implements_abc():
    from fsvlm.readers.csv_reader import CSVLabelReader

    reader = CSVLabelReader()
    assert isinstance(reader, LabelReader)


def test_json_reader_implements_abc():
    from fsvlm.readers.json_reader import JSONLabelReader

    reader = JSONLabelReader()
    assert isinstance(reader, LabelReader)


def test_html_report_implements_abc():
    from fsvlm.reports.html_report import HTMLReportGenerator

    gen = HTMLReportGenerator()
    assert isinstance(gen, ReportGenerator)


def test_json_report_implements_abc():
    from fsvlm.reports.json_report import JSONReportGenerator

    gen = JSONReportGenerator()
    assert isinstance(gen, ReportGenerator)
