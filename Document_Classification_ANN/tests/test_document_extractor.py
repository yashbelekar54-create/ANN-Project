import io

import pytest

from src.document_extractor import (ExtractionError, NoExtractableTextError, UnsupportedFileTypeError, extract_text)


def test_txt_extraction_utf8_and_latin1():
    assert extract_text("a.txt", "Hello wörld".encode("utf-8")).text == "Hello wörld"
    assert "caf" in extract_text("b.txt", "café".encode("latin-1")).text


def test_empty_txt_is_rejected():
    with pytest.raises(NoExtractableTextError):
        extract_text("empty.txt", b"   \n ")
    with pytest.raises(NoExtractableTextError):
        extract_text("zero.txt", b"")


def test_unsupported_type_is_rejected():
    with pytest.raises(UnsupportedFileTypeError):
        extract_text("image.png", b"data")
    with pytest.raises(UnsupportedFileTypeError):
        extract_text("noextension", b"data")


def test_size_limit():
    with pytest.raises(ExtractionError):
        extract_text("big.txt", b"x" * (2 * 1024 * 1024), max_mb=1)


def test_docx_extraction():
    docx = pytest.importorskip("docx")
    d = docx.Document()
    d.add_paragraph("Space shuttle launch.")
    t = d.add_table(rows=1, cols=1)
    t.cell(0, 0).text = "Table cell text"
    buf = io.BytesIO()
    d.save(buf)
    res = extract_text("doc.docx", buf.getvalue())
    assert "Space shuttle launch." in res.text and "Table cell text" in res.text and res.file_type == "docx"


def test_empty_docx_is_rejected():
    docx = pytest.importorskip("docx")
    buf = io.BytesIO()
    docx.Document().save(buf)
    with pytest.raises(NoExtractableTextError):
        extract_text("empty.docx", buf.getvalue())


def test_corrupt_docx_is_handled():
    pytest.importorskip("docx")
    with pytest.raises(ExtractionError):
        extract_text("bad.docx", b"this is not a zip file")


def test_pdf_extraction_and_blank_pdf():
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "Hockey game tonight at the arena")
    res = extract_text("a.pdf", doc.tobytes())
    assert "Hockey" in res.text and res.n_pages == 1

    blank = fitz.open()
    blank.new_page()
    with pytest.raises(NoExtractableTextError):
        extract_text("scan.pdf", blank.tobytes())


def test_corrupt_pdf_is_handled():
    pytest.importorskip("fitz")
    with pytest.raises(ExtractionError):
        extract_text("bad.pdf", b"not a pdf")
