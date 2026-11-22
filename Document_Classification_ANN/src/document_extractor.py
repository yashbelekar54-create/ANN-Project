"""Text extraction from uploaded TXT, PDF and DOCX files."""
from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import Optional

SUPPORTED_EXTENSIONS = (".txt", ".pdf", ".docx")


class ExtractionError(Exception):
    """Base class for extraction problems that should be shown to the user."""


class UnsupportedFileTypeError(ExtractionError):
    pass


class NoExtractableTextError(ExtractionError):
    pass


@dataclass
class ExtractionResult:
    filename: str
    file_type: str
    text: str
    n_pages: Optional[int] = None

    @property
    def n_words(self) -> int:
        return len(self.text.split())


def _decode_text(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _extract_pdf(data: bytes) -> ExtractionResult:
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ExtractionError("PDF support requires PyMuPDF (pip install PyMuPDF).") from exc
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise ExtractionError(f"The PDF could not be opened ({exc}).") from exc
    try:
        if doc.needs_pass:
            raise ExtractionError("The PDF is password-protected and cannot be read.")
        pages = [page.get_text("text") for page in doc]
        n_pages = len(pages)
    finally:
        doc.close()
    text = "\n".join(pages).strip()
    if not text:
        raise NoExtractableTextError(
            "This PDF contains no extractable text. It is probably a scanned document; "
            "OCR is not supported by this application."
        )
    return ExtractionResult("", "pdf", text, n_pages)


def _extract_docx(data: bytes) -> ExtractionResult:
    try:
        import docx
    except ImportError as exc:  # pragma: no cover
        raise ExtractionError("DOCX support requires python-docx (pip install python-docx).") from exc
    try:
        document = docx.Document(io.BytesIO(data))
    except Exception as exc:
        raise ExtractionError(f"The DOCX file could not be opened ({exc}).") from exc
    parts = [p.text for p in document.paragraphs if p.text.strip()]
    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                if cell.text.strip():
                    parts.append(cell.text)
    text = "\n".join(parts).strip()
    if not text:
        raise NoExtractableTextError("The DOCX file contains no text.")
    return ExtractionResult("", "docx", text)


def extract_text(filename: str, data: bytes, max_mb: Optional[float] = None) -> ExtractionResult:
    """Extract text from an uploaded file.

    Raises :class:`UnsupportedFileTypeError`, :class:`NoExtractableTextError` or
    :class:`ExtractionError` with a message suitable for display to the user.
    """
    ext = os.path.splitext(filename or "")[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{ext or 'unknown'}'. Supported types: {', '.join(SUPPORTED_EXTENSIONS)}."
        )
    if not data:
        raise NoExtractableTextError("The file is empty.")
    if max_mb is not None and len(data) > max_mb * 1024 * 1024:
        raise ExtractionError(f"The file is larger than the {max_mb:g} MB limit.")

    if ext == ".txt":
        text = _decode_text(data).strip()
        if not text:
            raise NoExtractableTextError("The text file is empty.")
        result = ExtractionResult("", "txt", text)
    elif ext == ".pdf":
        result = _extract_pdf(data)
    else:
        result = _extract_docx(data)
    result.filename = filename
    return result
