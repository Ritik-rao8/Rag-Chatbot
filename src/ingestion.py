"""Extract each PDF page independently so citations retain their provenance."""

import hashlib
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader


class PDFError(ValueError):
    """An actionable PDF input error."""


@dataclass
class ExtractedSource:
    document_id: str
    filename: str
    pages: list[Document]
    total_pages: int
    empty_pages: list[int]
    source_type: str = "pdf"
    url: str = ""
    fetched_at: str = ""


def extract_pdf(
    content: bytes, filename: str, *, max_bytes: int = 20 * 1024 * 1024, max_pages: int = 300
) -> ExtractedSource:
    if not content:
        raise PDFError("This file is empty. Choose a text-based PDF.")
    if len(content) > max_bytes:
        raise PDFError(f"This PDF exceeds the {max_bytes // (1024 * 1024)} MB limit.")
    if not content[:1024].lstrip().startswith(b"%PDF-"):
        raise PDFError("This file does not look like a valid PDF.")

    document_id = hashlib.sha256(content).hexdigest()
    filename = Path(filename.replace("\\", "/")).name or "document.pdf"
    try:
        reader = PdfReader(BytesIO(content))
        if reader.is_encrypted:
            raise PDFError("Password-protected PDFs are not supported. Upload an unlocked copy.")
        total_pages = len(reader.pages)
        if total_pages > max_pages:
            raise PDFError(f"This PDF has {total_pages} pages; the limit is {max_pages}.")
        labels = reader.page_labels
        pages = []
        empty_pages = []
        for number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").replace("\x00", "").strip()
            if not text:
                empty_pages.append(number)
            pages.append(
                Document(
                    page_content=text,
                    metadata={
                        "document_id": document_id,
                        "source": filename,
                        "page_number": number,
                        "page_label": str(labels[number - 1]),
                        "total_pages": total_pages,
                    },
                )
            )
    except PDFError:
        raise
    except Exception as exc:
        raise PDFError(
            "Could not read this PDF. It may be damaged or use unsupported encoding."
        ) from exc

    if not any(page.page_content for page in pages):
        raise PDFError(
            "No extractable text was found. This may be a scanned PDF; OCR is a later milestone."
        )
    return ExtractedSource(document_id, filename, pages, total_pages, empty_pages)
