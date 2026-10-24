"""Render one local PDF page at a time; never send PDFs to an external viewer."""

import io
import math
from threading import RLock
from uuid import uuid4

import streamlit as st

# PDFium is not thread-safe, including calls using different documents.
PDFIUM_LOCK = RLock()
MAX_PIXELS = 12_000_000


@st.cache_data(max_entries=12, ttl=300, show_spinner=False)
def render_page(path: str, modified_ns: int, size_bytes: int, page_number: int):
    import pypdfium2 as pdfium

    with PDFIUM_LOCK, pdfium.PdfDocument(path) as pdf:
        if not 1 <= page_number <= len(pdf):
            raise ValueError("The requested PDF page is not available.")
        page = pdf[page_number - 1]
        try:
            width, height = page.get_size()
            if width <= 0 or height <= 0:
                raise ValueError("This PDF page cannot be displayed.")
            scale = min(1400 / width, 3, math.sqrt(MAX_PIXELS / (width * height)))
            bitmap = page.render(scale=scale)
            try:
                image = bitmap.to_pil()
                output = io.BytesIO()
                image.save(output, format="PNG")
                return output.getvalue(), len(pdf)
            finally:
                bitmap.close()
        finally:
            page.close()


def queue_pdf_preview(document_id, page_number=1, quote=""):
    st.session_state["pdf_preview_request"] = {
        "document_id": document_id,
        "page_number": page_number,
        "quote": quote,
        "token": uuid4().hex,
    }


def change_page(key, delta):
    st.session_state[key] += delta


def render_page_view(library, document_id, page_number, *, key):
    path = library.pdf_path(document_id)
    if not path.is_file():
        st.info("The original PDF is no longer available in this library.")
        return
    try:
        info = path.stat()
        image, count = render_page(str(path), info.st_mtime_ns, info.st_size, page_number)
        st.image(image, caption=f"PDF page {page_number} of {count}", width="stretch")
    except Exception:  # noqa: BLE001 - renderer may include private paths in errors
        st.warning("This PDF page could not be rendered. You can download the source to view it.")


def close_preview():
    st.session_state.pop("pdf_preview_active", None)


@st.dialog("PDF preview", width="large", on_dismiss=close_preview)
def preview_dialog(library, request):
    sources = {s["document_id"]: s for s in library.list_sources()}
    source = sources.get(request["document_id"])
    if not source or source.get("source_type", "pdf") != "pdf":
        st.info("The original PDF is no longer available in this library.")
        return
    st.text(source["filename"])
    total = source["total_pages"]
    start = min(max(1, request["page_number"]), total)
    key = f"preview-page-{request['token']}"
    st.session_state.setdefault(key, start)
    controls = st.columns([1, 2, 1], vertical_alignment="bottom")
    with controls[0]:
        st.button(
            "Previous",
            key=f"previous-{key}",
            disabled=st.session_state[key] <= 1,
            on_click=change_page,
            args=(key, -1),
            width="stretch",
        )
    with controls[1]:
        page = st.number_input("PDF page", min_value=1, max_value=total, step=1, key=key)
    with controls[2]:
        st.button(
            "Next",
            key=f"next-{key}",
            disabled=st.session_state[key] >= total,
            on_click=change_page,
            args=(key, 1),
            width="stretch",
        )
    if request.get("quote"):
        with st.expander("Cited excerpt"):
            st.text(request["quote"])
    render_page_view(library, source["document_id"], page, key=request["token"])
    try:
        original = library.pdf_path(source["document_id"]).read_bytes()
    except OSError:
        st.info("The original PDF is no longer available in this library.")
        return
    st.download_button(
        "Download source PDF",
        original,
        file_name=source["filename"],
        mime="application/pdf",
        key=f"preview-download-{request['token']}",
    )
    st.caption("Preview rendered locally. PDF page positions may differ from printed page labels.")


def render_requested_preview(library):
    request = st.session_state.pop("pdf_preview_request", None)
    if request:
        st.session_state["pdf_preview_active"] = request
    if st.session_state.get("pdf_preview_active"):
        preview_dialog(library, st.session_state["pdf_preview_active"])
