"""Inspect PDFs, search passages, and generate grounded answers with citations."""

import logging

import streamlit as st

from src.chat_ui import render_answers
from src.config import get_settings
from src.sidebar_ui import render_sidebar
from src.source_ui import location, source_access
from src.vector_store import DocumentLibrary

logger = logging.getLogger(__name__)
st.set_page_config(page_title="Document Explorer", page_icon="📚", layout="wide")


@st.cache_resource
def get_library():
    return DocumentLibrary(get_settings())


library = get_library()
st.title("Document Explorer")
st.write("Chat with your PDFs and webpages, with evidence behind each answer.")
st.markdown(
    """<style>
    .stMainBlockContainer {max-width: 1080px; padding-top: 2rem; padding-bottom: 2rem;}
    [data-testid="stChatMessage"] {border-radius: 16px; padding: 1.1rem; margin: .65rem 0;}
    [data-testid="stChatMessage"] p {line-height: 1.7;}
    [data-testid="stChatInput"] {border-radius: 16px;}
    [data-testid="stMetricValue"] {font-size: 1.5rem;}
    [data-testid="stSidebar"] [data-testid="stMetricLabel"] {font-size: .76rem;}
    [data-testid="stSidebar"] [data-testid="stExpander"] {border-radius: 12px;}
    [data-testid="stSidebar"] [data-testid="stExpander"] summary p {
        overflow-wrap: anywhere; line-height: 1.4; font-size: .88rem;
    }
    .st-key-library_summary [data-testid="stHorizontalBlock"] {
        display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: .5rem;
    }
    .st-key-library_summary [data-testid="stColumn"] {min-width: 0 !important; width: auto !important;}
    .st-key-library_summary [data-testid="stMetric"] {
        padding: .6rem .45rem; border: 1px solid #304253; border-radius: 10px;
        background: #101b29;
    }
    [data-testid="stSidebar"] hr {margin: 1rem 0;}
    .st-key-source_list {scrollbar-width: thin;}
    .st-key-source_list [data-testid="stExpander"] {background: #101b29;}
    .st-key-source_list [class*="st-key-remove-source-"] button {color: #fca5a5;}
    .st-key-source_list [class*="st-key-confirm-remove-"] button {
        background: #7f1d1d; border-color: #ef4444; color: #fee2e2;
    }
    [data-testid="stSidebar"] h3 {font-size: 1.12rem;}
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {line-height: 1.5;}
    </style>""",
    unsafe_allow_html=True,
)

sources = render_sidebar(library)
by_id = {source["document_id"]: source for source in sources}


def source_label(document_id):
    if document_id == "all":
        return "All documents"
    return f"{by_id[document_id]['filename']} · {document_id[:8]}"


if not sources:
    st.info("Upload a text-based PDF in the sidebar, or paste a public webpage URL to get started.")

answer_tab, search_tab, inspect_tab = st.tabs(
    ["Chat", "Search passages", "Inspect documents"], key="workspace_tabs", on_change="rerun"
)
with answer_tab:
    render_answers(library, by_id)
with search_tab:
    with st.form("search"):
        query = st.text_input("Question or search phrase", placeholder="What is the refund policy?")
        selected = st.selectbox("Search within", ["all", *by_id], format_func=source_label)
        k = st.slider("Number of passages", 1, 10, 4)
        submitted = st.form_submit_button("Search", disabled=not sources, type="primary")
    if submitted:
        try:
            with st.spinner("Searching your documents…"):
                results = library.search(
                    query, k=k, document_id=None if selected == "all" else selected
                )
            st.session_state["search_result"] = (query, results)
        except ValueError as exc:
            st.session_state.pop("search_result", None)
            st.warning(str(exc))
        except Exception:
            logger.exception("Search failed")
            st.session_state.pop("search_result", None)
            st.error("Search failed. Check the terminal for details, then try again.")
    if "search_result" in st.session_state:
        searched, results = st.session_state["search_result"]
        st.write(f"Results for: {searched}")
        st.caption(
            "Cosine distance: lower means closer. This is not a confidence score. "
            "Search returns the nearest passages even when none answers the question."
        )
        for rank, (document, distance) in enumerate(results, 1):
            meta = document.metadata
            with st.container(border=True):
                st.subheader(f"{rank}. {meta['source']} — {location(meta)}")
                st.caption(f"Chunk {meta['chunk_number']} · Cosine distance {distance:.4f}")
                st.text(document.page_content)
                if meta.get("source_type") == "url":
                    source_access(library, meta, key=f"search-source-{rank}")
                with st.expander("Chunk metadata"):
                    st.json(meta)

with inspect_tab:
    if sources:
        inspected = st.selectbox(
            "Document", list(by_id), format_func=source_label, key="inspect_source"
        )
        source = by_id[inspected]
        if source["empty_pages"]:
            st.warning(f"Pages without extractable text: {source['empty_pages']}.")
        pages = library.read_pages(inspected)
        source_access(library, pages[0].metadata, key="inspect-source")
        page_number = (
            1
            if source.get("source_type") == "url"
            else st.number_input(
                "PDF page", min_value=1, max_value=len(pages), step=1, key=f"page-{inspected}"
            )
        )
        page = pages[page_number - 1]
        st.caption(location(page.metadata))
        st.text(page.page_content or "No extractable text on this page.")
        chunks = [
            chunk
            for chunk in library.read_chunks(inspected)
            if chunk.metadata["page_number"] == page_number
        ]
        st.write(
            f"{len(chunks)} chunks in this {'webpage' if source.get('source_type') == 'url' else 'page'}"
        )
        for chunk in chunks:
            with st.expander(
                f"Chunk {chunk.metadata['chunk_number']} · {chunk.metadata['token_count']} tokens"
            ):
                st.text(chunk.page_content)
                st.json(chunk.metadata)

with st.expander("Pipeline settings"):
    st.json(
        {
            key: str(value) if key == "data_dir" else value
            for key, value in library.describe().items()
        }
    )
