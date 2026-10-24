"""Shared source labels; webpages do not have PDF page numbers."""

import streamlit as st


def location(metadata):
    if metadata.get("source_type") == "url":
        return "Webpage"
    return f"PDF page {metadata['page_number']}"


def source_access(library, metadata, *, key):
    if metadata.get("source_type") == "url":
        st.link_button("Open original webpage", metadata["url"])
        st.caption(f"Indexed snapshot: {metadata.get('fetched_at', '')}")
    else:
        path = library.pdf_path(metadata["document_id"])
        if path.exists():
            st.download_button(
                "Download source PDF",
                path.read_bytes(),
                file_name=metadata["source"],
                mime="application/pdf",
                key=key,
            )


def sync_library_state(source_ids):
    signature = tuple(sorted(source_ids))
    previous = st.session_state.get("library_signature")
    if previous is not None and previous != signature:
        st.session_state.pop("answer_turns", None)
        st.session_state.pop("search_result", None)
        st.session_state.pop("answer_source", None)
        st.session_state.pop("retry_turn_id", None)
        st.session_state.pop("pending_source_delete", None)
        st.session_state.pop("pdf_preview_request", None)
        st.session_state.pop("pdf_preview_active", None)
        for key in (
            "active_conversation_id",
            "active_conversation_title",
            "conversation_name",
            "conversation_dirty",
        ):
            st.session_state.pop(key, None)
    st.session_state["library_signature"] = signature
