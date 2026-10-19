"""Library sidebar: add sources, find sources, and act on a specific source."""

import logging
from datetime import datetime
from urllib.parse import urlsplit

import streamlit as st

from src.chat_ui import escape_markdown
from src.ingestion import PDFError
from src.source_ui import sync_library_state
from src.web_ingestion import WebError

logger = logging.getLogger(__name__)


def matching_sources(sources, query="", source_type="All"):
    query = query.strip().casefold()
    wanted = {"PDFs": "pdf", "Webpages": "url"}.get(source_type)
    return sorted(
        [
            source
            for source in sources
            if (not wanted or source.get("source_type", "pdf") == wanted)
            and (
                not query
                or query in source["filename"].casefold()
                or query in source.get("url", "").casefold()
            )
        ],
        key=lambda source: (source["filename"].casefold(), source["document_id"]),
    )


def readable_size(size):
    return (
        f"{size / 1024 / 1024:.1f} MB"
        if size >= 1024 * 1024
        else f"{max(1, round(size / 1024))} KB"
    )


def focus_source(document_id, destination):
    if destination == "Chat":
        st.session_state["answer_source"] = document_id
    else:
        st.session_state["inspect_source"] = document_id
    st.session_state["workspace_tabs"] = destination


def render_add_sources(library, has_sources):
    with st.expander("Add sources", icon=":material/add_circle:", expanded=not has_sources):
        pdf_tab, url_tab = st.tabs(["PDF upload", "Webpage"])
        with pdf_tab:
            uploads = st.file_uploader(
                "Choose PDFs",
                type=["pdf"],
                accept_multiple_files=True,
                help="Text-based PDFs, up to 20 MB and 300 pages each. Scans need OCR.",
            )
            st.caption("Text-based PDFs · Up to 20 MB each")
            if st.button(
                "Add PDFs to library", disabled=not uploads, type="primary", width="stretch"
            ):
                for uploaded in uploads:
                    try:
                        with st.spinner(f"Adding {uploaded.name}…"):
                            manifest, added = library.index_pdf(uploaded.getvalue(), uploaded.name)
                        st.success(
                            f"Added {manifest['filename']}"
                            if added
                            else f"Already in library: {manifest['filename']}"
                        )
                        if manifest["empty_pages"]:
                            st.warning(
                                f"Pages without extractable text: {manifest['empty_pages']}."
                            )
                    except (PDFError, ValueError) as exc:
                        st.error(str(exc))
                    except Exception as exc:  # noqa: BLE001 - keep source content out of logs
                        logger.error("PDF indexing failed (%s)", type(exc).__name__)
                        st.error("Could not add this PDF. Please try again.")
        with url_tab:
            with st.form("url_ingestion"):
                url = st.text_input("Public webpage URL", placeholder="https://example.com/article")
                import_url = st.form_submit_button("Add webpage", width="stretch", type="primary")
            st.caption("One public page at a time. No login or JavaScript rendering.")
            if import_url:
                try:
                    with st.spinner("Reading webpage…"):
                        manifest, added = library.index_url(url)
                    st.success(
                        "Webpage added to your library."
                        if added
                        else "This snapshot is already in your library."
                    )
                except (WebError, ValueError) as exc:
                    st.error(str(exc))
                except Exception as exc:  # noqa: BLE001 - do not log fetched page data
                    logger.error("Webpage indexing failed (%s)", type(exc).__name__)
                    st.error("Could not add this webpage. Please try again.")


def render_source_card(library, source):
    document_id = source["document_id"]
    is_web = source.get("source_type") == "url"
    with st.expander(
        escape_markdown(source["filename"]),
        icon=":material/language:" if is_web else ":material/description:",
    ):
        if is_web:
            st.caption(f"WEBPAGE · {urlsplit(source.get('url', '')).hostname or 'Saved snapshot'}")
            if source.get("fetched_at"):
                try:
                    stamp = datetime.fromisoformat(source["fetched_at"]).strftime(
                        "%d %b %Y, %H:%M UTC"
                    )
                    st.caption(f"Saved {stamp}")
                except ValueError:
                    pass
        else:
            pages = source["total_pages"]
            st.caption(
                f"PDF · {pages} {'page' if pages == 1 else 'pages'} · {readable_size(source.get('size_bytes', 0))}"
            )
        count = source["chunk_count"]
        st.caption(f"{count} searchable {'passage' if count == 1 else 'passages'}")
        if st.session_state.get("answer_source") == document_id:
            st.caption(":material/check_circle: Selected for chat")
        st.button(
            "Chat with this source",
            key=f"chat-source-{document_id}",
            icon=":material/chat_bubble_outline:",
            width="stretch",
            on_click=focus_source,
            args=(document_id, "Chat"),
        )
        st.button(
            "Inspect source",
            key=f"inspect-source-{document_id}",
            icon=":material/visibility:",
            width="stretch",
            on_click=focus_source,
            args=(document_id, "Inspect documents"),
        )
        if st.session_state.get("pending_source_delete") == document_id:
            with st.container(border=True):
                st.warning("Remove this source from your library?")
                st.markdown(f"**{escape_markdown(source['filename'])}**")
                st.caption(
                    "Removes the saved copy and search index. The original file stays untouched. Chat history clears when the library changes."
                )
                if st.button(
                    "Confirm removal",
                    type="primary",
                    key=f"confirm-remove-{document_id}",
                    width="stretch",
                ):
                    try:
                        library.delete_source(document_id)
                        st.session_state.pop("pending_source_delete", None)
                        st.session_state["source_notice"] = f"Removed {source['filename']}"
                        st.rerun()
                    except Exception as exc:  # noqa: BLE001 - do not expose source/provider details
                        logger.error("Source deletion failed (%s)", type(exc).__name__)
                        st.error("Could not remove this source. Please refresh and try again.")
                if st.button("Keep source", key=f"cancel-remove-{document_id}", width="stretch"):
                    st.session_state.pop("pending_source_delete", None)
                    st.rerun()
        elif st.button(
            "Remove source",
            key=f"remove-source-{document_id}",
            icon=":material/delete_outline:",
            type="tertiary",
        ):
            st.session_state["pending_source_delete"] = document_id
            st.rerun()


def render_sidebar(library):
    with st.sidebar:
        sources = library.list_sources()
        st.markdown("## Your library")
        st.caption("Your documents, ready for conversation.")
        summary = st.container(key="library_summary")
        render_add_sources(library, bool(sources))
        # Ingestion may have committed new sources during this run.
        sources = library.list_sources()
        with summary:
            stats = st.columns(3)
            stats[0].metric("Sources", len(sources))
            stats[1].metric(
                "PDFs", sum(source.get("source_type", "pdf") == "pdf" for source in sources)
            )
            stats[2].metric(
                "Webpages", sum(source.get("source_type") == "url" for source in sources)
            )
        sync_library_state(source["document_id"] for source in sources)
        if st.session_state.get("inspect_source") not in {s["document_id"] for s in sources}:
            st.session_state.pop("inspect_source", None)
        notice = st.session_state.pop("source_notice", None)
        if notice:
            st.success(notice)
        st.divider()
        st.markdown("### Manage sources")
        if not sources:
            st.info("Your library is empty. Add a PDF or webpage above to get started.")
        else:
            query = st.text_input(
                "Find a source",
                placeholder="Search by name or website…",
                key="source_filter",
                label_visibility="collapsed",
            )
            kind = st.pills(
                "Source type",
                ["All", "PDFs", "Webpages"],
                default="All",
                required=True,
                key="source_kind",
                label_visibility="collapsed",
            )
            filtered = matching_sources(sources, query, kind)
            st.caption(f"{len(filtered)} of {len(sources)} sources · A–Z")
            if st.session_state.get("pending_source_delete") not in {
                s["document_id"] for s in filtered
            }:
                st.session_state.pop("pending_source_delete", None)
            if not filtered:
                st.info("No matching sources. Try another name or source type.")
                st.button("Reset filters", width="stretch", on_click=clear_filters)
            with st.container(
                height=440 if len(filtered) > 5 else "content", border=False, key="source_list"
            ):
                for source in filtered:
                    render_source_card(library, source)
        st.divider()
        st.caption(
            ":material/lock_outline: Files and search index stay on this device. Questions and retrieved passages are sent to Groq when you chat."
        )
    return sources


def clear_filters():
    st.session_state["source_filter"] = ""
    st.session_state["source_kind"] = "All"
