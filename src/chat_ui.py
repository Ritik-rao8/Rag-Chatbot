"""A conversation-first UI with citations and validated answers."""

import logging
import re
from uuid import uuid4

import streamlit as st

from src.chat import (
    AnswerError,
    AnswerResult,
    answer_question,
    conversation_context,
    groq_configuration,
)
from src.source_ui import location, source_access

logger = logging.getLogger(__name__)


def escape_markdown(text: str) -> str:
    return re.sub(r"([\\`*_{}\[\]()<>#+.!|~$-])", r"\\\1", text)


def queue_retry(turn):
    st.session_state["retry_turn_id"] = turn["id"]
    st.session_state["answer_source"] = turn["scope_id"]


def render_turn(library, turn):
    with st.chat_message("user"):
        st.markdown(escape_markdown(turn["question"]))
    with st.chat_message("assistant"):
        result = turn["result"]
        if result.status != "answered":
            st.warning(result.message) if result.status in {
                "error",
                "validation_failed",
            } else st.info(result.message)
        else:
            for statement in result.draft.statements:
                ids = list(dict.fromkeys(e.source_id for e in statement.evidence))
                links = " ".join(
                    f"[{source_id}](#answer-{turn['id']}-source-{source_id})" for source_id in ids
                )
                st.markdown(f"{escape_markdown(statement.text)} {links}")
            used = {}
            for statement in result.draft.statements:
                for evidence in statement.evidence:
                    used.setdefault(evidence.source_id, []).append(evidence.quote)
            st.caption("Sources")
            for passage in result.passages:
                if passage.source_id not in used:
                    continue
                meta = passage.document.metadata
                st.markdown(
                    f'<div id="answer-{turn["id"]}-source-{passage.source_id}"></div>',
                    unsafe_allow_html=True,
                )
                label = f"[{passage.source_id}] {meta['source']} · {location(meta)}"
                with st.expander(label):
                    for quote in dict.fromkeys(used[passage.source_id]):
                        st.text(quote)
                    if meta.get("source_type") != "url" and meta.get("page_label"):
                        st.caption(f"Printed page label: {meta['page_label']}")
                    source_access(library, meta, key=f"download-{turn['id']}-{passage.source_id}")
        elapsed = getattr(result, "elapsed_seconds", 0)
        st.caption(f"{turn['scope']}" + (f" · {elapsed:.1f}s" if elapsed else ""))
        if result.status in {"error", "validation_failed"}:
            st.button("Retry answer", key=f"retry-{turn['id']}", on_click=queue_retry, args=(turn,))
        if result.passages:
            with st.expander("Retrieval details"):
                st.text(f"Resolved question: {result.resolved_question or turn['question']}")
                st.caption(
                    "Combined semantic and keyword retrieval. Cosine distance is not confidence; ranking also uses keyword matches."
                )
                for passage in result.passages:
                    meta = passage.document.metadata
                    st.text(
                        f"[{passage.source_id}] {meta['source']} · {location(meta)} · Distance {passage.distance:.4f}"
                    )
                    st.text(passage.document.page_content)
                st.caption(
                    f"Pipeline calls (excluding SDK retries/fallbacks): {getattr(result, 'api_calls', 0)}"
                )


def export_chat(turns):
    lines = ["Document Explorer conversation", ""]
    for turn in turns:
        result = turn["result"]
        lines.extend([f"Question: {turn['question']}", f"Scope: {turn['scope']}"])
        if result.status == "answered":
            for statement in result.draft.statements:
                ids = dict.fromkeys(e.source_id for e in statement.evidence)
                lines.append(statement.text + " " + " ".join(f"[{source_id}]" for source_id in ids))
            used = {
                e.source_id for statement in result.draft.statements for e in statement.evidence
            }
            for passage in result.passages:
                if passage.source_id in used:
                    meta = passage.document.metadata
                    lines.append(
                        f"Source [{passage.source_id}]: {meta['source']} · {location(meta)}"
                        + (f" · {meta['url']}" if meta.get("url") else "")
                    )
        else:
            lines.append(result.message)
        lines.append("")
    return "\n".join(lines)


def render_answers(library, by_id: dict):
    key, model_name = groq_configuration()
    controls = st.columns([3, 1, 1], vertical_alignment="bottom")
    with controls[0]:
        selected = st.selectbox(
            "Answer using",
            ["all", *by_id],
            key="answer_source",
            format_func=lambda item: "All documents" if item == "all" else by_id[item]["filename"],
        )
    with controls[1]:
        if st.button(
            "New chat", use_container_width=True, disabled=not st.session_state.get("answer_turns")
        ):
            st.session_state["answer_turns"] = []
            st.session_state.pop("retry_turn_id", None)
    with controls[2]:
        st.download_button(
            "Export chat",
            export_chat(st.session_state.get("answer_turns", [])),
            file_name="document-chat.txt",
            mime="text/plain",
            use_container_width=True,
            disabled=not st.session_state.get("answer_turns"),
        )
    st.caption(f"{model_name} · Answers grounded in your sources")
    turns = st.session_state.get("answer_turns", [])
    retry_id = st.session_state.pop("retry_turn_id", None)
    retry = next((turn for turn in turns if turn["id"] == retry_id), None)
    transcript = st.container()
    with transcript:
        if not turns:
            st.markdown("### What would you like to understand?")
            st.write(
                "Ask a question about your documents. Each answer links to the passages that support it."
            )
            if by_id:
                st.caption(
                    "Try a definition, compare two topics, or ask for a policy and its exceptions."
                )
        for turn in turns:
            if turn is not retry:
                render_turn(library, turn)
    if not key:
        st.info("Add GROQ_API_KEY to your local .env file to enable chat.")
    if not by_id:
        st.info("Add a PDF or webpage from the sidebar to start chatting.")
    question = st.chat_input(
        "Ask your documents…", key="chat_prompt", max_chars=1200, disabled=not by_id or not key
    )
    st.caption(
        "Questions and retrieved passages go to Groq. The latest 10 turns stay in this session. Review source quotes for important answers."
    )
    if retry:
        question = retry["question"]
    if not question:
        return
    with transcript:
        with st.chat_message("user"):
            st.markdown(escape_markdown(question))
        with (
            st.chat_message("assistant"),
            st.status(
                "Finding supporting passages and checking the answer…", expanded=False
            ) as status,
        ):
            try:
                history_turns = turns[: turns.index(retry)] if retry else turns
                result = answer_question(
                    library,
                    question,
                    document_id=None if selected == "all" else selected,
                    history=conversation_context(history_turns, selected),
                )
                status.update(
                    label="Answer checked" if result.status == "answered" else "Request complete",
                    state="complete",
                )
            except (AnswerError, ValueError) as exc:
                result = AnswerResult("error", message=str(exc), model=model_name)
                status.update(label="Could not complete the answer", state="error")
            except Exception as exc:  # noqa: BLE001 - keep prompts and credentials out of logs
                logger.error("Answer generation failed (%s)", type(exc).__name__)
                result = AnswerResult(
                    "error",
                    message="Something went wrong while generating the answer. Please retry.",
                    model=model_name,
                )
                status.update(label="Could not complete the answer", state="error")
    new_turn = {
        "id": retry["id"] if retry else uuid4().hex,
        "question": question,
        "scope": "All documents" if selected == "all" else by_id[selected]["filename"],
        "scope_id": selected,
        "result": result,
    }
    if retry:
        turns[turns.index(retry)] = new_turn
    else:
        turns.append(new_turn)
    st.session_state["answer_turns"] = turns[-10:]
    st.rerun()
