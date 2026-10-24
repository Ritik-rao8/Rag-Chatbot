"""Save, reopen, rename, and remove local named conversations."""

import streamlit as st

from src.conversations import ConversationError, ConversationStore


@st.cache_resource
def get_conversation_store(data_dir):
    return ConversationStore(data_dir)


def detach_conversation():
    for key in (
        "active_conversation_id",
        "active_conversation_title",
        "conversation_name",
        "conversation_dirty",
    ):
        st.session_state.pop(key, None)


def new_chat():
    st.session_state["answer_turns"] = []
    for key in ("retry_turn_id", "pdf_preview_request", "pdf_preview_active"):
        st.session_state.pop(key, None)
    detach_conversation()


def open_conversation(store, conversation_id, source_ids):
    conversation = store.load(conversation_id)
    st.session_state["answer_turns"] = conversation["turns"]
    st.session_state["active_conversation_id"] = conversation["id"]
    st.session_state["active_conversation_title"] = conversation["title"]
    st.session_state["conversation_name"] = conversation["title"]
    st.session_state["conversation_dirty"] = False
    st.session_state["answer_source"] = (
        conversation["scope_id"] if conversation["scope_id"] in source_ids else "all"
    )
    st.session_state["workspace_tabs"] = "Chat"
    st.session_state.pop("retry_turn_id", None)
    st.session_state.pop("pdf_preview_request", None)
    st.session_state.pop("pdf_preview_active", None)


def render_saved_conversations(library, sources):
    with st.expander("Saved conversations", icon=":material/forum:"):
        try:
            store = get_conversation_store(library.settings.data_dir)
            chats = store.list()
        except ConversationError as exc:
            st.warning(str(exc))
            return
        if not chats:
            st.caption("Save a chat with a name to find it here after restarting the app.")
            return
        by_id = {chat["id"]: chat for chat in chats}
        chosen = st.selectbox(
            "Saved conversation",
            list(by_id),
            format_func=lambda item: by_id[item]["title"],
            key="saved_chat_selection",
        )
        chat = by_id[chosen]
        if st.session_state.get("pending_chat_delete") != chosen:
            st.session_state.pop("pending_chat_delete", None)
        count = chat["turn_count"]
        st.caption(
            f"{count} {'turn' if count == 1 else 'turns'} · Updated {chat['updated_at'][:10]}"
        )
        if (
            st.session_state.get("answer_turns")
            and st.session_state.get("active_conversation_id") != chosen
        ):
            st.caption("Opening replaces the current chat. Save it first if you want to keep it.")
        if st.button("Open conversation", key="open_saved_chat", width="stretch"):
            try:
                open_conversation(store, chosen, {s["document_id"] for s in sources})
                st.rerun()
            except ConversationError as exc:
                st.error(str(exc))
        with st.form(f"rename-conversation-{chosen}"):
            name = st.text_input("Rename conversation", value=chat["title"], max_chars=80)
            if st.form_submit_button("Rename", width="stretch"):
                try:
                    store.rename(chosen, name)
                    if st.session_state.get("active_conversation_id") == chosen:
                        st.session_state["active_conversation_title"] = name.strip()
                        st.session_state["conversation_name"] = name.strip()
                    st.rerun()
                except ConversationError as exc:
                    st.error(str(exc))
        if st.session_state.get("pending_chat_delete") == chosen:
            st.warning("Remove this saved conversation? Your source documents will remain.")
            if st.button("Confirm chat removal", width="stretch"):
                try:
                    store.delete(chosen)
                    if st.session_state.get("active_conversation_id") == chosen:
                        new_chat()
                    st.session_state.pop("pending_chat_delete", None)
                    st.rerun()
                except ConversationError as exc:
                    st.error(str(exc))
            if st.button("Keep conversation", width="stretch"):
                st.session_state.pop("pending_chat_delete", None)
                st.rerun()
        elif st.button("Remove saved conversation", type="tertiary"):
            st.session_state["pending_chat_delete"] = chosen
            st.rerun()


def save_current(library, title, scope_id):
    store = get_conversation_store(library.settings.data_dir)
    conversation_id = store.save(
        title,
        st.session_state.get("answer_turns", []),
        scope_id=scope_id,
        library_signature=[s["document_id"] for s in library.list_sources()],
        profile=library.collection_name,
        conversation_id=st.session_state.get("active_conversation_id"),
    )
    st.session_state["active_conversation_id"] = conversation_id
    st.session_state["active_conversation_title"] = title.strip()
    st.session_state["conversation_dirty"] = False


def render_save_chat(library, selected):
    with st.popover(
        "Save chat",
        icon=":material/bookmark_add:",
        disabled=not st.session_state.get("answer_turns"),
    ):
        turns = st.session_state.get("answer_turns", [])
        if not turns:
            st.caption("Send a message before saving a conversation.")
            return
        st.session_state.setdefault(
            "conversation_name",
            st.session_state.get("active_conversation_title")
            or (turns[0]["question"][:60] if turns else ""),
        )
        with st.form("save_named_conversation"):
            title = st.text_input("Conversation name", max_chars=80, key="conversation_name")
            st.caption(
                "Saved locally. New replies are saved automatically after you name the chat."
            )
            if st.form_submit_button("Save conversation", type="primary"):
                try:
                    save_current(library, title, selected)
                    st.session_state["conversation_notice"] = "Conversation saved."
                    st.rerun()
                except ConversationError as exc:
                    st.error(str(exc))


def autosave(library, selected):
    if not st.session_state.get("active_conversation_id"):
        return
    try:
        save_current(library, st.session_state["active_conversation_title"], selected)
    except ConversationError as exc:
        st.session_state["conversation_dirty"] = True
        st.session_state["conversation_notice"] = str(exc)
