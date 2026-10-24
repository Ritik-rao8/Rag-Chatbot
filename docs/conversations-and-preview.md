# PDF preview and saved conversations

## Preview a source

Expand a PDF citation and choose **Preview PDF page N**. The preview opens on that
one-based PDF page, with Previous/Next controls, a page selector, the cited excerpt,
and the original PDF download. **Manage sources → Preview PDF** opens page one.
**Inspect documents** also shows the selected page and its extracted text.

Pages are rendered locally with PDFium; no external PDF viewer receives your file.
Only the selected page is rendered, with bounded image dimensions and a small temporary
cache. Previewing does not call Groq. This is a visual page preview, without selectable
PDF text or automatic highlighting. Printed page labels may differ from PDF positions.

## Save and resume a chat

1. Send a question, then choose **Save chat**, enter a name, and save.
2. Further replies automatically update that named conversation.
3. Open **Saved conversations** in the sidebar to reopen, rename, or remove a chat.
4. **New chat** starts an unnamed conversation; existing saved chats stay on disk.

Each conversation holds up to 50 question/answer turns, source scope, retrieved excerpts,
validated citations, and model/timing metadata. The database is
`RAG_DATA_DIR/conversations.sqlite3` (normally `data/conversations.sqlite3`) and is ignored
by the repository rules. API credentials and model configuration secrets are not copied
into the conversation snapshot. The current local app shares one library and saved-chat
store among its sessions.

Removing a source does not remove quoted excerpts already saved in conversations. Those
answers are marked as archived and their original file preview becomes unavailable.
New questions retrieve from the current library; old-library turns are excluded from
follow-up context. To remove archived chat text, remove its saved conversation separately.
Removing a conversation does not remove source documents.

Opening a chat does not regenerate its answers or call Groq. Save failures leave the
current chat open and display an unsaved-changes notice. Duplicate names are rejected
rather than silently overwriting a different conversation.

## Verification

The tests cover PDF page rendering/navigation, renderer cache invalidation, saved
citation round trips, fresh-session reopening, automatic updates, duplicate names,
renaming/removal, corrupted snapshots, and source removal/context boundaries.

```bash
python -m pytest tests/test_conversations.py tests/test_pdf_preview.py tests/test_saved_chat_ui.py -q
```
