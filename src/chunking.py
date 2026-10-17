"""Split within each page, using the embedding model's actual tokenizer."""

from collections import defaultdict

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_pages(
    pages: list[Document], tokenizer, *, chunk_tokens: int = 220, overlap_tokens: int = 32
) -> list[Document]:
    if not 0 <= overlap_tokens < chunk_tokens:
        raise ValueError("Chunk overlap must be smaller than the chunk size.")
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=chunk_tokens,
        chunk_overlap=overlap_tokens,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents([page for page in pages if page.page_content.strip()])
    page_counts = defaultdict(int)
    for chunk in chunks:
        key = (chunk.metadata["document_id"], chunk.metadata["page_number"])
        page_counts[key] += 1
        chunk.metadata["chunk_number"] = page_counts[key]
        chunk.metadata["chunk_id"] = f"{key[0]}:p{key[1]}:c{page_counts[key]}"
        chunk.metadata["token_count"] = len(
            tokenizer.encode(chunk.page_content, add_special_tokens=False)
        )
    return chunks
