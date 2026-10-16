"""Small in-memory PDFs and deterministic embeddings for offline correctness tests."""

from io import BytesIO

from langchain_core.embeddings import Embeddings
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast


def make_pdf(texts: list[str], *, encrypted=False) -> bytes:
    writer = PdfWriter()
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    for text in texts:
        page = writer.add_blank_page(width=612, height=792)
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})}
        )
        if text:
            escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            stream = DecodedStreamObject()
            stream.set_data(f"BT /F1 12 Tf 50 740 Td ({escaped}) Tj ET".encode("ascii"))
            page[NameObject("/Contents")] = writer._add_object(stream)
    if encrypted:
        writer.encrypt("test-password")
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


class TestEmbeddings(Embeddings):
    __test__ = False

    def __init__(self):
        vocabulary = ["[UNK]", "refund", "returns", "shipping", "delivery", "garden", "plants"]
        tokenizer = Tokenizer(
            WordPiece({word: i for i, word in enumerate(vocabulary)}, unk_token="[UNK]")
        )
        tokenizer.pre_tokenizer = Whitespace()
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]")
        self.max_tokens = 256

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        text = text.lower()
        return [
            float("refund" in text or "returns" in text),
            float("shipping" in text or "delivery" in text),
            float("garden" in text or "plants" in text),
            0.01,
        ]
