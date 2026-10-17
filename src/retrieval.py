"""Small local BM25 index and reciprocal-rank fusion, without another model download."""

import math
import re
from collections import Counter

STOP_WORDS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "what",
    "which",
    "who",
    "when",
    "where",
    "why",
    "how",
    "do",
    "does",
    "did",
    "can",
    "could",
    "would",
    "should",
    "i",
    "you",
    "we",
    "it",
    "its",
    "my",
    "your",
    "of",
    "to",
    "in",
    "on",
    "for",
    "with",
    "and",
    "or",
    "from",
    "this",
    "that",
}


def terms(text):
    return [
        word
        for word in re.findall(r"[^\W_]+(?:[-.][^\W_]+)*", text.casefold())
        if word not in STOP_WORDS
    ]


class KeywordIndex:
    def __init__(self, documents):
        self.documents = documents
        self.counts = [Counter(terms(doc.page_content)) for doc in documents]
        self.lengths = [sum(count.values()) for count in self.counts]
        self.average = sum(self.lengths) / max(1, len(documents)) or 1
        self.frequency = Counter(word for count in self.counts for word in count)

    def search(self, query, allowed, k=12):
        query_terms = set(terms(query))
        scored = []
        for doc, count, length in zip(self.documents, self.counts, self.lengths):
            if doc.metadata["document_id"] not in allowed:
                continue
            score = 0.0
            for word in query_terms & count.keys():
                frequency = self.frequency[word]
                inverse = math.log(1 + (len(self.documents) - frequency + 0.5) / (frequency + 0.5))
                score += (
                    inverse
                    * count[word]
                    * 2.5
                    / (count[word] + 1.5 * (0.25 + 0.75 * length / self.average))
                )
            if score > 0:
                scored.append((score, doc))
        return [
            doc
            for _, doc in sorted(scored, key=lambda pair: (-pair[0], pair[1].metadata["chunk_id"]))[
                :k
            ]
        ]


def fuse(semantic, lexical, k):
    scores, docs = {}, {}
    for ranking in (semantic, lexical):
        for rank, doc in enumerate(ranking, 1):
            key = doc.metadata["chunk_id"]
            docs[key] = doc
            scores[key] = scores.get(key, 0) + 1 / (60 + rank)
    ranked = sorted(scores, key=lambda key: -scores[key])
    selected, seen = [], set()
    for key in ranked:
        doc = docs[key]
        # Repeated headers or overlapping identical chunks should not consume context.
        identity = (doc.metadata["document_id"], " ".join(doc.page_content.split()))
        if identity not in seen:
            selected.append(doc)
            seen.add(identity)
        if len(selected) == k:
            break
    return selected
