import math
import re
import unicodedata
from collections import Counter, defaultdict

from finrag.config import FinragError
from finrag.domain.chunks import ChunkRecord
from finrag.domain.retrieval import LexicalConfig, RetrievalHit

TOKEN_PATTERN = re.compile(r"[^\W_]+(?:[.,][0-9]+)*", re.UNICODE)


def tokenize(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).lower()
    folded = "".join(
        character
        for character in unicodedata.normalize("NFKD", normalized)
        if not unicodedata.combining(character)
    )
    return TOKEN_PATTERN.findall(folded)


class BM25:
    def __init__(self, chunks: list[ChunkRecord], config: LexicalConfig):
        if not chunks:
            raise FinragError("BM25 requires a nonempty chunk corpus.")
        if len({chunk.chunk_id for chunk in chunks}) != len(chunks):
            raise FinragError("BM25 corpus contains duplicate chunk IDs.")
        self.chunks = sorted(chunks, key=lambda chunk: chunk.chunk_id)
        self.config = config
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.lengths = []
        for position, chunk in enumerate(self.chunks):
            terms = tokenize(chunk.text)
            self.lengths.append(len(terms))
            for term, frequency in Counter(terms).items():
                self.postings[term].append((position, frequency))
        self.average_length = sum(self.lengths) / len(self.lengths)
        if self.average_length == 0:
            raise FinragError("BM25 corpus has no lexical tokens.")

    def search(self, query: str, top_k: int = 10) -> list[RetrievalHit]:
        if top_k <= 0:
            raise FinragError("Retrieval top_k must be positive.")
        terms = Counter(tokenize(query))
        if not terms:
            return []
        scores: dict[int, float] = defaultdict(float)
        total = len(self.chunks)
        for term in sorted(terms):
            postings = self.postings.get(term, [])
            if not postings:
                continue
            idf = math.log1p((total - len(postings) + 0.5) / (len(postings) + 0.5))
            for position, frequency in postings:
                length_ratio = self.lengths[position] / self.average_length
                denominator = frequency + self.config.k1 * (
                    1 - self.config.b + self.config.b * length_ratio
                )
                scores[position] += (
                    terms[term] * idf * frequency * (self.config.k1 + 1) / denominator
                )
        # Zero-score chunks have no matching evidence; returning them would create arbitrary hits.
        ranked = sorted(
            scores, key=lambda position: (-scores[position], self.chunks[position].chunk_id)
        )[:top_k]
        return [
            self._hit(position, rank, scores[position])
            for rank, position in enumerate(ranked, start=1)
        ]

    def _hit(self, position: int, rank: int, score: float) -> RetrievalHit:
        chunk = self.chunks[position]
        return RetrievalHit(
            rank=rank,
            lexical_score=score,
            document_id=chunk.document_id,
            source_filename=chunk.source_filename,
            source_sha256=chunk.source_sha256,
            physical_page_number=chunk.physical_page_number,
            chunk_id=chunk.chunk_id,
            text=chunk.text,
            configuration_id=self.config.configuration_id,
        )
