"""RAG retrieval pipeline combining embeddings + vector store."""

from dataclasses import dataclass

from rag_slm_system.embeddings.base import BaseEmbedder
from rag_slm_system.vectorstore.base import BaseVectorStore, SearchResult


@dataclass
class RetrievalResult:
    """Result from RAG retrieval with formatted context."""

    results: list[SearchResult]
    context: str
    query: str

    @property
    def chunks(self):
        return [r.chunk for r in self.results]

    @property
    def texts(self):
        return [r.chunk.text for r in self.results]


class RAGRetriever:
    """RAG retrieval: embed query -> search vector store -> return ranked context."""

    def __init__(
        self,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.top_k = top_k
        self.score_threshold = score_threshold

    def retrieve(self, query: str, top_k: int | None = None) -> RetrievalResult:
        """Retrieve relevant chunks for a query.

        Args:
            query: The search query.
            top_k: Override default top_k for this query.

        Returns:
            RetrievalResult with ranked chunks and formatted context.
        """
        k = top_k or self.top_k
        query_embedding = self.embedder.embed_query(query)
        results = self.vector_store.search(query_embedding, top_k=k)

        if self.score_threshold > 0:
            results = [r for r in results if r.score >= self.score_threshold]

        context = self._format_context(results)

        return RetrievalResult(
            results=results,
            context=context,
            query=query,
        )

    def _format_context(self, results: list[SearchResult]) -> str:
        """Format search results into a context string for LLM consumption."""
        if not results:
            return ""

        parts: list[str] = []
        for i, result in enumerate(results, 1):
            source_info = f" (source: {result.chunk.source})" if result.chunk.source else ""
            parts.append(
                f"[Document {i}{source_info}]\n{result.chunk.text}"
            )

        return "\n\n---\n\n".join(parts)

    def retrieve_with_answer(
        self,
        query: str,
        answer_fn: callable,
        top_k: int | None = None,
    ) -> dict:
        """Retrieve context and generate an answer using the provided function.

        Args:
            query: The question.
            answer_fn: Function that takes (query, context) -> answer string.
            top_k: Override default top_k.

        Returns:
            Dict with 'query', 'context', 'answer', and 'sources'.
        """
        retrieval = self.retrieve(query, top_k)

        answer = answer_fn(query, retrieval.context)

        return {
            "query": query,
            "answer": answer,
            "context": retrieval.context,
            "sources": [
                {
                    "text": r.chunk.text[:200],
                    "source": r.chunk.source,
                    "score": r.score,
                }
                for r in retrieval.results
            ],
        }
