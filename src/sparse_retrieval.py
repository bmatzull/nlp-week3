from rank_bm25 import BM25Okapi

class SparseRetriever:
    def __init__(self):
        """Initializes BM25 Sparse Retriever"""
        self.corpus_chunks = []
        self.tokenized_corpus = []
        self.bm25 = None

    def add_chunks(self, chunks: list[str]):
        """Saves the chunks and builds the BM25 keyword index"""
        self.corpus_chunks.extend(chunks)

        self.tokenized_corpus = [chunk.lower().split(" ") for chunk in self.corpus_chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def query(self, query_text:str, k: int = 3) -> list[str]:
        """Retrieves the top-k chunks based on exact keyword overlap"""
        if k == 0 or self.bm25 is None:
            return []

        tokenized_query = query_text.lower().split(" ")
        results = self.bm25.get_top_n(tokenized_query, self.corpus_chunks, k)
        return results