import chromadb
from chromadb.utils import embedding_functions

class VectorDatabase:
    def __init__(self, collection_name: str, model_name: str = "all-MiniLM-L6-v2"):
        """Initializes the ChromaDB client and sets the embedding model."""

        self.client = chromadb.PersistentClient(path="data/chromadb")
        self.embeddin_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embeddin_fn
        )

    def add_chunks(self, chunks: list[str], source_name:str ):
        """Converts chunks to embeddings and saves them to the database"""
        ids = [f"{source_name}_chunk{i}" for i in range(len(chunks))]
        metadatas = [{"source": source_name, "chunk_index": i} for i in range(len(chunks))]

        self.collection.add(
            documents=chunks,
            metadatas=metadatas,
            ids=ids
        )

    def query(self, query_text:str, k:int = 3) -> list[str]:
        """Retrieves top-k most similar chunks for a given question"""
        if k == 0:
            return []

        results = self.collection.query(
            query_texts=[query_text],
            n_results=k,
        )

        return results["documents"][0]