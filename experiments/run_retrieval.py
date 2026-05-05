import sys
import os
import pandas as pd
from datasets import load_dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.vector_db import VectorDatabase
from src.sparse_retrieval import SparseRetriever
from src.generation import Generator
from src.chunking import chunk_by_sentence


def main():
    retrievers = {
        "Dense_1_MiniLM": VectorDatabase(collection_name="dense_minilm", model_name="all-MiniLM-L6-v2"),
        "Dense_2_MPNet": VectorDatabase(collection_name="dense_mpnet", model_name="all-mpnet-base-v2"),
        "Sparse_BM25": SparseRetriever()
    }

    generators = {
        "T5_Small": Generator("google/flan-t5-small"),
        "T5_Base": Generator("google/flan-t5-base")
    }

    k_values = [0, 1, 5, 10]

    datasets_to_run = [
        {
            "name": "TriviaQA",
            "path": "mandarjoshi/trivia_qa",
            "config": "rc",
            "split": "validation[:25]"
        },
        {
            "name": "Natural_Questions",
            "path": "sentence-transformers/natural-questions",
            "config": None,
            "split": "train[:25]"
        }
    ]

    results = []

    for ds_info in datasets_to_run:
        print(f"\n=== Processing Dataset: {ds_info['name']} ===")

        if ds_info["config"]:
            dataset = load_dataset(
                ds_info["path"],
                ds_info["config"],
                split=ds_info["split"],
            )
        else:
            dataset = load_dataset(ds_info["path"], split=ds_info["split"])

        if ds_info["name"] == "TriviaQA":
            questions = dataset["question"]
            answers = [ans["normalized_value"] for ans in dataset["answer"]]
            contexts = []
            for search_res in dataset["search_results"]:
                for doc in search_res.get("search_context", []):
                    if doc and str(doc).strip():
                        contexts.append(str(doc))
        else:
            questions = dataset["query"]
            answers = dataset["answer"]
            contexts = dataset["answer"]

        all_chunks = []
        for text in contexts[:50]:
            all_chunks.extend(chunk_by_sentence(str(text), chunk_size=300, chunk_overlap=30))

        if not all_chunks:
            print(f"Warning: No chunks generated for {ds_info['name']}, skipping.")
            continue

        for ret_name, retriever in retrievers.items():
            if ret_name == "Sparse_BM25":
                retriever.add_chunks(all_chunks)
            else:
                retriever.add_chunks(all_chunks, ds_info['name'])

        for q_idx, question in enumerate(questions):
            true_ans = answers[q_idx]

            for ret_name, retriever in retrievers.items():
                for gen_name, generator in generators.items():
                    for k in k_values:
                        retrieved_chunks = retriever.query(question, k=k) if k > 0 else []
                        predicted_ans = generator.generate(question, retrieved_chunks, k)

                        results.append({
                            "Dataset": ds_info['name'],
                            "Question": question,
                            "True Answer": true_ans,
                            "Retriever": ret_name,
                            "Generator": gen_name,
                            "K": k,
                            "Predicted Answer": predicted_ans
                        })

    df = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/results.csv", index=False)
    print("Done! Results saved.")


if __name__ == "__main__":
    main()