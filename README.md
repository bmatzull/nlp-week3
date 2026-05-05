# NLP RAG Pipeline

## 1. Project Overview
This project builds a Retrieval-Augmented Generation (RAG) pipeline to evaluate how much grounding a language model in external context actually helps with factual question answering. The pipeline tests this across different datasets, retrieval methods, generators, and how many documents you retrieve (K).

## 2. Environment Setup
This project uses `uv` for environment management. Library code lives in `src/`, the actual experiment script is in `experiments/`.

1. Clone the repository
2. Activate the virtual environment:
   - Linux/Mac: `source .venv/bin/activate`
   - Windows: `.venv\Scripts\activate`
3. Install dependencies:
   `uv pip install -r requirements.txt`
4. Run the RAG pipeline:
   `python experiments/run_retrieval.py`

## 3. Pipeline Architecture

To reduce hallucinations from pure generation, the pipeline retrieves relevant document chunks before prompting the generator. Architecture:

**Chunking (`src/chunking.py`)** — context documents are split into overlapping chunks of 300 characters with 30-character overlap using `RecursiveCharacterTextSplitter`. Overlap ensures nothing gets cut off at a bad boundary.

**Indexing** — chunks are indexed in two ways:
- **Dense (`src/vector_db.py`)** — chunks are embedded using a sentence transformer and stored in ChromaDB. At query time the question is also embedded and the nearest vectors are retrieved.
- **Sparse (`src/sparse_retrieval.py`)** — BM25, a classical keyword frequency method. No embeddings, just term matching.

**Generation (`src/generation.py`)** — retrieved chunks are concatenated into a context block and fed to the generator alongside the question. K=0 means no retrieval at all and the model answers purely from memory.

## 4. Evaluation Matrix

The pipeline was evaluated across 1,200 combinations:

- **Datasets:** TriviaQA, Natural Questions
- **Retrievers:** Dense (MiniLM, MPNet), Sparse (BM25)
- **Generators:** flan-t5-small, flan-t5-base
- **K values:** 0, 1, 5, 10

Full results are in `results/results.csv`.

### Results Snapshot

| Dataset | Question | True Answer | Retriever | Generator | K | Predicted | Outcome |
|---|---|---|---|---|---|---|---|
| TriviaQA | Who was the man behind The Chipmunks? | david seville | MiniLM | T5-Small | 0 | john scott | Hallucination |
| TriviaQA | Who was the man behind The Chipmunks? | david seville | MiniLM | T5-Small | 5 | Ross Bagdasarian | Correct |
| Natural_Q | who plays sonny's father in general hospital | Ron Hale | BM25 | T5-Base | 0 | John Wayne | Hallucination |
| Natural_Q | who plays sonny's father in general hospital | Ron Hale | BM25 | T5-Base | 5 | Ron Hale | Correct |
| TriviaQA | If I Were A Rich Man was a hit from which stage show? | fiddler on roof | MPNet | T5-Base | 5 | Oliver! | Retrieval Failure |

## 5. Failure Case Analysis

Going through the results, a few clear patterns came up. The most surprising one was that K=10 sometimes performed worse than K=1, meaning more context isn't always better.

**1. K=0 Hallucinations (Generator)**
- Question: "Who was the man behind The Chipmunks?" — True: David Seville
- Predicted: `john mccartney` (T5-Base, K=0)
- With no context, the model just guesses from memory. It produced a plausible-sounding but wrong name and was probably mixing up famous musicians from that time. 

**2. Too Much Context Hurts (Generator)**
- Question: "Who had a 70s No 1 hit with Kiss You All Over?" — True: Exile
- Predicted: `Michael Jackson` (MPNet / T5-Base, K=10)
- At K=1 the model got this right. At K=10 it was flooded with extra music-related chunks and defaulted to the most famous name in the context.
  
**3. Repetition Loop (Generator)**
- Question: "Where does Alaska the Last Frontier take place?"
- Predicted: `southeastern southeastern southeastern...` (T5-Small, K=0)
- Without any context T5-Small got stuck repeating the same token until it hit the generation limit. Seems to be a known issue with small encoder-decoder models when they have nothing to ground on.

**4. BM25 Keyword Mismatch (Retriever)**
- Question: "Who was the man behind The Chipmunks?" — True: David Seville
- Predicted: `Wayne B. Wheeler` (BM25 / T5-Small, K=5)
- BM25 matched on the phrase "man behind" and retrieved a chunk about Wayne B. Wheeler, "the man behind Prohibition." Semantically completely wrong, but keyword-wise it made sense. Dense retrievers handled this one much better.

**5. Wrong Phrase Extracted (Generator)**
- Question: "Rita Coolidge sang the title song for which Bond film?" — True: Octopussy
- Predicted: `year hiatus` (MiniLM / T5-Base, K=10)
- The retrieved chunk mentioned "After a year hiatus, the Bond series rebooted..." The model picked up a noun phrase from the context instead of the actual movie title.

**6. Context Bleed Between Questions (Retriever)**
- Question: "If I Were A Rich Man was a hit from which stage show?" — True: Fiddler on the Roof
- Predicted: `Octopussy` (BM25 / T5-Small, K=5)
- Because all contexts were indexed together in one corpus, BM25 accidentally pulled a chunk from the Bond question above due to keyword overlap. The model had no way to know the context was irrelevant.

**7. Answer Is There, Model Misses It (Generator)**
- Question: "who sang what in the world's come over you" — True: Jack Scott
- Predicted: `American` (MiniLM / T5-Small, K=10)
- The retrieved text literally started with "Jack Scott (singer)". The model skipped past the name and pulled an adjective from deeper in the chunk instead, maybe the phrasing of the question threw it off.

**8. Empty Output (Generator)**
- Question: "when did 17 nam summit meet take place"
- Predicted: *(empty)* (MiniLM / T5-Base, K=1)
- The context was dense with dates and complex clauses. The model seemed to just give up and predict end-of-sequence immediately. The context looked relevant, so it's probably a generation issue rather than retrieval.

**9. Close But Wrong Date (Retriever)**
- Question: "In which decade did stereo records first go on sale?" — True: 1930s
- Predicted: `1958` (BM25 / T5-Small, K=1)
- BM25 retrieved a chunk about the commercial popularity of LPs in 1958. The model extracted that year, which is historically related but not the right answer.

**10. Pre-training Bias Overrides Context (Generator)**
- Question: "Which 90s sitcom featured the character Cosmo Kramer?" — True: Seinfeld
- Predicted: `Friends` (T5-Base, K=0)
- At K=0 the model defaulted to the most famous 90s sitcom from its pre-training instead of the correct one. Even at K=1 it sometimes still said Friends, suggesting retrieval alone doesn't always override strong pre-training biases.
