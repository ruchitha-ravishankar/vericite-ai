# VeriCite AI: Research Agent with Citations

A retrieval-augmented research agent that answers technical questions from a document corpus, attaches sentence-level citations to its answers, checks claims against the retrieved evidence, and refuses questions the corpus cannot answer.

Built for the Rooman AI Challenge (Junior AI Research Associate selection round), Agent Track: Research Agent with Citations.

## Problem Statement

Language-model answers to technical questions are often fluent but unsupported. This project explores a research agent whose every claim can be traced to a specific passage, and which declines to answer when the evidence is weak.

## Key Features

- **Hybrid retrieval:** BM25 (Okapi) keyword search combined with dense embeddings using Reciprocal Rank Fusion (k = 60).
- **Document ingestion:** PDF (page-aware), Markdown, TXT and JSON loaders with sliding-window chunking (650 characters, 120 overlap).
- **Citations:** answers carry markers of the form `[^doc#chunk]` linked to the source document, section, page and quote.
- **Refusal guardrail:** if retrieval relevance falls below a threshold (0.180), the agent declines instead of guessing.
- **Claim verification:** generated sentences are checked against retrieved text using N-gram overlap.
- **Query decomposition:** complex questions are split into sub-queries.
- **Interfaces:** a FastAPI web app and a terminal CLI.
- **Pluggable LLM backend:** Gemini, OpenAI, Groq or Ollama through `.env`, or a built-in local mode that needs no API key.

## Technology Stack

Python, FastAPI, Pydantic, rank-bm25, pypdf, pytest, Rich (CLI), vanilla HTML/CSS/JS front end. Optional LLM providers: Gemini, OpenAI, Groq, Ollama.

## System Architecture

```
Question -> Query analyzer (sub-queries)
         -> Hybrid retriever (BM25 + dense, fused with RRF)
         -> Relevance check (refuse if below threshold)
         -> Synthesizer (cited answer)
         -> Verifier (claim vs. evidence check)
         -> Answer + citations + verification report
```

## Project Structure

```
app/
  config.py        settings and retrieval hyperparameters
  models/          Pydantic schemas (documents, research, benchmark)
  ingestion/       loaders and chunker
  retrieval/       bm25.py, dense.py, hybrid.py, indexer.py
  agent/           engine, query analyzer, synthesizer, verifier, LLM provider
  api/             FastAPI routes and server
  static/          web interface
data/
  sources/         8 research documents used as the corpus
  questions.json   10 benchmark questions with expected outcomes
docs/
  retrieval_approach.md
  tradeoffs.md
tests/             ingestion, retrieval, agent and unanswerable-query tests
cli.py             terminal interface and benchmark runner
run.sh             one-command setup and start
```

## Installation

```bash
git clone https://github.com/ruchitha-ravishankar/rooman-ai-challenge.git
cd rooman-ai-challenge
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional: to use an external LLM, copy `.env.example` to `.env` and add your own key. Never commit `.env`.

## How to Run

Quick start (creates the environment, installs dependencies, indexes the corpus, starts the server):

```bash
./run.sh
```

Or start the web app manually and open http://localhost:8000:

```bash
uvicorn app.api.server:app --host 0.0.0.0 --port 8000
```

CLI:

```bash
python cli.py                       # interactive mode
python cli.py --query "your question"
python cli.py --benchmark           # run the 10-question benchmark
python cli.py --list-docs           # list indexed documents
```

Run the tests:

```bash
pytest tests/ -v
```

## Screenshots / Demo

_Add 2 screenshots here: the web interface showing an answer with citations, and a refusal on an out-of-corpus question._

## Results

On the 10-question benchmark in `data/questions.json` (8 factual, comparative and multi-document questions, 2 out-of-corpus questions), the system answered the 8 answerable questions with the expected source citations and refused both unanswerable ones.

| Category | Questions | Outcome |
|---|---|---|
| Factual | 5 | Expected source cited |
| Comparative | 2 | Expected source cited |
| Multi-document | 1 | Both expected sources cited |
| Out-of-corpus | 2 | Refused, no sources |

**How to read these results:** the benchmark and the corpus were both written by the author and are small, so this shows the pipeline works end to end; it is not evidence of general accuracy. Latencies in the original report (about 2 ms) were measured with the local, no-API mode and do not reflect LLM-backed runs.

## Challenges and Learnings

- Calibrating the refusal threshold: too low and the agent answers off-topic questions, too high and it refuses valid ones. See `docs/tradeoffs.md`.
- Combining keyword and semantic retrieval: BM25 handles exact terms, acronyms and numbers better, while embeddings handle paraphrase, so the two are fused with RRF.
- N-gram claim verification is a lightweight check and can miss paraphrased or subtly wrong claims.

## Future Enhancements

- Larger, independently written evaluation set, with standard retrieval metrics (recall@k, MRR).
- Cross-encoder reranking.
- Entailment-model-based claim verification instead of N-gram overlap.
- Support for user-uploaded corpora with persistent storage.

## License

MIT. See `LICENSE`.
