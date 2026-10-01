# Moment RAG

Moment RAG is a session-scoped video research studio. Add one or more YouTube
videos, ask questions across their captions, and jump from every citation to the
exact moment in the source video.

The app starts with an empty live index. Only videos added after the server starts
are searchable in that server session, so bundled course data does not leak into
answers or suggested questions.

## What it demonstrates

- YouTube caption ingestion with timestamps
- Semantic transcript chunking
- OpenAI dense embeddings
- In-memory Qdrant hybrid retrieval (dense vectors and BM25 sparse vectors)
- Query decomposition and metadata-aware self-query
- Reciprocal Rank Fusion (RRF) and cross-encoder reranking
- Streamed answers with timestamped citations
- Video-aware suggested questions generated from the current session
- Side-by-side comparison with a simple dense-retrieval baseline

## Run locally

Requirements:

- Python 3.10 or newer
- An OpenAI API key
- YouTube videos with captions enabled

From the repository root:

```bash
cd modules/Module_3_Production_Agentic_RAG_AI_Systems/Moment_RAG
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Add your key to `.env`:

```dotenv
OPENAI_API_KEY=your_openai_api_key
```

Start the app:

```bash
uvicorn app:app --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000), add a YouTube URL, wait for
ingestion to complete, and ask a question.

Do not use Uvicorn's `--reload` option for a normal demo. Reloading restarts the
process and clears the session-scoped video list and in-memory Qdrant index.

The first ingestion or query can take longer while FastEmbed downloads its local
semantic chunking, sparse retrieval, and reranker models.

## How it works

### Ingestion

1. The UI sends up to 10 YouTube URLs to `/ingest/batch`.
2. `youtube-transcript-api` fetches each video's captions and timestamps.
3. The transcript is converted into timed words and semantic chunks.
4. OpenAI creates a dense embedding for every chunk.
5. Qdrant indexes dense vectors, BM25 sparse vectors, and optional generated
   question vectors.
6. The app generates three suggested questions from the newly added videos.

The batch endpoint streams progress events for transcript download, chunking,
embedding, and indexing, which drive the status pills in the UI.

### Retrieval and answer generation

1. The question is checked by a lightweight input guardrail.
2. A self-query step infers useful metadata filters.
3. Complex questions are decomposed into focused sub-queries.
4. Each sub-query runs dense, BM25, and generated-question retrieval branches.
5. Reciprocal Rank Fusion combines the ranked result lists.
6. A cross-encoder reranks the strongest candidates.
7. The answer model streams a cited response from the retrieved moments.
8. Clicking a citation seeks the embedded YouTube player to its exact timestamp.

### Baseline comparison

After an answer is generated, **Compare pipelines** runs the same question through:

- **Baseline RAG:** one dense query over chunks from the current session.
- **Moment RAG:** self-query, decomposition, hybrid retrieval, RRF, reranking, and
  cited synthesis.

The comparison is tied to the current question. Typing or selecting a new question
clears the old comparison until **Compare pipelines** is clicked again.

## Session and data behavior

- `QDRANT_LOCATION=:memory:` is the default.
- The live Qdrant collection and `/videos` list start empty on every server boot.
- Only videos added in the current process are included in search, comparison, and
  question suggestions.
- Restarting the server clears the live session index.
- Downloaded transcripts and generated chunks are still written to
  `data/transcripts/` and `data/chunks/` as local processing artifacts. They are
  not automatically loaded on the next boot.

This separation keeps demos reproducible: local artifacts can help with debugging,
but the searchable corpus always reflects what the user added in the current
session.

## Configuration

Copy `.env.example` to `.env`. The main settings are:

| Variable | Default | Purpose |
| --- | --- | --- |
| `OPENAI_API_KEY` | — | Required for embeddings and LLM calls |
| `EMBED_MODEL` | `text-embedding-3-large` | Dense embedding model |
| `EMBED_DIM` | `1024` | Dense vector dimensions |
| `LLM_MODEL` | `gpt-4o-mini` | Query analysis, metadata, and suggestions |
| `SYNTH_MODEL` | `gpt-4o` | User-facing cited answer |
| `QDRANT_LOCATION` | `:memory:` | Qdrant storage location |
| `QDRANT_COLLECTION` | `moment_rag_chunks` | Qdrant collection name |
| `RERANKER_ENABLED` | `true` | Enables cross-encoder reranking |
| `RERANKER_MODEL` | `jinaai/jina-reranker-v1-tiny-en` | FastEmbed reranker |
| `SEMANTIC_CHUNKING` | `true` | Enables embedding-based chunk boundaries |
| `APP_USER` / `APP_PASSWORD` | unset | Optional HTTP Basic Authentication |

Use matching values for `EMBED_MODEL` and `EMBED_DIM`. If you intentionally point
`QDRANT_LOCATION` at persistent storage, the application still creates a fresh
collection at startup to preserve session-only behavior.

## Project structure

```text
Moment_RAG/
├── app.py                   # FastAPI routes, streaming, ingestion, comparison
├── studio_ui.html           # Single-page research studio
├── requirements.txt
├── data/
│   ├── transcripts/         # Generated timed transcripts
│   └── chunks/              # Generated semantic chunks
└── src/
    ├── baseline_rag.py      # Simple current-session comparison pipeline
    ├── embed.py             # OpenAI embedding client
    ├── ingest/
    │   ├── youtube_fetch.py # Caption download and timestamp conversion
    │   ├── chunk.py         # Transcript-to-chunk pipeline
    │   ├── semantic.py      # Semantic boundary detection
    │   └── metadata.py      # Chunk metadata enrichment
    └── selfbuilt/
        ├── search.py        # Moment RAG orchestration
        ├── qdrant_store.py  # Dense, sparse, and question-vector retrieval
        ├── query_breakdown.py
        ├── self_query.py
        └── rerank.py
```

## Troubleshooting

- **No captions found:** choose a public YouTube video with captions enabled.
- **No sources in an answer:** confirm at least one video completed all ingestion
  stages in the current server session.
- **The video list disappeared:** the server restarted; add the videos again.
- **First request is slow:** wait for local FastEmbed model downloads to finish.
- **OpenAI authentication error:** verify `OPENAI_API_KEY` in `.env`, then restart
  the server.
