# Project Description: Hybrid Vector + Graph AI Retrieval Engine

## 1. High-Level Overview

This project is a **backend + API** for an AI retrieval engine that combines:

- **Semantic vector search** using **ChromaDB** and **SentenceTransformers**
- **Graph-based reasoning** using **NetworkX**
- **Noun-phrase entity extraction** using **spaCy**

The system ingests raw text documents, turns them into dense embeddings, extracts key
concepts as graph entities (e.g. "machine learning", "neural networks"), and
builds a bipartite graph between documents and entities. At query time, it
performs **hybrid search**:

1. Vector search for semantic similarity
2. Graph expansion via entity/document relationships
3. Merges the two into a unified result set

The core of the system lives in the `backend/` and `models/` directories, with a
separate React/Vite frontend (documented in `frontend_desc.md`).

---

## 2. Code Structure (Backend)

Key backend files:

- `backend/main.py` – FastAPI app and HTTP endpoints
- `backend/db/chroma_db.py` – ChromaDB vector store wrapper
- `backend/db/graph_db.py` – NetworkX graph manager, persistence
- `backend/services/ingest_service.py` – document ingestion pipeline
- `backend/services/search_service.py` – semantic/vector search API
- `backend/services/hybrid_service.py` – hybrid (vector + graph) search and graph views
- `backend/services/entity_extractor.py` – spaCy-based noun-phrase entity extractor
- `models/embedding_model.py` – SentenceTransformer singleton + `generate_embedding`
- `run_dev.py` – dev server entrypoint (uvicorn with reload)
- `test_api.py` – end-to-end test script for all endpoints

Persistent data:

- `data/chroma_store/` – ChromaDB vector store (auto-created)
- `data/graph_store.pkl` – NetworkX graph pickled to disk (auto-created)

---

## 3. Data Model and Concepts

### 3.1 Documents

A **document** is arbitrary text plus optional metadata, identified by a
client-provided `doc_id`:

- Stored in ChromaDB with:
  - `id` = `doc_id`
  - `document` = original text
  - `embedding` = 384‑dim float vector
  - `metadata` = optional dict (e.g. `{ "category": "AI" }`)

### 3.2 Embeddings

`models/embedding_model.py` wraps a SentenceTransformer model
(`all-MiniLM-L6-v2`) in a singleton (`EmbeddingModel`) to avoid reloading:

- `generate_embedding(text: str) -> List[float]`
  - Validates non-empty text
  - Encodes to a NumPy vector, converts to a Python list

This function is used by both ingestion and search.

### 3.3 Entities (Noun-Phrase Keyphrases)

Entities are **normalized noun phrases** extracted from document text using spaCy
(see `backend/services/entity_extractor.py`). Examples:

- `"machine learning"`
- `"neural networks"`
- `"statistical models"`
- `"natural language processing"`
- `"artificial intelligence"`

They are stored as **entity nodes** in the NetworkX graph and connected to
documents by edges.

### 3.4 Graph Structure

The graph (managed by `backend/db/graph_db.py`) is essentially bipartite:

- **Document nodes**: `node_type='document'`, key = `doc_id`
- **Entity nodes**: `node_type='entity'`, key = entity string
- **Edges**: `doc_id ↔ entity` (undirected), meaning "document mentions entity"

The graph is persisted to `data/graph_store.pkl` so it survives restarts.

---

## 4. ChromaDB: Vector Store Layer

File: `backend/db/chroma_db.py`

This module defines `ChromaDBManager`, a thin wrapper around a persistent
ChromaDB collection named `"documents"`:

- Constructor:
  - Ensures `./data/chroma_store` exists
  - Creates `chromadb.PersistentClient(path=persist_directory)`
  - Gets collection `"documents"`

Key methods:

- `add_document_to_chroma(doc_id, text, embedding, metadata)`
  - Adds a single document + embedding into ChromaDB
  - Only sets `metadatas` if metadata is non-empty

- `semantic_search(query_embedding, top_k)`
  - Wraps `collection.query(query_embeddings=[...], n_results=top_k)`
  - Returns a normalized dict: `doc_ids`, `documents`, `metadatas`, `distances`

- `get_document(doc_id)`
  - `collection.get(ids=[doc_id], include=["documents","metadatas","embeddings"])`
  - Handles different Chroma return shapes safely
  - Returns a dict with `doc_id`, `document`, `metadata`, `embedding` or `None`

- `list_document_ids()` / `count_documents()`
  - Utility methods used by APIs (e.g. `/stats`).

A singleton accessor `get_chroma_db()` creates/returns a global `ChromaDBManager`
instance.

---

## 5. GraphDB: NetworkX Graph Layer

File: `backend/db/graph_db.py`

`GraphDBManager` owns a NetworkX `Graph` and handles persistence:

- `__init__(persist_path="./data/graph_store.pkl")`:
  - Calls `load_graph()` which either loads from pickle or creates an empty graph

Persistence:

- `load_graph()` – loads `graph_store.pkl` if it exists
- `save_graph()` – saves the current graph to `graph_store.pkl`

Core graph operations:

- `add_document_node(doc_id, **attrs)` – adds document node, `node_type='document'`
- `add_entity_node(entity, **attrs)` – adds entity node, `node_type='entity'`
- `add_edge_between(node1, node2, **attrs)` – connects documents ↔ entities

Read/query helpers:

- `get_neighbors(node, depth=1)` – BFS traversal up to `depth` hops
- `get_all_nodes()` – list of all nodes
- `get_node_attributes(node)` – attributes of a node
- `get_document_entities(doc_id)` – entities connected to a given document
- `get_related_documents(entity)` – documents connected to a given entity
- `node_exists(node)` – existence check
- `get_graph_stats()` – counts:
  - total_nodes, total_edges, document_nodes, entity_nodes

A singleton accessor `get_graph_db()` manages a single global instance.

---

## 6. Entity Extraction: spaCy Noun Phrases

File: `backend/services/entity_extractor.py`

This module encapsulates **spaCy-based keyphrase extraction** for graph
entities.

Model loading:

- Tries `en_core_web_md` first, then falls back to `en_core_web_sm`
- Uses an `@lru_cache` helper `_load_spacy_model()` so it only loads once
- Raises a clear `RuntimeError` if a model is not installed

Core function:

- `extract_entities_spacy(text: str) -> List[str>`

Logic:

1. Run `doc = _nlp(text)`
2. Iterate over `doc.noun_chunks`
3. For each chunk:
   - Remove tokens that are space or punctuation
   - Skip if the remaining tokens are all stopwords
   - Build a surface string from tokens (`" " .join(t.text ...)`)
   - Normalize via `_normalize_phrase()`:
     - lowercase
     - trim whitespace
     - strip leading/trailing punctuation
     - require at least one alphabetic character
4. Deduplicate phrases while preserving order (`seen` set)

The output is a compact list of high-quality noun phrases that become **entity
nodes** in the graph.

---

## 7. Document Ingestion Pipeline

File: `backend/services/ingest_service.py`

### 7.1 `ingest_document(doc_id, text, metadata)`

This is the central ingestion function used by `POST /add_document`.

Pipeline:

1. **Generate embedding**
   - Calls `generate_embedding(text)` from `models/embedding_model.py`.

2. **Store in ChromaDB**
   - Gets the singleton via `get_chroma_db()`
   - Calls `add_document_to_chroma(doc_id, text, embedding, metadata)`
   - If this fails, returns an error without touching the graph.

3. **Extract entities (noun phrases)**
   - Calls `extract_entities_spacy(text)` from `backend/services/entity_extractor.py`
   - Produces phrases like `"machine learning"`, `"neural networks"`.

4. **Build graph**
   - Gets graph manager via `get_graph_db()`
   - Adds a **document node**: `graph_db.add_document_node(doc_id)`
   - For each entity:
     - `graph_db.add_entity_node(entity)`
     - `graph_db.add_edge_between(doc_id, entity)`

5. **Persist graph**
   - Calls `graph_db.save_graph()` to write `graph_store.pkl` to disk.

6. **Return summary**
   - On success, returns:
     - `success=True`
     - `doc_id`
     - `entities_extracted` + `entities` list
     - `metadata` (or `{}`)
     - `embedding_dim` (length of the embedding vector)

### 7.2 `batch_ingest_documents(documents)`

- Iterates over a list of `{ doc_id, text, metadata }`
- Calls `ingest_document` for each
- Returns totals: `total`, `successful`, `failed`, plus per-document details

---

## 8. Semantic Search Pipeline

File: `backend/services/search_service.py`

### 8.1 `semantic_search(query: str, top_k: int = 5)`

Steps:

1. **Embed query**
   - Uses `generate_embedding(query)` (same model as ingestion)

2. **Search in ChromaDB**
   - Calls `chroma_db.semantic_search(query_embedding, top_k)`

3. **Format results**
   - Produces a list of result dicts with:
     - `doc_id`
     - `document` (full text)
     - `metadata`
     - `distance` (Chroma distance)
     - `relevance_score = 1 - distance`

This function is used by the `GET /search` endpoint.

### 8.2 Additional helpers

- `search_by_document_id(doc_id)` – wraps `chroma_db.get_document()`
- `get_similar_documents(doc_id, top_k)` – uses a stored document embedding to
  perform a similarity search against other documents.

---

## 9. Hybrid Search and Graph Views

File: `backend/services/hybrid_service.py`

### 9.1 `hybrid_search(query: str, top_k: int = 5, graph_depth: int = 1)`

This function combines **vector search** and **graph expansion**.

Pipeline:

1. **Semantic vector search**
   - Calls `semantic_search(query, top_k)`
   - Collects initial `vector_results` and `vector_doc_ids`

2. **Graph expansion**
   - For each `doc_id` in `vector_doc_ids`, calls
     `graph_db.get_neighbors(doc_id, depth=graph_depth)`
   - Aggregates all neighbors:
     - Splits them into **document nodes** and **entity nodes** (based on
       `node_type` in graph attributes)

3. **Fetch expanded documents from ChromaDB**
   - For document neighbors that were *not* already in the vector hits,
     calls `chroma_db.get_document(doc_id)`
   - Builds `graph_expansion_results` list.

4. **Assemble hybrid results**
   - Vector hits are copied and tagged with:
     - `source = "vector_search"`
     - `graph_neighbors_count`
   - Graph-expanded docs are tagged with `source = "graph_expansion"`
   - Concatenated into `hybrid_results`

5. **Entity info**
   - For up to 20 entity nodes, builds a summary object:
     - `entity` (string)
     - `related_document_count`
     - `related_documents` (doc IDs)

6. **Return structure**
   - `vector_hits` + count
   - `graph_expansion` + count
   - `entities` + count
   - `hybrid_results` + count

This is exposed via the `GET /hybrid` endpoint.

### 9.2 `graph_neighbors(doc_id: str, depth: int = 1)`

- Uses `graph_db.get_neighbors(doc_id, depth)` and categorizes neighbors into:
  - `document_neighbors`
  - `entity_neighbors`
- Returns counts and full neighbor list
- Used by the `/graph_neighbors` endpoint.

### 9.3 `get_document_relationships(doc_id: str)`

- Uses:
  - `graph_db.get_document_entities(doc_id)` to find entities for a document
  - `graph_db.get_related_documents(entity)` for each entity
- Builds:
  - `entities` list + `entities_count`
  - `related_via_entities` mapping: `entity -> [doc_ids]`
  - `related_documents` (union of all related docs excluding the source)
- Exposed via the `/relationships/{doc_id}` endpoint.

---

## 10. FastAPI Application and Endpoints

File: `backend/main.py`

`backend/main.py` wires everything into a FastAPI app (`app = FastAPI(...)`).
It also enables CORS for browser-based frontends.

### 10.1 Core endpoints

- `POST /add_document`
  - Body: `{ "doc_id": str, "text": str, "metadata": Optional[dict] }`
  - Calls `ingest_document()`

- `POST /add_documents`
  - Body: `{ "documents": [DocumentRequest, ...] }`
  - Calls `batch_ingest_documents()`

- `GET /search`
  - Query: `q`, `top_k`
  - Calls `semantic_search()`

- `GET /hybrid`
  - Query: `q`, `top_k`, `depth`
  - Calls `hybrid_search()`

- `GET /graph_neighbors`
  - Query: `doc_id`, `depth`
  - Calls `graph_neighbors()` from `hybrid_service`

- `GET /document/{doc_id}`
  - Retrieves a specific document from ChromaDB

- `GET /relationships/{doc_id}`
  - Detailed entity/document relationships for a given document

- `GET /stats`
  - Returns ChromaDB counts + graph stats + system status
  - Uses a **simple in-process cache** (with a TTL of ~5 seconds) so repeated
    polls from dashboards don’t hammer Chroma or the graph.

- `GET /health`
  - Simple health check endpoint.

### 10.2 Running the backend

- Development (with reload):
  - `python run_dev.py`

- Directly:
  - `python -m backend.main` or
  - `uvicorn backend.main:app --reload`

---

## 11. Testing and Demo Flow

File: `test_api.py`

The `test_api.py` script is an end-to-end smoke test of the entire backend:

1. `GET /health` – health check
2. `GET /` – root endpoint and metadata
3. `POST /add_document` – ingests 3 AI/NLP-related documents
4. `GET /stats` – verifies Chroma + graph stats
5. `GET /search` – semantic search
6. `GET /document/{doc_id}` – document retrieval
7. `GET /graph_neighbors` – graph traversal
8. `GET /relationships/{doc_id}` – entity-based relationships
9. `GET /hybrid` – hybrid search end-to-end

Running it:

```bash
python test_api.py
```

This exercises the **full ingestion → vector store → entity extraction → graph
→ hybrid search** pipeline.

---

## 12. Frontend (Brief)

The frontend is a separate Vite + React app described in `frontend_desc.md`. It
uses the same API endpoints documented above to:

- Ingest documents
- Run semantic and hybrid search
- Display stats from `/stats`
- Explore relationships via `/relationships` and `/graph_neighbors`

This `project_desc.md` focuses on the backend architecture that powers that UI.
