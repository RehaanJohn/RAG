# Retrieval-Augmented Generation: A Technical Overview

## What is RAG?

Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval
with large language model (LLM) generation. Instead of relying solely on the knowledge
encoded in a model's parameters during training, RAG retrieves relevant documents from
an external knowledge base at query time and provides them as context to the LLM.

This approach addresses two fundamental limitations of standalone LLMs:
1. Knowledge cutoff — models cannot answer questions about events after their training date.
2. Hallucination — models may generate plausible-sounding but factually incorrect content.

## Chunking Strategy

Document chunking is a critical preprocessing step in a RAG pipeline. The goal is to
split source documents into segments that are:
- Small enough to fit within the LLM's context window.
- Large enough to carry sufficient semantic meaning.
- Structured to preserve logical boundaries (headings, paragraphs, sections).

A structure-aware chunker first identifies heading boundaries and splits text along
them. Within each section, it applies a sliding-window token-based split, targeting
256 to 512 tokens per chunk with approximately 12% overlap to preserve context
across chunk boundaries.

Poor chunking is one of the most common causes of degraded RAG quality. Chunks that
are too small lose context; chunks that are too large dilute the relevant signal.

## Embedding Normalisation

For cosine similarity search, embedding vectors must be L2-normalised (unit vectors).
When vectors are normalised, the dot product between two vectors equals their cosine
similarity. This allows the use of inner-product-optimised indexing structures.

The BGE (BAAI General Embedding) family of models, such as BAAI/bge-large-en-v1.5,
produces 1024-dimensional embeddings and is designed to be used with normalised
embeddings for cosine similarity. Skipping normalisation when using cosine similarity
metrics can produce subtly incorrect rankings.

## Dense vs Sparse Retrieval

Dense retrieval uses neural embedding models to encode both queries and documents
into dense continuous vector spaces, then finds nearest neighbours in that space.
It captures semantic similarity even when exact keyword overlap is low.

Sparse retrieval (e.g., BM25, TF-IDF) works on term frequency statistics. It
excels at exact keyword matching but misses paraphrasing and conceptual synonymy.

Hybrid retrieval combines both signals, typically using Reciprocal Rank Fusion (RRF)
to merge ranked lists from both systems. Phase 1 of this system uses dense retrieval
only; hybrid retrieval is reserved for a future phase.

## HNSW Indexes

Hierarchical Navigable Small World (HNSW) is a graph-based approximate nearest
neighbour (ANN) indexing algorithm. It organises embedding vectors into a multi-layer
graph where higher layers have fewer, longer-range connections and lower layers have
denser, shorter-range connections.

HNSW is the preferred index type for pgvector in production workloads because:
- Query time is sublinear in the number of indexed vectors (typically O(log N)).
- It supports cosine, L2, and inner product distance metrics.
- The `m` parameter (edges per node) and `ef_construction` (candidate list size
  during build) allow quality-speed trade-offs.

Default values for this system: m=16, ef_construction=64, which are appropriate
for corpora up to several million chunks. For very large corpora, increasing
ef_construction to 128 or 256 improves recall at the cost of longer build time.

## Citation Validation

A robust RAG system must ensure that every generated answer is grounded in the
retrieved context. Citation validation enforces two rules:
1. Every factual sentence must carry an inline citation tag.
2. Every cited ID must correspond to a document that was actually retrieved.

Violations of rule 2 (hallucinated citations) are especially dangerous because
they give users a false sense of confidence in fabricated information.
