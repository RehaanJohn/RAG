"""
smoke_tests/generate_test_pdf.py

Generates a test PDF (smoke_tests/sample_docs/architecture_diagram.pdf)
containing structured text and an embedded raster diagram.

Uses PIL and matplotlib (which are available without extra dependencies).
"""
from __future__ import annotations

import io
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


def generate_pdf(output_path: Path) -> None:
    """Generate architecture_diagram.pdf using matplotlib."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(8.5, 11), dpi=100)

    # Title & Text
    fig.text(
        0.1, 0.92,
        "Vector Database Architecture and Latency Benchmarks",
        fontsize=16, weight="bold", color="#1a365d"
    )
    fig.text(
        0.1, 0.88,
        "System Architecture & Performance Overview",
        fontsize=13, weight="semibold", color="#2b6cb0"
    )

    body_text = (
        "This report analyzes the retrieval latency of various vector indexing backends\n"
        "for high-throughput RAG workloads. Our enterprise architecture integrates Postgres\n"
        "with pgvector using HNSW (Hierarchical Navigable Small World) indexes to balance\n"
        "search latency and strict multi-tenant data isolation.\n\n"
        "Key benchmark findings:\n"
        "• pgvector HNSW achieves 12ms average latency for 1024-dimensional embeddings\n"
        "• FAISS in-memory index reaches 8ms latency but requires manual index sharding\n"
        "• Managed cloud vector databases show 28ms to 45ms latency under similar load"
    )
    fig.text(0.1, 0.74, body_text, fontsize=10, linespacing=1.6, color="#2d3748")

    # Bar chart subplot
    ax = fig.add_axes([0.15, 0.35, 0.7, 0.32])
    backends = ["FAISS (In-Memory)", "pgvector HNSW", "Pinecone Cloud", "ChromaDB Local"]
    latencies = [8, 12, 28, 45]
    colors = ["#38a169", "#3182ce", "#d69e2e", "#e53e3e"]

    bars = ax.barh(backends, latencies, color=colors, height=0.55, edgecolor="#2d3748", linewidth=0.8)
    ax.set_xlabel("Query Latency (ms) - Lower is Better", fontsize=10, weight="bold")
    ax.set_title("Vector Database Query Latency Comparison (1024-dim, Top-K=10)", fontsize=11, weight="bold", pad=10)
    ax.set_xlim(0, 55)
    ax.grid(axis="x", linestyle="--", alpha=0.7)

    for bar in bars:
        width = bar.get_width()
        ax.text(width + 1.2, bar.get_y() + bar.get_height() / 2, f"{width} ms",
                va="center", ha="left", fontsize=9, weight="bold", color="#2d3748")

    # Lower section text
    lower_text = (
        "Chart Analysis & Visual Summary:\n"
        "As depicted in the latency comparison chart above, pgvector HNSW provides the optimal\n"
        "balance of speed (12ms) and transactional ACID guarantees while keeping all client data\n"
        "and embedding vectors securely hosted on-premise without external API egress."
    )
    fig.text(0.1, 0.18, lower_text, fontsize=10, linespacing=1.6, color="#2d3748")

    fig.savefig(str(output_path), format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Generated test PDF: {output_path} ({output_path.stat().st_size} bytes)")


if __name__ == "__main__":
    out = Path(__file__).parent / "sample_docs" / "architecture_diagram.pdf"
    generate_pdf(out)
