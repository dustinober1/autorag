# AutoRAG Project Specification

## Overview

AutoRAG is a self-building Knowledge Graph RAG (Retrieval-Augmented Generation) toolkit that is local-first and designed for privacy-focused AI applications. It combines vector retrieval with knowledge graph extraction to provide grounded, citeable answers from local documents.

## Core Features

- **Multi-strategy Retrieval**: Hybrid search, fusion, and reranking
- **Knowledge Graph Extraction**: Build graphs from documents for better context
- **Local LLM Integration**: Ollama support for privacy-preserving inference
- **PDF Processing**: Advanced PDF parsing with pypdf and pdfplumber
- **arXiv Integration**: Search and ingest academic papers
- **Chunking Strategies**: Multiple options (fixed, semantic breakpoint, heading recursive, sentence window)
- **Evaluation Framework**: Metrics and A/B testing capabilities
- **Streamlit UI**: Professional web interface for document management and querying

## Tech Stack

- **Language:** Python 3.11+
- **Web Framework:** Streamlit 1.42+
- **Data Validation:** Pydantic 2.10+
- **CLI:** Typer
- **Database:** DuckDB 1.1+ (vector storage), SQLite (knowledge graph)
- **ML/AI:**
  - fastembed (embeddings)
  - torch, transformers (vision capabilities)
  - networkx (knowledge graph)
  - ollama (local LLM)
- **Document Processing:**
  - pypdf, pdfplumber (PDF parsing)
  - arxiv (academic paper API)
- **Configuration:** PyYAML
- **Testing:** pytest
- **Code Quality:** ruff, mypy

## Architecture Notes

- **Module Structure:**
  - `src/autokg_rag/ingest/` - Document ingestion and parsing
  - `src/autokg_rag/chunking/` - Text chunking strategies
  - `src/autokg_rag/retrieval/` - Retrieval strategies
  - `src/autokg_rag/vector/` - Vector indexing
  - `src/autokg_rag/kg/` - Knowledge graph operations
  - `src/autokg_rag/eval/` - Evaluation and metrics
  - `src/autokg_rag/arxiv/` - arXiv integration
  - `src/autokg_rag/answer/` - Answer generation and grounding
  - `src/autokg_rag/app_api/` - API endpoints for Streamlit
  - `app/` - Streamlit UI components

- **Storage:** Local-first with DuckDB for vectors and SQLite for knowledge graphs
- **No cloud dependencies:** Everything runs locally
- **Config-driven:** YAML-based configuration for embeddings, chunking, experiments

## Integrations

- **Ollama:** Local LLM inference (model selection, health checking)
- **arXiv API:** Academic paper search and ingestion
- **GitHub:** CI/CD workflows in `.github/workflows/`

## Non-Functional Requirements

- **Privacy:** All processing local, no external API calls (except arXiv)
- **Extensibility:** Plugin-based chunking, retrieval, and embedding strategies
- **Performance:** Configurable top-k, hybrid fusion, reranking support
- **Quality:** Type hints throughout, mypy strict mode, ruff formatting
