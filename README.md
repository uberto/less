# LESS (LLM Empowered Semantic Search)

This tool allows you to index and search through your documents (PDF, text, etc.) using ChromaDB and sentence transformers for efficient semantic search.

## Features

- Smart document indexing with automatic duplicate detection
- Sliding window chunking for better context preservation
- State-of-the-art embeddings using `multi-qa-mpnet-base-dot-v1`
- Vector storage using ChromaDB for fast similarity search
- Semantic search capabilities
- Simple command-line interface

## How it works

1. **Document Processing**:
   - Loads your documents
   - Checks if each document is already indexed (using content hash)
   - Extracts text content from PDFs

2. **Text Chunking**:
   - Uses sliding window chunking (500 characters with 50 character overlap)
   - Preserves context between chunks

3. **Embedding & Storage**:
   - Embeds text chunks using `multi-qa-mpnet-base-dot-v1` model
   - Stores vectors and metadata in ChromaDB
   - Maintains document-chunk relationships

## Installation

1. Make sure you have Poetry installed. If not, follow the [Poetry installation guide](https://python-poetry.org/docs/#installation)

2. Install the project dependencies:
```bash
poetry install
```

## Usage

Run the script using Poetry:
```bash
poetry run python indexer.py
```

The tool provides three options:

1. **Index directory**: Point the tool to a directory containing your documents (PDF, text, etc.) to create/update the search index
2. **Search indexed documents**: Search through your indexed documents using keywords
3. **Exit**: Close the application

## How it works

- The tool uses PyPDF2 to extract text from PDF files
- Whoosh is used as the search engine to create and query the index
- Search results include file paths, relevance scores, and highlighted excerpts
- The index is stored in a `.less_index` directory (created automatically)
