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

### Indexing Documents

To index a directory containing your documents (PDF, text, etc.):
```bash
poetry run python -m less.indexer /path/to/your/documents
```

This will:
- Create a `.less_index` directory to store the search index
- Process all supported documents (PDF, MD, TXT)
- Skip any previously indexed documents (using content hash)
- Show progress with a nice progress bar

### Searching Documents

To search through your indexed documents:
```bash
poetry run python -m less.searcher /path/to/your/documents "your search query"
```

This will:
- Return the top 10 most relevant results (configurable)
- Show the matching text snippets with context
- Display the source document for each match

### Examples

```bash
# Index all documents in your ebooks directory
poetry run python -m less.indexer ~/ebooks

# Search for specific topics in your indexed documents
poetry run python -m less.searcher ~/ebooks "machine learning fundamentals"
```

## How it works

- The tool uses PyPDF2 to extract text from PDF files
- Whoosh is used as the search engine to create and query the index
- Search results include file paths, relevance scores, and highlighted excerpts
- The index is stored in a `.less_index` directory (created automatically)
