"""Constants shared between indexer and searcher."""

# Collection settings
COLLECTION_NAME = "document_collection"
INDEX_DIR_NAME = ".less_index"

# Model settings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
VECTOR_SPACE = "cosine"

# Chunking settings
DEFAULT_CHUNK_SIZE = 500  # characters per chunk
DEFAULT_CHUNK_OVERLAP = 50  # characters overlap between chunks

# Search settings
DEFAULT_SEARCH_LIMIT = 5  # number of results to return

# File types
SUPPORTED_EXTENSIONS = [".pdf",".md", ".txt"]  # list of supported file extensions
