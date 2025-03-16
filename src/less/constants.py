"""Constants shared between indexer and searcher."""

# Collection settings
COLLECTION_NAME = "document_collection"
INDEX_DIR_NAME = ".less_index"

# Model settings
#try also nomic-embed-text
EMBEDDING_MODEL = "intfloat/e5-mistral-7b-instruct"
VECTOR_SPACE = "cosine"

# Chunking settings
DEFAULT_CHUNK_SIZE = 500  # characters per chunk
DEFAULT_CHUNK_OVERLAP = 50  # characters overlap between chunks

# Search settings
DEFAULT_SEARCH_LIMIT = 10  # number of results to return

# File types
# Search padding template
SEARCH_PADDING = "Query: {query}\nFind relevant information from the provided context."

# File types
SUPPORTED_EXTENSIONS = [".pdf",".md", ".txt"]  # list of supported file extensions
