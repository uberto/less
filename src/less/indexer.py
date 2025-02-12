#!/usr/bin/env python3

import os
import sys
import argparse
from pypdf import PdfReader
from tqdm import tqdm
import chromadb
from chromadb.utils import embedding_functions
import hashlib
from typing import List, Dict, Any
import textwrap
import spacy

from .constants import (
    COLLECTION_NAME,
    INDEX_DIR_NAME,
    EMBEDDING_MODEL,
    VECTOR_SPACE,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    SUPPORTED_EXTENSIONS
)

class DocIndexer:
    def __init__(self, base_directory: str):
        """Initialize the indexer with a base directory to store indices.
        
        Args:
            base_directory: The directory containing the documents to index.
                           The index will be stored in .less_index subdirectory.
        """
        self.base_directory = os.path.abspath(base_directory)
        self.index_directory = os.path.join(self.base_directory, INDEX_DIR_NAME)
        self.chunk_size = DEFAULT_CHUNK_SIZE
        self.chunk_overlap = DEFAULT_CHUNK_OVERLAP
        
        # Create index directory if it doesn't exist
        os.makedirs(self.index_directory, exist_ok=True)
        
        # Initialize ChromaDB with sentence-transformers embedding function
        self.chroma_client = chromadb.PersistentClient(path=self.index_directory)
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        
        # Load spaCy model for better text chunking
        self.nlp = spacy.load('en_core_web_sm')
        
        # Create or get the collection
        collections = self.chroma_client.list_collections()
        collection_exists = any(c.name == COLLECTION_NAME for c in collections)
        
        if collection_exists:
            self.collection = self.chroma_client.get_collection(
                name=COLLECTION_NAME,
                embedding_function=self.embedding_function
            )
        else:
            self.collection = self.chroma_client.create_collection(
                name=COLLECTION_NAME,
                embedding_function=self.embedding_function,
                metadata={
                    "hnsw:space": VECTOR_SPACE,
                    "base_directory": self.base_directory
                }
            )

    def get_document_id(self, file_path: str) -> str:
        """Generate a unique ID for a document based on its path and content."""
        with open(file_path, 'rb') as f:
            content = f.read()
            return hashlib.sha256(content).hexdigest()

    def create_chunks(self, text: str) -> List[str]:
        """Create semantic chunks using spaCy's sentence segmentation."""
        doc = self.nlp(text)
        chunks = []
        current_chunk = ""
        current_length = 0

        for sent in doc.sents:
            sentence = sent.text.strip()
            if not sentence:
                continue

            # If adding this sentence would exceed chunk size, start a new chunk
            if current_length + len(sentence) + 2 > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
                current_length = len(sentence) + 2
            # Otherwise, add the sentence to the current chunk
            else:
                current_chunk += sentence + ". "
                current_length += len(sentence) + 2

        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(current_chunk.strip())

        # If chunks are too long, split them further
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > self.chunk_size:
                # Split into smaller chunks while trying to maintain sentence boundaries
                words = chunk.split()
                sub_chunk = ""
                sub_length = 0
                for word in words:
                    if sub_length + len(word) + 1 <= self.chunk_size:
                        sub_chunk += word + " "
                        sub_length += len(word) + 1
                    else:
                        if sub_chunk:
                            final_chunks.append(sub_chunk.strip())
                        sub_chunk = word + " "
                        sub_length = len(word) + 1
                if sub_chunk:
                    final_chunks.append(sub_chunk.strip())
            else:
                final_chunks.append(chunk)

        return final_chunks

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text content from a PDF file."""
        try:
            with open(pdf_path, 'rb') as file:
                try:
                    reader = PdfReader(file)
                    if reader.is_encrypted:
                        try:
                            reader.decrypt('')  # Try empty password first
                        except:
                            print(f"Warning: {pdf_path} is encrypted and requires a password")
                            return ""
                    
                    text = ""
                    total_pages = len(reader.pages)
                    
                    for i in tqdm(range(total_pages), 
                                 desc=f"Reading {os.path.basename(pdf_path)}",
                                 leave=False):
                        try:
                            page = reader.pages[i]
                            text += page.extract_text() + "\n"
                        except Exception as e:
                            print(f"Warning: Could not read page {i+1} in {pdf_path}: {str(e)}")
                            continue
                            
                    return text.strip()
                    
                except Exception as e:
                    print(f"Error: Invalid PDF format in {pdf_path}: {str(e)}")
                    return ""
                except Exception as e:
                    print(f"Error processing {pdf_path}: {str(e)}")
                    return ""
        except Exception as e:
            print(f"Error opening {pdf_path}: {str(e)}")
            return ""

    def index_documents(self) -> None:
        """Index all PDF files in the specified directory using sliding window chunking."""
        # Get list of PDF files
        pdf_files = [f for f in os.listdir(self.base_directory) if any(f.lower().endswith(ext) for ext in SUPPORTED_EXTENSIONS)]
        print(f"Found {len(pdf_files)} PDF files...")

        successful = 0
        skipped = 0
        failed = 0
        
        for pdf_file in tqdm(pdf_files, desc="Processing PDFs", unit="file"):
            try:
                pdf_path = os.path.join(self.base_directory, pdf_file)
                doc_id = self.get_document_id(pdf_path)

                # Check if document is already indexed
                try:
                    existing_docs = self.collection.get(
                        ids=[f"{doc_id}_0"]  # Try to get the first chunk
                    )
                    if existing_docs and len(existing_docs['ids']) > 0:
                        print(f"\nSkipping {pdf_file} - already indexed")
                        skipped += 1
                        continue
                except Exception:
                    pass  # Document not found, continue with indexing

                # Extract and chunk the text
                content = self.extract_text_from_pdf(pdf_path)
                if not content:
                    failed += 1
                    continue

                chunks = self.create_chunks(content)
                if not chunks:
                    print(f"\nWarning: No text content found in {pdf_file}")
                    failed += 1
                    continue
                
                # Prepare the chunks for ChromaDB
                ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
                metadatas = [{
                    "doc_id": doc_id,
                    "file_path": pdf_path,
                    "file_name": pdf_file,
                    "chunk_index": str(i),  # Convert to string for ChromaDB
                    "total_chunks": str(len(chunks)),  # Convert to string for ChromaDB
                    "source": "pdf"
                } for i in range(len(chunks))]

                # Add the chunks to ChromaDB in batches to avoid memory issues
                batch_size = 100
                for i in range(0, len(chunks), batch_size):
                    end_idx = min(i + batch_size, len(chunks))
                    self.collection.add(
                        ids=ids[i:end_idx],
                        documents=chunks[i:end_idx],
                        metadatas=metadatas[i:end_idx]
                    )
                
                successful += 1
                
            except Exception as e:
                print(f"\nError processing {pdf_file}: {str(e)}")
                failed += 1
                continue

        print("\nIndexing Summary:")
        print(f"✓ Successfully indexed: {successful} files")
        print(f"↷ Skipped (already indexed): {skipped} files")
        print(f"✗ Failed to index: {failed} files")

    def search(self, query_text: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search through indexed PDFs using vector similarity."""
        results = self.collection.query(
            query_texts=[query_text],
            n_results=limit
        )

        if not results or not results['distances']:
            print("No matching documents found.")
            return []

        search_results = []
        seen_files = set()

        for i, (doc_id, distance, metadata, text) in enumerate(zip(
            results['ids'][0],
            results['distances'][0],
            results['metadatas'][0],
            results['documents'][0]
        )):
            file_path = metadata['file_path']
            if file_path not in seen_files:
                seen_files.add(file_path)
                search_results.append({
                    'file_name': metadata['file_name'],
                    'file_path': file_path,
                    'score': 1 - distance,  # Convert distance to similarity score
                    'excerpt': textwrap.fill(text, width=80)
                })

        return search_results

def main():
    parser = argparse.ArgumentParser(
        description="LESS - LLM Empowered Semantic Search Indexer"
    )
    parser.add_argument(
        "directory",
        help="Directory containing PDF files to index"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Size of text chunks in characters (default: {DEFAULT_CHUNK_SIZE})"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help=f"Overlap between chunks in characters (default: {DEFAULT_CHUNK_OVERLAP})"
    )
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.directory):
        print(f"Error: {args.directory} is not a directory")
        return 1
        
    try:
        indexer = DocIndexer(args.directory)
        indexer.chunk_size = args.chunk_size
        indexer.chunk_overlap = args.chunk_overlap
        indexer.index_documents()
        return 0
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
