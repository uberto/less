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
    def __init__(self, base_directory: str, force_reset: bool = False):
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
        try:
            if force_reset:
                try:
                    self.chroma_client.delete_collection(name=COLLECTION_NAME)
                except ValueError:
                    pass  # Collection doesn't exist, that's fine
                self.collection = self.chroma_client.create_collection(
                    name=COLLECTION_NAME,
                    embedding_function=self.embedding_function,
                    metadata={
                        "hnsw:space": VECTOR_SPACE,
                        "base_directory": self.base_directory
                    }
                )
            else:
                self.collection = self.chroma_client.get_collection(
                    name=COLLECTION_NAME,
                    embedding_function=self.embedding_function
                )
        except ValueError:
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

    def create_chunks(self, text: str, page_number: int = None) -> List[Dict[str, Any]]:
        """Create semantic chunks using spaCy's sentence segmentation."""
        if not isinstance(text, str):
            print(f"Warning: Expected string but got {type(text)}")
            return []

        # Clean the text
        text = text.strip()
        if not text:
            return []

        try:
            # Process with spaCy
            doc = self.nlp(text)
            chunks = []
            current_chunk = ""
            current_length = 0

            # Split into sentences
            for sent in doc.sents:
                sentence = sent.text.strip()
                if not sentence:
                    continue

                # If adding this sentence would exceed chunk size, start a new chunk
                if current_length + len(sentence) + 2 > self.chunk_size:
                    if current_chunk:
                        chunks.append({
                            'text': current_chunk.strip(),
                            'page': page_number
                        })
                    current_chunk = sentence + ". "
                    current_length = len(sentence) + 2
                # Otherwise, add the sentence to the current chunk
                else:
                    current_chunk += sentence + ". "
                    current_length += len(sentence) + 2

            # Add the last chunk if it exists
            if current_chunk:
                chunks.append({
                    'text': current_chunk.strip(),
                    'page': page_number
                })

            # If any chunk is too long, split it further
            final_chunks = []
            for chunk in chunks:
                chunk_text = chunk['text']
                if len(chunk_text) > self.chunk_size:
                    # Split into smaller chunks
                    start = 0
                    while start < len(chunk_text):
                        end = start + self.chunk_size
                        # Try to find a sentence boundary
                        if end < len(chunk_text):
                            # Look for the last period before the chunk size
                            last_period = chunk_text.rfind('. ', start, end)
                            if last_period != -1:
                                end = last_period + 1

                        sub_chunk = chunk_text[start:end].strip()
                        if sub_chunk:
                            final_chunks.append({
                                'text': sub_chunk,
                                'page': page_number
                            })
                        start = end
                else:
                    final_chunks.append(chunk)

            return final_chunks

        except Exception as e:
            print(f"Warning: Error creating chunks: {str(e)}")
            # Fallback to simple chunking if spaCy fails
            chunks = []
            for i in range(0, len(text), self.chunk_size):
                chunk = text[i:i + self.chunk_size].strip()
                if chunk:
                    chunks.append({
                        'text': chunk,
                        'page': page_number
                    })
            return chunks

    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract text content from a PDF file with page numbers.
        
        Returns:
            List of dictionaries containing text and page number for each page.
        """
        try:
            with open(pdf_path, 'rb') as file:
                try:
                    reader = PdfReader(file)
                    if reader.is_encrypted:
                        try:
                            reader.decrypt('')  # Try empty password first
                        except:
                            print(f"Warning: {pdf_path} is encrypted and requires a password")
                            return []
                    
                    pages_content = []
                    total_pages = len(reader.pages)
                    
                    for i in tqdm(range(total_pages), 
                                 desc=f"Reading {os.path.basename(pdf_path)}",
                                 leave=False):
                        try:
                            page = reader.pages[i]
                            text = ""
                            
                            # Try multiple text extraction methods
                            try:
                                text = page.extract_text()
                            except Exception as e1:
                                print(f"Warning: Primary text extraction failed for page {i+1} in {pdf_path}: {str(e1)}")
                                try:
                                    # Fallback: try to extract text from raw content
                                    if hasattr(page, '_objects'):
                                        for obj in page._objects.values():
                                            if hasattr(obj, 'get_data'):
                                                try:
                                                    text = obj.get_data().decode('utf-8', errors='ignore')
                                                    if text.strip():
                                                        break
                                                except:
                                                    continue
                                except Exception as e2:
                                    print(f"Warning: Fallback text extraction failed for page {i+1} in {pdf_path}: {str(e2)}")

                            # Clean and validate the extracted text
                            if text:
                                text = text.strip()
                                # Remove any non-printable characters
                                text = ''.join(char for char in text if char.isprintable() or char in '\n\t')
                                
                            if text:  # Only add non-empty pages
                                pages_content.append({
                                    'text': text,
                                    'page': i + 1  # 1-based page numbers
                                })
                        except Exception as e:
                            print(f"Warning: Could not process page {i+1} in {pdf_path}: {str(e)}")
                            continue
                            
                    if not pages_content:
                        print(f"Warning: No text content extracted from {pdf_path}")
                    return pages_content
                    
                except Exception as e:
                    print(f"Error: Invalid PDF format in {pdf_path}: {str(e)}")
                    return ""
                except Exception as e:
                    print(f"Error processing {pdf_path}: {str(e)}")
                    return ""
        except Exception as e:
            print(f"Error opening {pdf_path}: {str(e)}")
            return ""

    def extract_text_from_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract text content from a text file.
        
        Returns:
            List of dictionaries containing text and page number (always 1 for text files).
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()
                if content:
                    return [{
                        'text': content.strip(),
                        'page': 1  # Text files are considered as single page
                    }]
                else:
                    print(f"Warning: No content in {file_path}")
                    return []
        except Exception as e:
            print(f"Error reading {file_path}: {str(e)}")
            return []
            
    def index_documents(self) -> None:
        """Index all supported files in the specified directory using sliding window chunking."""
        # Get list of supported files
        files = [f for f in os.listdir(self.base_directory) if any(f.lower().endswith(ext) for ext in SUPPORTED_EXTENSIONS)]
        print(f"Found {len(files)} files...")

        successful = 0
        skipped = 0
        failed = 0
        
        for file_name in tqdm(files, desc="Processing files", unit="file"):
            try:
                file_path = os.path.join(self.base_directory, file_name)
                doc_id = self.get_document_id(file_path)

                # Check if document is already indexed
                try:
                    existing_docs = self.collection.get(
                        ids=[f"{doc_id}_0"]  # Try to get the first chunk
                    )
                    if existing_docs and len(existing_docs['ids']) > 0:
                        print(f"\nSkipping {file_name} - already indexed")
                        skipped += 1
                        continue
                except Exception:
                    pass  # Document not found, continue with indexing

                # Extract text based on file type
                pages = []
                file_type = ""
                
                if file_name.lower().endswith('.pdf'):
                    pages = self.extract_text_from_pdf(file_path)
                    file_type = "pdf"
                elif file_name.lower().endswith(('.txt', '.md')):
                    pages = self.extract_text_from_file(file_path)
                    file_type = "text"
                
                if not pages:
                    failed += 1
                    continue

                all_chunks = []
                chunk_metadatas = []

                # Process each page
                chunk_index = 0
                for page_content in pages:
                    page_chunks = self.create_chunks(page_content['text'], page_content['page'])
                    if not page_chunks:
                        continue

                    # Prepare chunks for batch addition
                    for chunk in page_chunks:
                        all_chunks.append(chunk['text'])
                        chunk_metadatas.append({
                            "doc_id": doc_id,
                            "file_path": file_path,
                            "file_name": file_name,
                            "chunk_index": str(chunk_index),
                            "page": str(chunk['page']),  # Store page number
                            "source": file_type
                        })
                        chunk_index += 1

                if not all_chunks:
                    print(f"\nWarning: No text content found in {pdf_file}")
                    failed += 1
                    continue

                # Update total chunks in metadata
                for metadata in chunk_metadatas:
                    metadata["total_chunks"] = str(len(all_chunks))

                # Prepare IDs
                ids = [f"{doc_id}_{i}" for i in range(len(all_chunks))]

                # Add the chunks to ChromaDB in batches to avoid memory issues
                batch_size = 100
                for i in range(0, len(all_chunks), batch_size):
                    end_idx = min(i + batch_size, len(all_chunks))
                    self.collection.add(
                        ids=ids[i:end_idx],
                        documents=all_chunks[i:end_idx],
                        metadatas=chunk_metadatas[i:end_idx]
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
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset the index before indexing"
    )
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.directory):
        print(f"Error: {args.directory} is not a directory")
        return 1
        
    try:
        if args.reset:
            print("Resetting index...")
        indexer = DocIndexer(args.directory, force_reset=args.reset)
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
