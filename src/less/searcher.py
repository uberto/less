#!/usr/bin/env python3

import os
import sys
import argparse
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any, Optional
import textwrap
from dataclasses import dataclass
from datetime import datetime

from .constants import (
    COLLECTION_NAME,
    INDEX_DIR_NAME,
    DEFAULT_SEARCH_LIMIT
)

@dataclass
class SearchFilter:
    author: Optional[str] = None
    title: Optional[str] = None
    year: Optional[int] = None
    category: Optional[str] = None
    publisher: Optional[str] = None
    language: Optional[str] = None
    tags: Optional[List[str]] = None

class DocSearcher:
    def __init__(self, base_directory: str):
        """Initialize the searcher with a base directory containing the index.
        
        Args:
            base_directory: The directory containing the documents and index.
                           The index should be in .less_index subdirectory.
        """
        self.base_directory = os.path.abspath(base_directory)
        self.index_directory = os.path.join(self.base_directory, INDEX_DIR_NAME)
        
        if not os.path.exists(self.index_directory):
            raise ValueError(f"No index found in {self.base_directory}. Please run the indexer first.")
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(path=self.index_directory)
        
        # Get the collection
        try:
            self.collection = self.chroma_client.get_collection(name=COLLECTION_NAME)
        except ValueError:
            raise ValueError(f"No documents indexed in {self.base_directory}. Please run the indexer first.")

    def _build_where_clause(self, search_filter: Optional[SearchFilter] = None) -> Dict:
        """Build ChromaDB where clause from search filters."""
        where = {}
        
        if not search_filter:
            return where

        if search_filter.author:
            where["metadata/author"] = search_filter.author
        if search_filter.title:
            where["metadata/title"] = search_filter.title
        if search_filter.year:
            where["metadata/year"] = search_filter.year
        if search_filter.category:
            where["metadata/category"] = search_filter.category
        if search_filter.publisher:
            where["metadata/publisher"] = search_filter.publisher
        if search_filter.language:
            where["metadata/language"] = search_filter.language
        if search_filter.tags:
            where["metadata/tags"] = {"$containsAny": search_filter.tags}
            
        return where

    def search(self, query_text: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search through indexed PDFs using vector similarity.
        
        Args:
            query_text: The search query
            limit: Maximum number of results to return
            
        Returns:
            List of search results with metadata and content
        """
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=limit
            )
            
            if not results['ids'] or not results['ids'][0]:
                return []
            search_results = []
            for i, (doc_id, distance, metadata, text) in enumerate(zip(
                results['ids'][0],
                results['distances'][0],
                results['metadatas'][0],
                results['documents'][0]
            )):
                result = {
                    'file_path': metadata['file_path'],
                    'file_name': metadata['file_name'],
                    'page': int(metadata.get('page', '1')),  # Get page number, default to 1
                    'score': 1 - distance,  # Convert distance to similarity score
                    'content': text,
                    'metadata': metadata
                }
                search_results.append(result)

            # Sort results by score and then by page number
            search_results.sort(key=lambda x: (-x['score'], x['page']))
            return search_results
            
        except Exception as e:
            print(f"Search error: {str(e)}")
            return []

def main():
    parser = argparse.ArgumentParser(
        description="LESS - LLM Empowered Semantic Search"
    )
    parser.add_argument(
        "directory",
        help="Directory containing indexed documents"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_SEARCH_LIMIT,
        help=f"Number of results to show per query (default: {DEFAULT_SEARCH_LIMIT})"
    )
    
    args = parser.parse_args()
    
    try:
        searcher = DocSearcher(args.directory)
        print(f"\nConnected to index in {args.directory}")
        print("Type your search query. Press Ctrl+C or type 'exit' to quit.\n")
        
        while True:
            try:
                query = input("🔍 ").strip()
                if not query:
                    continue
                if query.lower() == 'exit':
                    break
                    
                results = searcher.search(query, limit=args.limit)
                
                if not results:
                    print("\nNo results found.\n")
                    continue
                    
                print("\nFound these relevant passages:\n")
                for i, result in enumerate(results, 1):
                    print(f"[{i}] From: {os.path.basename(result['file_path'])}")
                    print(f"Score: {result['score']:.2f}")
                    print("─" * 80)
                    print(textwrap.fill(result['content'], width=80))
                    print("─" * 80 + "\n")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\nError: {str(e)}\n")
                continue
                
        print("\nGoodbye!")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
        
    return 0

if __name__ == "__main__":
    main()
