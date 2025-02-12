import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any, Optional
import textwrap
from dataclasses import dataclass
from datetime import datetime

@dataclass
class SearchFilter:
    author: Optional[str] = None
    title: Optional[str] = None
    year: Optional[int] = None
    category: Optional[str] = None
    publisher: Optional[str] = None
    language: Optional[str] = None
    tags: Optional[List[str]] = None

class PDFSearcher:
    def __init__(self, persist_directory="pdf_index"):
        self.persist_directory = persist_directory
        
        # Initialize ChromaDB with sentence-transformers embedding function
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="multi-qa-mpnet-base-dot-v1"
        )
        
        # Get the collection
        self.collection = self.chroma_client.get_collection(
            name="pdf_collection",
            embedding_function=self.embedding_function
        )

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

    def search(self, 
               query_text: str, 
               search_filter: Optional[SearchFilter] = None,
               limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search through indexed PDFs using vector similarity and metadata filters.
        
        Args:
            query_text: The search query
            search_filter: Optional SearchFilter object for metadata filtering
            limit: Maximum number of results to return
            
        Returns:
            List of search results with metadata and excerpts
        """
        where = self._build_where_clause(search_filter)
        
        results = self.collection.query(
            query_texts=[query_text],
            where=where,
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
                result = {
                    'file_name': metadata['file_name'],
                    'file_path': file_path,
                    'score': 1 - distance,  # Convert distance to similarity score
                    'excerpt': textwrap.fill(text, width=80)
                }
                
                # Add available metadata
                for key in ['author', 'title', 'year', 'category', 'publisher', 
                          'language', 'tags']:
                    if f'metadata/{key}' in metadata:
                        result[key] = metadata[f'metadata/{key}']
                
                search_results.append(result)

        return search_results

def main():
    searcher = PDFSearcher()
    # !!! dir
    
    
    while True:
        print("\nPDF Search Menu:")
        print("1. Simple Search")
        print("2. Advanced Search with Filters")
        print("3. Exit")
        
        choice = input("Enter your choice (1-3): ")
        
        if choice == "1":
            query = input("Enter your search query: ")
            results = searcher.search(query)
            
        elif choice == "2":
            query = input("Enter your search query: ")
            
            # Collect filter criteria
            print("\nEnter filter criteria (press Enter to skip):")
            author = input("Author: ").strip() or None
            title = input("Title: ").strip() or None
            year = input("Year: ").strip()
            year = int(year) if year.isdigit() else None
            category = input("Category: ").strip() or None
            publisher = input("Publisher: ").strip() or None
            language = input("Language: ").strip() or None
            tags = input("Tags (comma-separated): ").strip()
            tags = [t.strip() for t in tags.split(",")] if tags else None
            
            search_filter = SearchFilter(
                author=author,
                title=title,
                year=year,
                category=category,
                publisher=publisher,
                language=language,
                tags=tags
            )
            
            results = searcher.search(query, search_filter=search_filter)
            
        elif choice == "3":
            print("Goodbye!")
            break
        else:
            print("Invalid choice! Please try again.")
            continue
            
        # Display results
        if results:
            print("\nSearch Results:")
            for i, result in enumerate(results, 1):
                print(f"\n{i}. File: {result['file_name']}")
                print(f"   Path: {result['file_path']}")
                print(f"   Score: {result['score']:.2f}")
                
                # Display available metadata
                for key in ['author', 'title', 'year', 'category', 'publisher', 
                          'language', 'tags']:
                    if key in result:
                        print(f"   {key.title()}: {result[key]}")
                
                print(f"   Excerpt: {result['excerpt']}")
        else:
            print("No results found.")

if __name__ == "__main__":
    main()
