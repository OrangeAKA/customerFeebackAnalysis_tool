"""
ChromaDB Retrieval Tools for CrewAI Agents

Provides tools for searching product documentation and release notes
stored in ChromaDB vector store.
"""

from crewai.tools import BaseTool
from typing import Type, Optional, Any
from pydantic import BaseModel, Field


class SearchInput(BaseModel):
    """Input schema for document search."""
    query: str = Field(..., description="The search query to find relevant documents")


class DocumentSearchTool(BaseTool):
    """Tool for searching documents in ChromaDB."""
    
    name: str = "search_documents"
    description: str = (
        "Search through product documentation and release notes to find "
        "relevant information. Use this when you need to find documentation "
        "about features, known issues, or release changes."
    )
    args_schema: Type[BaseModel] = SearchInput
    
    collection: Any = None
    doc_type: str = "documents"
    
    def __init__(self, collection: Any, doc_type: str = "documents", **kwargs):
        super().__init__(**kwargs)
        self.collection = collection
        self.doc_type = doc_type
        self.description = (
            f"Search through {doc_type} to find relevant information. "
            "Use this when you need context about features, issues, or changes."
        )
    
    def _run(self, query: str) -> str:
        """Execute the search query against ChromaDB."""
        if self.collection is None:
            return f"No {self.doc_type} have been uploaded for searching."
        
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=5
            )
            
            if not results or not results.get('documents') or not results['documents'][0]:
                return f"No relevant {self.doc_type} found for query: {query}"
            
            # Format results
            formatted_results = []
            for i, (doc, metadata) in enumerate(zip(
                results['documents'][0], 
                results.get('metadatas', [[]])[0] if results.get('metadatas') else [{}] * len(results['documents'][0])
            )):
                source = metadata.get('source', 'Unknown') if metadata else 'Unknown'
                formatted_results.append(f"[Result {i+1}] (Source: {source})\n{doc}")
            
            return "\n\n---\n\n".join(formatted_results)
            
        except Exception as e:
            return f"Error searching {self.doc_type}: {str(e)}"


def create_retrieval_tool(collection: Any, doc_type: str = "documents") -> DocumentSearchTool:
    """
    Factory function to create a document search tool.
    
    Args:
        collection: ChromaDB collection to search
        doc_type: Type of documents (e.g., "product documentation", "release notes")
    
    Returns:
        DocumentSearchTool instance configured for the collection
    """
    return DocumentSearchTool(collection=collection, doc_type=doc_type)
