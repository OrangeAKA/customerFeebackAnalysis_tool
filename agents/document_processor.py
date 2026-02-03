"""
Document Processing for RAG Pipeline

Handles PDF and Markdown file parsing, text chunking, and ChromaDB indexing
for the root cause analysis feature.
"""

import re
from typing import List, Dict, Optional, Any
from io import BytesIO


class DocumentProcessor:
    """
    Processes documents (PDF, Markdown, Text) into chunks and stores them
    in ChromaDB for retrieval.
    """
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize the document processor.
        
        Args:
            chunk_size: Target size for text chunks (in characters)
            chunk_overlap: Overlap between consecutive chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._chroma_client = None
        self._embedding_function = None
    
    def _get_chroma_client(self):
        """Lazy-load ChromaDB client."""
        if self._chroma_client is None:
            import chromadb
            self._chroma_client = chromadb.Client()
        return self._chroma_client
    
    def _get_embedding_function(self):
        """Lazy-load embedding function."""
        if self._embedding_function is None:
            from chromadb.utils import embedding_functions
            self._embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
        return self._embedding_function
    
    def parse_pdf(self, file_content: bytes, filename: str) -> str:
        """
        Parse PDF file content to text.
        
        Args:
            file_content: Raw bytes of the PDF file
            filename: Name of the file for metadata
            
        Returns:
            Extracted text from the PDF
        """
        try:
            from pypdf import PdfReader
            
            pdf_file = BytesIO(file_content)
            reader = PdfReader(pdf_file)
            
            text_parts = []
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(f"[Page {page_num + 1}]\n{page_text}")
            
            return "\n\n".join(text_parts)
            
        except Exception as e:
            raise ValueError(f"Failed to parse PDF '{filename}': {str(e)}")
    
    def parse_markdown(self, file_content: bytes, filename: str) -> str:
        """
        Parse Markdown file content.
        
        Args:
            file_content: Raw bytes of the markdown file
            filename: Name of the file for metadata
            
        Returns:
            Text content from the markdown
        """
        try:
            text = file_content.decode('utf-8')
            # Keep markdown structure for context
            return text
        except Exception as e:
            raise ValueError(f"Failed to parse Markdown '{filename}': {str(e)}")
    
    def parse_text(self, file_content: bytes, filename: str) -> str:
        """
        Parse plain text file content.
        
        Args:
            file_content: Raw bytes of the text file
            filename: Name of the file for metadata
            
        Returns:
            Text content
        """
        try:
            return file_content.decode('utf-8')
        except Exception as e:
            raise ValueError(f"Failed to parse text file '{filename}': {str(e)}")
    
    def parse_file(self, uploaded_file) -> str:
        """
        Parse an uploaded file based on its extension.
        
        Args:
            uploaded_file: Streamlit UploadedFile object
            
        Returns:
            Extracted text content
        """
        filename = uploaded_file.name.lower()
        content = uploaded_file.read()
        
        # Reset file pointer for potential re-reads
        uploaded_file.seek(0)
        
        if filename.endswith('.pdf'):
            return self.parse_pdf(content, uploaded_file.name)
        elif filename.endswith('.md') or filename.endswith('.markdown'):
            return self.parse_markdown(content, uploaded_file.name)
        elif filename.endswith('.txt'):
            return self.parse_text(content, uploaded_file.name)
        else:
            # Try to parse as text
            try:
                return self.parse_text(content, uploaded_file.name)
            except:
                raise ValueError(f"Unsupported file format: {uploaded_file.name}")
    
    def chunk_text(self, text: str, source: str) -> List[Dict[str, str]]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Full text to chunk
            source: Source filename for metadata
            
        Returns:
            List of chunk dictionaries with 'text' and 'metadata'
        """
        chunks = []
        
        # Split by paragraphs first to maintain context
        paragraphs = re.split(r'\n\s*\n', text)
        
        current_chunk = ""
        chunk_id = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # If adding this paragraph would exceed chunk size
            if len(current_chunk) + len(para) > self.chunk_size and current_chunk:
                chunks.append({
                    'text': current_chunk.strip(),
                    'metadata': {
                        'source': source,
                        'chunk_id': chunk_id
                    }
                })
                chunk_id += 1
                
                # Keep overlap from previous chunk
                if self.chunk_overlap > 0:
                    words = current_chunk.split()
                    overlap_words = words[-min(len(words), self.chunk_overlap // 5):]
                    current_chunk = " ".join(overlap_words) + "\n\n" + para
                else:
                    current_chunk = para
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
        
        # Don't forget the last chunk
        if current_chunk.strip():
            chunks.append({
                'text': current_chunk.strip(),
                'metadata': {
                    'source': source,
                    'chunk_id': chunk_id
                }
            })
        
        return chunks
    
    def create_collection(self, collection_name: str) -> Any:
        """
        Create or get a ChromaDB collection.
        
        Args:
            collection_name: Name for the collection
            
        Returns:
            ChromaDB collection
        """
        client = self._get_chroma_client()
        embedding_fn = self._get_embedding_function()
        
        # Delete existing collection if it exists
        try:
            client.delete_collection(collection_name)
        except:
            pass
        
        return client.create_collection(
            name=collection_name,
            embedding_function=embedding_fn
        )
    
    def index_documents(
        self, 
        files: List[Any], 
        collection_name: str = "documents"
    ) -> Any:
        """
        Process and index multiple documents into ChromaDB.
        
        Args:
            files: List of Streamlit UploadedFile objects
            collection_name: Name for the ChromaDB collection
            
        Returns:
            ChromaDB collection with indexed documents
        """
        collection = self.create_collection(collection_name)
        
        all_chunks = []
        for file in files:
            try:
                text = self.parse_file(file)
                chunks = self.chunk_text(text, file.name)
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"Warning: Failed to process {file.name}: {str(e)}")
                continue
        
        if not all_chunks:
            return None
        
        # Add chunks to collection
        collection.add(
            documents=[c['text'] for c in all_chunks],
            metadatas=[c['metadata'] for c in all_chunks],
            ids=[f"chunk_{i}" for i in range(len(all_chunks))]
        )
        
        return collection
    
    def get_document_stats(self, collection: Any) -> Dict[str, Any]:
        """
        Get statistics about indexed documents.
        
        Args:
            collection: ChromaDB collection
            
        Returns:
            Dictionary with document statistics
        """
        if collection is None:
            return {'total_chunks': 0, 'sources': []}
        
        try:
            count = collection.count()
            # Get unique sources
            results = collection.get(include=['metadatas'])
            sources = list(set(
                m.get('source', 'Unknown') 
                for m in results.get('metadatas', [])
            ))
            
            return {
                'total_chunks': count,
                'sources': sources
            }
        except:
            return {'total_chunks': 0, 'sources': []}
