"""
RAG (Retrieval-Augmented Generation) system for the personal assistant.
Handles document storage, retrieval, and context-aware generation.
"""
import os
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import hashlib
from datetime import datetime
import numpy as np

# Import required libraries
try:
    from sentence_transformers import SentenceTransformer
    from qdrant_client import QdrantClient
    from qdrant_client.models import VectorParams, Distance, PointStruct
    from qdrant_client.http import models
    from qdrant_client.http.exceptions import UnexpectedResponse
    from qdrant_client.http.models import Filter, FieldCondition, MatchValue
    
    # Try to import tqdm for progress bars
    try:
        from tqdm import tqdm
        TQDM_AVAILABLE = True
    except ImportError:
        TQDM_AVAILABLE = False
        
except ImportError as e:
    print("Error: Required packages not installed. Please install with:")
    print("pip install sentence-transformers qdrant-client numpy tqdm")
    raise

@dataclass
class Document:
    """Represents a document to be stored in the vector database."""
    text: str
    metadata: Dict[str, Any] = None
    embedding: Optional[np.ndarray] = None
    id: Optional[str] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if not self.id:
            self.id = hashlib.md5(self.text.encode()).hexdigest()

class RAGSystem:
    """
    A complete RAG (Retrieval-Augmented Generation) system that handles
    document storage, retrieval, and context-aware generation.
    """
    
    def __init__(
        self,
        collection_name: str = "assistant_knowledge",
        model_name: str = "all-MiniLM-L6-v2",
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        embedding_dim: int = 384
    ):
        """
        Initialize the RAG system.
        
        Args:
            collection_name: Name of the Qdrant collection
            model_name: Name of the sentence transformer model
            qdrant_url: URL of the Qdrant server
            qdrant_api_key: API key for Qdrant
            embedding_dim: Dimensionality of the embeddings
        """
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        
        # Initialize the embedding model
        self.embedding_model = SentenceTransformer(model_name)
        
        # Initialize Qdrant client with environment variable fallbacks
        self.qdrant = QdrantClient(
            url=qdrant_url or os.getenv("QDRANT_HOST_URL"),
            api_key=qdrant_api_key or os.getenv("QDRANT_API_KEY"),
            prefer_grpc=True
        )
        
        # Create collection if it doesn't exist
        self._initialize_collection()
    
    def _initialize_collection(self):
        """Initialize the Qdrant collection if it doesn't exist and set up required indices."""
        try:
            # Check if collection exists
            collections = self.qdrant.get_collections()
            collection_names = [collection.name for collection in collections.collections]
            
            if self.collection_name not in collection_names:
                # Create the collection with the vector configuration
                self.qdrant.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE
                    )
                )
                
                # Create an index on the metadata.type field for filtering
                self.qdrant.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="metadata.type",
                    field_schema="keyword"
                )
                
                print(f"Created new collection: {self.collection_name} with metadata.type index")
            else:
                # Ensure the metadata.type index exists
                try:
                    self.qdrant.create_payload_index(
                        collection_name=self.collection_name,
                        field_name="metadata.type",
                        field_schema="keyword",
                        wait=True
                    )
                    print(f"Ensured metadata.type index exists on collection: {self.collection_name}")
                except Exception as e:
                    print(f"Warning: Could not create metadata.type index (it may already exist): {str(e)}")
        except Exception as e:
            print(f"Error initializing collection: {e}")
            raise
    
    async def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of Document objects to add
            
        Returns:
            List of document IDs that were added
        """
        if not documents:
            return []
            
        try:
            # Generate embeddings for documents
            texts = [doc.text for doc in documents]
            
            # Use tqdm for progress if available
            if TQDM_AVAILABLE:
                embeddings = self.embedding_model.encode(
                    texts, 
                    show_progress_bar=True,
                    convert_to_numpy=True
                )
            else:
                embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
            
            # Prepare points for Qdrant
            points = []
            for doc, embedding in zip(documents, embeddings):
                point = PointStruct(
                    id=doc.id,
                    vector=embedding.tolist(),
                    payload={
                        "text": doc.text,
                        "metadata": doc.metadata,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )
                points.append(point)
            
            # Upsert points to Qdrant
            self.qdrant.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True
            )
            
            return [point.id for point in points]
            
        except Exception as e:
            print(f"Error adding documents: {e}")
            raise
    
    async def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.5,
        filter_metadata: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant documents.
        
        Args:
            query: The search query
            top_k: Number of results to return
            score_threshold: Minimum similarity score (0-1)
            filter_metadata: Optional metadata filters
            include_metadata: Whether to include metadata in results
            
        Returns:
            List of matching documents with scores
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # Prepare filters if provided
            query_filter = None
            if filter_metadata:
                must_conditions = []
                for key, value in filter_metadata.items():
                    must_conditions.append(
                        FieldCondition(
                            key=f"metadata.{key}",
                            match=MatchValue(value=value)
                        )
                    )
                query_filter = Filter(must=must_conditions)
            
            # Debug output
            print(f"Debug: Searching with filter: {query_filter}")
            
            # Search in Qdrant
            search_result = self.qdrant.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=query_filter,
                limit=top_k,
                score_threshold=score_threshold,
                with_payload=include_metadata,
                with_vectors=False
            )
            
            # Format results
            results = []
            for hit in search_result:
                result = {
                    "id": hit.id,
                    "text": hit.payload.get("text", ""),
                    "score": hit.score
                }
                if include_metadata:
                    result["metadata"] = hit.payload.get("metadata", {})
                    result["timestamp"] = hit.payload.get("timestamp")
                results.append(result)
            
            print(f"Debug: Found {len(results)} results")
            return results
            
        except Exception as e:
            print(f"Error searching documents: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    async def retrieve_and_generate(
        self,
        query: str,
        llm_client: Any,
        system_prompt: str = "You are a helpful AI assistant.",
        top_k: int = 3,
        score_threshold: float = 0.5
    ) -> str:
        """
        Retrieve relevant context and generate a response.
        
        Args:
            query: The user's query
            llm_client: The LLM client to use for generation
            system_prompt: System prompt for the LLM
            top_k: Number of documents to retrieve
            score_threshold: Minimum similarity score for retrieval
            
        Returns:
            Generated response as a string
        """
        try:
            # Retrieve relevant documents
            context_docs = await self.search(
                query=query,
                top_k=top_k,
                score_threshold=score_threshold
            )
            
            if not context_docs:
                return "I couldn't find any relevant information to answer your question."
            
            # Format context
            context = "\n".join([
                f"--- Context {i+1} (Relevance: {doc['score']:.2f}) ---\n{doc['text']}"
                for i, doc in enumerate(context_docs)
            ])
            
            # Prepare the prompt
            prompt = f"""
            You are a helpful AI assistant. Use the following context to answer the question.
            If the context doesn't contain the answer, say so.
            
            Context:
            {context}
            
            Question: {query}
            
            Answer:
            """
            
            # Generate response using LLM
            response = await llm_client.chat.completions.create(
                model="gpt-4",  # or your preferred model
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error in retrieve_and_generate: {e}")
            return "I'm having trouble generating a response right now. Please try again later."
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete a document by ID."""
        try:
            self.qdrant.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(
                    points=[document_id]
                )
            )
            return True
        except Exception as e:
            print(f"Error deleting document: {e}")
            return False
    
    async def clear_collection(self) -> bool:
        """Clear all documents from the collection."""
        try:
            self.qdrant.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter()
                )
            )
            return True
        except Exception as e:
            print(f"Error clearing collection: {e}")
            return False
