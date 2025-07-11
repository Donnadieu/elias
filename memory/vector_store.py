"""
Cloud-based vector memory store using Qdrant Cloud.
"""
import os
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
import uuid

# Try to load environment variables if dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, using environment variables directly


class CloudMemoryStore:
    """
    A cloud-based memory store using Qdrant for vector storage and retrieval.
    Uses sentence-transformers for embedding generation.
    """
    
    def __init__(
        self, 
        collection_name: str = "assistant_memories",
        embedding_model: str = "all-MiniLM-L6-v2",
        api_key: Optional[str] = None,
        host_url: Optional[str] = None
    ):
        """
        Initialize the cloud memory store with Qdrant client and embedding model.
        
        Args:
            collection_name: Name of the Qdrant collection to use
            embedding_model: Name of the sentence-transformer model to use
            api_key: Qdrant API key (defaults to QDRANT_API_KEY env var)
            host_url: Qdrant host URL (defaults to QDRANT_HOST_URL env var)
        """
        # Get API key and host URL from parameters or environment variables
        self.api_key = api_key or os.getenv("QDRANT_API_KEY")
        self.host_url = host_url or os.getenv("QDRANT_HOST_URL")
        
        if not self.api_key or not self.host_url:
            raise ValueError(
                "Qdrant API key and host URL must be provided either as parameters "
                "or as QDRANT_API_KEY and QDRANT_HOST_URL environment variables."
            )
        
        # Initialize the embedding model
        self.model = SentenceTransformer(embedding_model)
        self.vector_size = self.model.get_sentence_embedding_dimension()
        
        # Initialize Qdrant client
        self.client = QdrantClient(
            url=self.host_url,
            api_key=self.api_key,
        )
        
        # Collection name
        self.collection_name = collection_name
        
        # Create collection if it doesn't exist
        self._ensure_collection_exists()
    
    def _ensure_collection_exists(self) -> None:
        """
        Check if the collection exists and create it if it doesn't.
        """
        try:
            # First try to get the collection info to see if it exists
            self.client.get_collection(self.collection_name)
            # If we get here, the collection exists
            return
        except Exception:
            # Collection doesn't exist, create it
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE,
                ),
            )
    
    def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for the given text.
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        return self.model.encode(text).tolist()
    
    def store(self, text: str, metadata: Dict[str, Any] = None) -> str:
        """
        Store a memory with its metadata.
        
        Args:
            text: The text content to store
            metadata: Additional metadata to store with the text
            
        Returns:
            ID of the stored memory
        """
        if metadata is None:
            metadata = {}
        
        # Generate a unique ID for this memory
        memory_id = str(uuid.uuid4())
        
        # Make sure the collection exists before trying to store
        self._ensure_collection_exists()
        
        try:
            # Generate embedding for the text
            embedding = self._generate_embedding(text)
            
            # Store the memory in Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=memory_id,
                        vector=embedding,
                        payload={
                            "text": text,
                            **metadata
                        }
                    )
                ]
            )
            
            return memory_id
        except Exception as e:
            # If there was an error, try recreating the collection and storing again
            print(f"Error storing memory, recreating collection: {str(e)}")
            self.client.delete_collection(self.collection_name)
            self._ensure_collection_exists()
            
            # Try storing again
            embedding = self._generate_embedding(text)
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=memory_id,
                        vector=embedding,
                        payload={
                            "text": text,
                            **metadata
                        }
                    )
                ]
            )
            return memory_id
    
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Search for memories semantically similar to the query.
        
        Args:
            query: The search query
            top_k: Number of results to return
            
        Returns:
            List of dictionaries containing memory text and metadata
        """
        try:
            # Make sure the collection exists before searching
            self._ensure_collection_exists()
            
            # Generate embedding for the query
            query_embedding = self._generate_embedding(query)
            
            # Search for similar memories
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k
            )
            
            # Format the results
            results = []
            for result in search_results:
                payload = result.payload
                if payload and "text" in payload:  # Make sure we have valid payload with text
                    results.append({
                        "text": payload.pop("text"),
                        "score": result.score,
                        "metadata": payload
                    })
            
            return results
            
        except Exception as e:
            print(f"Error searching memories: {str(e)}")
            # If there was an error, try recreating the collection
            try:
                self.client.delete_collection(self.collection_name)
                self._ensure_collection_exists()
            except Exception as e2:
                print(f"Error recreating collection: {str(e2)}")
            
            return []  # Return empty list on error
