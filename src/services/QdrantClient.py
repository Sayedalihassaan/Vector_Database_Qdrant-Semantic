from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    PointStruct, 
    PointIdsList, 
    Filter
)
from typing import List, Dict, Optional, Any
from src.config import QDRANT_API_KEY, QDRANT_URL

class QdrantModel:
    """Model class for Qdrant vector database operations."""
    
    def __init__(self):
        """Initialize the Qdrant client."""
        self.client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    
    def search(self, 
            collection_name: str, 
            query_vector: List[float], 
            limit: int, 
            score_threshold: Optional[float] = None, 
            query_filter: Optional[Filter] = None) -> List[Dict]:
        """Search the Qdrant collection.
        """
        if query_filter:
            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=query_filter
            )
        else:
            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold
            )
        
        return results
    
    def upsert(self, collection_name: str, points: List[PointStruct], wait: bool = True) -> Dict:
        """Upsert points to the Qdrant collection.
        """
        response = self.client.upsert(
            collection_name=collection_name,
            points=points,
            wait=wait
        )
        return response
    
    def delete(self, collection_name: str, points_selector: PointIdsList) -> Dict:
        """Delete points from the Qdrant collection.
        """
        response = self.client.delete(
            collection_name=collection_name,
            points_selector=points_selector
        )
        return response
    
    def get_collection_info(self, collection_name: str) -> Any:
        """Get information about a Qdrant collection.
        """
        return self.client.get_collection(collection_name=collection_name)