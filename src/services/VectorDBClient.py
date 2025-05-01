from typing import List, Dict, Literal, Optional
from fastapi import HTTPException
from qdrant_client.http.models import PointStruct, PointIdsList, Filter, FieldCondition, MatchValue
from src.services.QdrantClient import QdrantModel
from src.services.EmbeddingModel import EmbeddingHelper


class VectorController:
    """Controller class for vector database operations."""
    
    def __init__(self):
        """Initialize the vector controller with models and helpers."""
        self.qdrant_model = QdrantModel()
        self.embedding_helper = EmbeddingHelper()
    
    def search_vector_db(self, 
                    query_text: str, 
                    top_k: int, 
                    threshold: Optional[float] = None, 
                    class_type: Optional[Literal['class-a', 'class-b']] = None, 
                    collection: Literal['all-MiniLM-L6-v2', 'all-MiniLM-L12-v2'] = 'all-MiniLM-L6-v2') -> List[Dict]:
        """Search for similar vectors in the vector database.
        """
        try:
            # Get embeddings of the input query
            query_embedding = self.embedding_helper.encode_text(query_text, collection)
            
            # Prepare filter if class_type is provided
            query_filter = None
            if class_type in ['class-a', 'class-b']:
                query_filter = Filter(
                    must=[FieldCondition(key='class', match=MatchValue(value=class_type))]
                )
            
            # Search in Qdrant
            results = self.qdrant_model.search(
                collection_name=collection,
                query_vector=query_embedding,
                limit=top_k,
                score_threshold=threshold,
                query_filter=query_filter
            )
            
            # Format the results
            formatted_results = [
                {
                    'id': int(point.id), 
                    'score': float(point.score), 
                    'class': point.payload['class']
                } 
                for point in results
            ]
                
            return formatted_results
            
        except Exception as e:
            raise HTTPException(status_code=500, detail='Failed to get similar records: ' + str(e))
    
    def insert_to_vector_db(self, 
                        text_id: int, 
                        text: str, 
                        class_type: Literal['class-a', 'class-b'], 
                        collection: Literal['all-MiniLM-L6-v2', 'all-MiniLM-L12-v2'] = 'all-MiniLM-L6-v2'):
        """Insert a new vector to the vector database.
        """
        try:
            # Get embeddings using the embedding helper
            embedding = self.embedding_helper.encode_text(text, collection)
            
            # Prepare point for Qdrant
            point = PointStruct(
                id=text_id,
                vector=embedding,
                payload={'class': class_type}
            )
            
            # Insert to Qdrant
            self.qdrant_model.upsert(
                collection_name=collection,
                points=[point]
            )
            
            # Get the count of vectors after upserting
            count_after = self.qdrant_model.get_collection_info(collection).points_count
            
            return {f'Upserting Done in {collection}: Count Now is {count_after} vectors.'}
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f'Failed to upsert to Qdrant vector DB: {str(e)}')
    
    def delete_from_vector_db(self, 
                            text_id: int, 
                            collection: Literal['all-MiniLM-L6-v2', 'all-MiniLM-L12-v2'] = 'all-MiniLM-L6-v2'):
        """Delete a vector from the vector database.
        """
        try:
            # Prepare points selector
            points_selector = PointIdsList(points=[text_id])
            
            # Delete from vector DB
            self.qdrant_model.delete(
                collection_name=collection,
                points_selector=points_selector
            )
            
            # Get the count of vectors after deletion
            count_after = self.qdrant_model.get_collection_info(collection).points_count
            
            return {f'Deleting Done in {collection}: Count Now is {count_after} vectors.'}
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f'Failed to delete from Qdrant vector DB: {str(e)}')