from typing import List, Dict, Literal, Optional
from fastapi import FastAPI
from src.services.VectorDBClient import VectorController

app = FastAPI(title="Qdrant Vector Database API", 
            description="API for Qdrant vector database operations")
vector_controller = VectorController()


@app.get("/search")
def search_vector_db(
    query_text: str,
    top_k: int,
    threshold: Optional[float] = None,
    class_type: Optional[Literal['class-a', 'class-b']] = None,
    collection: Literal['all-MiniLM-L6-v2', 'all-MiniLM-L12-v2'] = 'all-MiniLM-L6-v2'
) -> List[Dict]:
    """Search for similar vectors in the database.
    """
    return vector_controller.search_vector_db(
        query_text=query_text,
        top_k=top_k,
        threshold=threshold,
        class_type=class_type,
        collection=collection
    )

@app.post("/insert")
def insert_to_vector_db(
    text_id: int,
    text: str,
    class_type: Literal['class-a', 'class-b'],
    collection: Literal['all-MiniLM-L6-v2', 'all-MiniLM-L12-v2'] = 'all-MiniLM-L6-v2'
):
    """Insert a new vector to the database.
    """
    return vector_controller.insert_to_vector_db(
        text_id=text_id,
        text=text,
        class_type=class_type,
        collection=collection
    )

@app.delete("/delete")
def delete_from_vector_db(
    text_id: int,
    collection: Literal['all-MiniLM-L6-v2', 'all-MiniLM-L12-v2'] = 'all-MiniLM-L6-v2'
):
    """Delete a vector from the database.
    """
    return vector_controller.delete_from_vector_db(
        text_id=text_id,
        collection=collection
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)