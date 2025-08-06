from fastapi import FastAPI, HTTPException, UploadFile, File, status
from preProces import Lemmatizer, NoiseRemover, SpellCorrector, StopwordRemover, TextNormalizer, TextPipeline
from vectorizers import VectorizationPipeline, TFIDFVectorizer  # o cualquier otro
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import traceback
import uuid
from pinecone import Pinecone
from dotenv import load_dotenv
import os
app = FastAPI()
load_dotenv()
PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
class TextDataWithIndex(BaseModel):
    text: str
    index_name: str = Field(..., description="Nombre del índice Pinecone a usar")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # o lista de dominios permitidos
    allow_methods=["*"],
    allow_headers=["*"],
)

text_pipeline = TextPipeline([ TextNormalizer(), StopwordRemover(), Lemmatizer()])

vector_pipeline = VectorizationPipeline(TFIDFVectorizer())

@app.get("/ping")
def ping():
    return {"message": "pong from Python API"}

pc = Pinecone(api_key=PINECONE_API_KEY)

INDEX_NAME = "developer-quickstart-py"
NAMESPACE = "ns1"

# Crear índice con modelo integrado si no existe
if not pc.has_index(INDEX_NAME):
    pc.create_index_for_model(
        name=INDEX_NAME,
        cloud="aws",
        region="us-east-1",
        embed={
            "model": "llama-text-embed-v2",
            "field_map": {"text": "chunk_text"}
        }
    )

index = pc.Index(INDEX_NAME)

@app.get("/ping")
def ping():
    return {"message": "pong from Python API"}

@app.post("/process_and_vectorize_text")
async def process_and_vectorize_text(data: TextDataWithIndex):
    try:
        text = data.text
        index_name = data.index_name.strip().lower()

        if not text or not isinstance(text, str):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="El campo 'text' debe ser una cadena de texto no vacía."
            )
        if not index_name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="El campo 'index_name' es obligatorio."
            )
        if not pc.has_index(index_name):
            pc.create_index_for_model(
                name=index_name,
                cloud="aws",
                region="us-east-1",
                embed={
                    "model": "llama-text-embed-v2",
                    "field_map": {"text": "chunk_text"}
                }
            )
        index = pc.Index(index_name)

        processed_text = text_pipeline.process(text)

        record_id = str(uuid.uuid4())
        record = {
            "_id": record_id,
            "chunk_text": processed_text,
            "original_text": text,
            "category": "custom"
        }

        index.upsert_records(namespace=NAMESPACE, records=[record])

        return {
            "message": "index_name is the way you have to make questions to database so dont forget it.",
            "index_name": index_name,
            "id": record_id,
            "processed_text": processed_text,
            "message": f"Texto procesado y almacenado en índice '{index_name}'."
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ocurrió un error al procesar el texto: {str(e)}"
        )
    
class QueryData(BaseModel):
    query: str
    index_name: str = Field(..., description="Nombre del índice Pinecone a usar")
    top_k: int = Field(5, description="Número de resultados a devolver")

import json

@app.post("/search_vector_db")
async def search_vector_db(data: QueryData):
    try:
        query = data.query
        index_name = data.index_name.strip().lower()
        top_k = data.top_k

        if not query or not isinstance(query, str):
            raise HTTPException(status_code=400, detail="El campo 'query' debe ser cadena no vacía.")
        if not index_name:
            raise HTTPException(status_code=400, detail="El campo 'index_name' es obligatorio.")
        if not pc.has_index(index_name):
            raise HTTPException(status_code=404, detail=f"Índice '{index_name}' no existe.")

        index = pc.Index(index_name)

        results = index.search(
            namespace=NAMESPACE,
            query={"top_k": top_k, "inputs": {"text": query}}
        )

        hits = results['result']['hits'] if 'result' in results and 'hits' in results['result'] else []

        # Convertir cada Hit a dict plano:
        hits_clean = []
        for hit in hits:
            hit_dict = {
                "id": hit.get("_id"),
                "score": hit.get("_score"),
                "fields": hit.get("fields")
            }
            hits_clean.append(hit_dict)

        return {
            "query": query,
            "index_name": index_name,
            "results": hits_clean
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Ocurrió un error al realizar la búsqueda: {str(e)}"
        )
