from fastapi import FastAPI, HTTPException, UploadFile, File, status
from preProces import Lemmatizer, NoiseRemover, SpellCorrector, StopwordRemover, TextNormalizer, TextPipeline
from vectorizers import VectorizationPipeline, TFIDFVectorizer  # o cualquier otro
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import traceback
import uuid
from pinecone import Pinecone
from dotenv import load_dotenv
import os
app = FastAPI()
load_dotenv()
PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
class TextData(BaseModel):
    text: str

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
async def process_and_vectorize_text(data: TextData):
    try:
        text = data.text
        if not text or not isinstance(text, str):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="El campo 'text' debe ser una cadena de texto no vacía."
            )

        # Procesa el texto (pipeline local)
        processed_text = text_pipeline.process(text)

        # Prepara el registro para upsert
        record_id = str(uuid.uuid4())
        record = {
            "_id": record_id,
            "chunk_text": processed_text,
            "original_text": text,
            "category": "custom"  # Puedes añadir categoría si quieres
        }

        # Upsert usando el método integrado: Pinecone hace el embedding automático
        index.upsert_records(namespace=NAMESPACE, records=[record])

        return {
            "id": record_id,
            "processed_text": processed_text,
            "message": "Texto procesado y almacenado en Pinecone."
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ocurrió un error al procesar el texto: {str(e)}"
        )