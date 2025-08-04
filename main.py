from fastapi import FastAPI, HTTPException, UploadFile, File, status
from preProces import Lemmatizer, NoiseRemover, SpellCorrector, StopwordRemover, TextNormalizer, TextPipeline
from vectorizers import VectorizationPipeline, TFIDFVectorizer  # o cualquier otro
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import traceback


app = FastAPI()

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

@app.post("/process_and_vectorize_text")
async def process_and_vectorize_text(data: TextData):
    try:
        text = data.text
        if not text or not isinstance(text, str):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="El campo 'text' debe ser una cadena de texto no vacía."
            )
        import time
        start = time.time()
        processed_text = text_pipeline.process(text)
        vectors = vector_pipeline.fit_transform([processed_text])
        print(f"⏱️ Procesamiento tomó {time.time() - start:.2f} segundos")
        return {
            "vector": vectors[0].tolist()
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ocurrió un error al procesar el texto: {str(e)}"
        )