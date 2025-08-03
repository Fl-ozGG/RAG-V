from fastapi import FastAPI, UploadFile, File
from preProces import Lemmatizer, SpellCorrector, StopwordRemover, TextNormalizer, TextPipeline

app = FastAPI()

# Crear el pipeline una sola vez
pipeline = TextPipeline([
    TextNormalizer(),
    SpellCorrector(),
    StopwordRemover(),
    Lemmatizer(),
])

@app.post("/process_markdown")
async def process_markdown(file: UploadFile = File(...)):
    if not file.filename.endswith(".md"):
        return {"error": "Solo se permiten archivos Markdown (.md)"}
    
    contents = await file.read()
    text = contents.decode("utf-8")

    processed_text = pipeline.process(text)
    return {
        "original": text,
        "processed": processed_text
    }
