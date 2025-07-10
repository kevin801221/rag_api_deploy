from fastapi import FastAPI
from routes import inference, update

app = FastAPI(
    title="RAPTOR RAG System API",
    description="A FastAPI application for a RAPTOR-based RAG system, supporting knowledge base updates and Dify-compatible inference.",
    version="1.0.0",
)

# Include API routers
app.include_router(inference.router)
app.include_router(update.router)

@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "Welcome to the RAPTOR RAG System API! Visit /docs for API documentation."}
