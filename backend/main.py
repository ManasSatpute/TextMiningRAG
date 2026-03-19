from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from backend.retriever import ask

# -----------------------------
# Create FastAPI app
# -----------------------------
app = FastAPI()

# -----------------------------
# Enable CORS (for React UI)
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Root endpoint (optional)
# -----------------------------
@app.get("/")
def home():
    return {"message": "RAG API is running 🚀"}


# -----------------------------
# Chat endpoint (UI)
# -----------------------------
class Query(BaseModel):
    question: str


@app.post("/chat")
def chat(q: Query):
    return ask(q.question)


# -----------------------------
# Batch endpoint (COURSEWORK)
# -----------------------------
class BatchQuery(BaseModel):
    queries: list


@app.post("/batch")
def batch_inference(data: BatchQuery):
    results = []

    for q in data.queries:
        result = ask(q["question"])

        results.append({
            "id": q["id"],
            "answer": result["answer"]
        })

    return {"results": results}