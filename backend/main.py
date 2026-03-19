from fastapi import FastAPI
from pydantic import BaseModel
from rag import ask

app = FastAPI()

# ---------- CHAT ----------
class Query(BaseModel):
    question: str

@app.post("/chat")
def chat(q: Query):
    answer = ask(q.question)
    return {"answer": answer}


# ---------- BATCH (IMPORTANT FOR COURSEWORK) ----------
class BatchQuery(BaseModel):
    queries: list

@app.post("/batch")
def batch_inference(data: BatchQuery):
    results = []

    for q in data.queries:
        answer = ask(q["question"])
        results.append({
            "id": q["id"],
            "answer": answer
        })

    return {"results": results}