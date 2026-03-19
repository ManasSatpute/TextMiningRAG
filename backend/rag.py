from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# -----------------------------
# Embeddings
# -----------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------------
# Load FAISS index
# -----------------------------
db = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# -----------------------------
# Retriever (max 5 chunks)
# -----------------------------
retriever = db.as_retriever(search_kwargs={"k": 5})

# -----------------------------
# Load Qwen model (COURSEWORK REQUIRED)
# -----------------------------
model_name = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

device = torch.device("cpu")
model.to(device)


# -----------------------------
# Generate answer
# -----------------------------
def generate_answer(query, context):
    prompt = f"""
You are a culinary assistant specialised in your cuisine.

Use ONLY the provided context to answer.

Context:
{context}

Question:
{query}

Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# -----------------------------
# Main RAG function
# -----------------------------
def ask(query):
    docs = retriever.get_relevant_documents(query)

    context = "\n\n".join([d.page_content for d in docs])

    answer = generate_answer(query, context)

    return {
        "answer": answer,
        "sources": [d.page_content[:150] for d in docs]
    }