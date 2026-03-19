from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load vector DB
db = FAISS.load_local("faiss_index", embeddings)

retriever = db.as_retriever(search_kwargs={"k": 5})  # ✅ max 5 chunks

# Load Qwen model (REQUIRED)
model_name = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def generate_answer(query, context):
    prompt = f"""
You are a culinary assistant specialised in your cuisine.

Use ONLY the context below to answer.

Context:
{context}

Question:
{query}

Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=200)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def ask(query):
    docs = retriever.get_relevant_documents(query)
    context = "\n".join([d.page_content for d in docs])

    return generate_answer(query, context)