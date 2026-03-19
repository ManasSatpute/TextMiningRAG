from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


# -----------------------------
# Config
# -----------------------------
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


# -----------------------------
# Generator class
# -----------------------------
class Generator:
    def __init__(self):
        print("🔹 Loading Qwen model...")

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
            device_map="auto"
        )

    # -------------------------
    # Build prompt
    # -------------------------
    def build_prompt(self, query: str, chunks: List[Dict]) -> str:
        context = "\n\n".join(
            [f"[Chunk {i+1}]\n{c['text']}" for i, c in enumerate(chunks)]
        )

        prompt = f"""
You are a helpful cooking assistant.

Use ONLY the provided context to answer the question.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{query}

Instructions:
- Be concise
- Use bullet points for ingredients
- Use numbered steps for cooking instructions

Answer:
"""
        return prompt.strip()

    # -------------------------
    # Generate answer
    # -------------------------
    def generate(self, query: str, chunks: List[Dict], max_tokens: int = 256):
        prompt = self.build_prompt(query, chunks)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            temperature=0.0
        )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # remove prompt from output
        answer = response[len(prompt):].strip()

        return answer


# -----------------------------
# Debug
# -----------------------------
if __name__ == "__main__":
    from retriever import Retriever

    retriever = Retriever()
    generator = Generator()

    query = "How do I cook carbonara?"

    chunks = retriever.get_top_chunks(query)

    answer = generator.generate(query, chunks)

    print("\n🧠 Answer:\n")
    print(answer)