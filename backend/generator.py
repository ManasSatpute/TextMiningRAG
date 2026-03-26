import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from Constants import LLM_MODEL_NAME


class Generator:
    """
    Generator class responsible for producing answers using retrieved context.
    """

    def __init__(self, model_name=LLM_MODEL_NAME, cuisine="South Asian"):
        self.cuisine = cuisine

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Device setup
        if torch.cuda.is_available():
            self.device = "cuda"
            dtype = torch.float16
        else:
            self.device = "cpu"
            dtype = torch.float32

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else None
        )

        if self.device == "cpu":
            self.model.to(self.device)

    def build_prompt(self, query, chunks, max_chars=3000):
        """
        Build dataset-aware prompt.
        """

        context = ""
        for chunk in chunks:
            text = chunk["text"] if isinstance(chunk, dict) else chunk
            if len(context) + len(text) < max_chars:
                context += text + "\n\n"
            else:
                break

        prompt = f"""
You are a culinary QA assistant specialising in {self.cuisine} cuisine.

Answer the question STRICTLY using the provided context.

Rules:
- Keep the answer SHORT (prefer 1 sentence)
- Do NOT add extra explanation
- If asking ingredients → list them clearly
- If asking "What is" → give a definition
- If asking "How do you start" → give ONLY the first step
- If asking "How is ... served" → give ONLY the final step
- If answer is not in context → say:
  "I don't know based on the given information."

Context:
{context}

Question:
{query}

Answer:
"""
        return prompt.strip()

    def generate(self, query, chunks, debug=False):
        """
        Generate answer for a query.
        """

        if not chunks:
            return "I don't know based on the given information."

        prompt = self.build_prompt(query, chunks)

        if debug:
            print("\nPROMPT:\n", prompt)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        )

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                temperature=0.0,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

        response = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        ).strip()

        return response


# -----------------------------
# Batch inference (COURSEWORK REQUIRED)
# -----------------------------
def run_inference(input_file, output_file, retriever, generator):
    import json

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []

    for item in data["queries"]:
        query_id = item["query_id"]
        query = item["query"]

        chunks = retriever.get_top_chunks(query)
        answer = generator.generate(query, chunks)

        results.append({
            "query_id": query_id,
            "answer": answer
        })

        print(f"Processed query {query_id}")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"answers": results}, f, indent=2, ensure_ascii=False)

    print("✅ Inference completed successfully!")


# -----------------------------
# CLI entry (important for demo)
# -----------------------------
if __name__ == "__main__":
    import argparse
    from retriever import Retriever

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    retriever = Retriever()
    generator = Generator()

    run_inference(args.input, args.output, retriever, generator)