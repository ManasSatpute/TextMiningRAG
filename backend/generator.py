import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .retriever import retriever


class Generator:
    def __init__(self, model_name = llm_model_name):
        print("🔹 Loading Qwen model...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # 🔥 Fix for Qwen (important)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Detect device
        if torch.cuda.is_available():
            self.device = "cuda"
            dtype = torch.float16
        else:
            self.device = "cpu"
            dtype = torch.float32

        print(f"🔹 Using device: {self.device}")

        # 🔥 SAFE model loading (no meta tensor issues)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            device_map="auto" if self.device == "cuda" else None,
            offload_folder="offload" if self.device == "cuda" else None,
            offload_state_dict=True if self.device == "cuda" else False
        )

        # If CPU, explicitly move model
        if self.device == "cpu":
            self.model.to(self.device)

        print("✅ Model loaded successfully!\n")

    def build_prompt(self, query, chunks):
        context = "\n\n".join(chunks)

        prompt = f"""You are a helpful AI assistant. Answer the question based only on the provided context.

        Context:
        {context}

        Question:
        {query}

        Answer:"""

        return prompt

    def generate(self, query, chunks):
        prompt = self.build_prompt(query, chunks)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        )

        # Move inputs to correct device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,   # 🔥 prevents hanging
                do_sample=False,      # deterministic, no warnings
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the answer part
        if "Answer:" in response:
            response = response.split("Answer:")[-1].strip()

        return response


# 🔹 Example usage
if __name__ == "__main__":
    generator = Generator()

    query = "How to make jalebi?"

    chunks = retriever.get_top_chunks(query)

    answer = generator.generate(query, chunks)

    print("\n🧠 Answer:\n")
    print(answer)