import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from backend.retriever import retriever
from backend.Constants import llm_model_name


class Generator:
    """
    Generator class responsible for loading the language model and producing
    answers using retrieved context (RAG generation step).
    """

    def __init__(self, model_name=llm_model_name, cuisine="South Asian"):
        """
        Initialize tokenizer and model, and configure device.

        Args:
            model_name (str): HuggingFace model name (Qwen required)
            cuisine (str): Cuisine specialization for the assistant
        """
        self.cuisine = cuisine

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Ensure pad token is set (required for Qwen models)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Detect available device and set data type
        if torch.cuda.is_available():
            self.device = "cuda"
            dtype = torch.float16
        else:
            self.device = "cpu"
            dtype = torch.float32

        # Load model with safe configuration
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else None,
            offload_folder="offload" if self.device == "cuda" else None,
            offload_state_dict=True if self.device == "cuda" else False
        )

        # Explicitly move model to CPU if needed
        if self.device == "cpu":
            self.model.to(self.device)

    def build_prompt(self, query, chunks, max_chars=3000):
        """
        Construct the prompt given query and retrieved chunks.

        Limits total context length to avoid exceeding model input size.

        Args:
            query (str): User question
            chunks (list): Retrieved text chunks
            max_chars (int): Maximum allowed context size

        Returns:
            str: Formatted prompt
        """

        # Concatenate chunks while respecting length limit
        context = ""
        for chunk in chunks:
            if len(context) + len(chunk) < max_chars:
                context += chunk + "\n\n"
            else:
                break

        # Prompt enforces grounding and domain specialization
        prompt = f"""
You are an expert culinary assistant specialising in {self.cuisine} cuisine.

Answer the question using only the information provided in the context.
If the answer is not present, say: "I don't know based on the given information."

Context:
{context}

Question:
{query}

Provide a clear and concise answer.

Answer:
"""
        return prompt.strip()

    def generate(self, query, chunks, debug=False):
        """
        Generate an answer given a query and retrieved context.

        Args:
            query (str): User question
            chunks (list): Retrieved context chunks
            debug (bool): If True, prints prompt for inspection

        Returns:
            str: Generated answer
        """

        # Handle case where retrieval fails
        if not chunks:
            return "I don't have enough information to answer this question."

        # Build prompt
        prompt = self.build_prompt(query, chunks)

        # Optional debugging output
        if debug:
            print("\nPrompt:\n", prompt)

        # Tokenize input prompt
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        )

        # Move tensors to the correct device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate output without gradient computation
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,            # limit response length
                do_sample=False,               # deterministic output
                temperature=0.0,               # no randomness
                repetition_penalty=1.1,        # reduce repetition
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # Extract only generated tokens (exclude prompt)
        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

        # Decode tokens into text
        response = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        ).strip()

        return response


# Example usage
if __name__ == "__main__":
    generator = Generator(cuisine="South Asian")

    query = "How to make jalebi?"

    # Retrieve top chunks from retriever
    chunks = retriever.get_top_chunks(query)

    # Generate answer
    answer = generator.generate(query, chunks, debug=True)

    print("\nAnswer:\n")
    print(answer)
