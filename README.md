COMP64702 Transforming Text Into Meaning
Coursework: Retrieval-Augmented Generation (RAG) System

1. Overview
This project implements a Retrieval-Augmented Generation (RAG) based culinary question answering system.
The system is designed to answer user queries related to a specific cuisine by retrieving relevant contextual information from a curated corpus
and generating responses using a language model.
The system follows the standard RAG architecture consisting of three main pipelines:

1. Ingestion Pipeline
2. Inference Pipeline
3. Evaluation Pipeline
The implementation uses the Qwen2.5-0.5B-Instruct model for generation, as required by the coursework specification.

2. System Architecture
The system is divided into the following components:
1. Chunking
   Splits raw documents into smaller overlapping text segments to preserve context and improve retrieval quality.
2. Embedding (Vectorisation)
   Converts text chunks into dense vector representations using a sentence embedding model.
3. Vector Store
   Stores embeddings and corresponding text chunks to enable efficient similarity-based retrieval.
4. Retrieval and Ranking
   Given a query, retrieves the top-k most relevant chunks (k ≤ 5) based on similarity.
5. Prompting and Generation
   Constructs a prompt using retrieved context and generates an answer using the Qwen language model.
6. Evaluation
   Compares generated answers with gold-standard answers using selected evaluation metrics.

3. Pipeline Description

3.1 Ingestion Pipeline
This pipeline is executed offline to prepare the knowledge base.
Steps:
i.   Load documents from the background corpus
ii.  Clean and preprocess text
iii. Split text into overlapping chunks
iv.  Convert chunks into embeddings
v.   Store embeddings in a vector database

Output:
A vector store containing chunk embeddings and corresponding text.

3.2 Inference Pipeline
This pipeline runs in real time to answer user queries.
Steps:
i.   Accept input query (from JSON file)
ii.  Convert query into embedding
iii. Retrieve top 5 most relevant chunks from vector store
iv.  Construct prompt using retrieved context
v.   Generate answer using Qwen model
vi.  Save output in required JSON format

3.3 Evaluation Pipeline
This pipeline evaluates system performance.
Steps:
i.   Load generated answers and gold-standard answers
ii.  Compute evaluation metrics such as Exact Match and F1 score
iii. Output performance results

4. File Structure

The project is organised as follows:

* ingestion.py        : Handles document processing and embedding creation
* retriever.py        : Performs similarity search and returns top chunks
* generator.py        : Builds prompts and generates answers using the LLM
* evaluation.py       : Computes evaluation metrics
* main.py             : Runs the inference pipeline
* constants.py        : Stores model names and configuration parameters
* data/               : Contains corpus and dataset files
* output/             : Stores generated outputs

5. Requirements

Python version: 3.9 or above
Required libraries:
* torch
* transformers
* sentence-transformers
* faiss-cpu (or faiss-gpu if CUDA is available)
* numpy
* scikit-learn

Install dependencies using:
pip install -r requirements.txt

6. How to Run the System

6.1 Step 1: Run Ingestion Pipeline

This step prepares the vector store.

Command:
python ingestion.py

6.2 Step 2: Run Inference Pipeline

This step generates answers for input queries.

Command:
python main.py --input input_queries.json --output output_answers.json

Input format:
JSON file containing queries

Output format:
JSON file containing generated answers

6.3 Step 3: Run Evaluation Pipeline
This step evaluates the generated answers.

Command:
python evaluation.py --pred output_answers.json --gold gold_answers.json

7. Configuration

The following parameters can be modified in constants.py:

* Model name (Qwen2.5-0.5B-Instruct)
* Chunk size and overlap
* Number of retrieved chunks (max 5)
* Maximum token length for generation

8. Design Choices

* Overlapping chunking is used to preserve context across boundaries
* Dense embeddings are used for semantic retrieval
* Top-k retrieval is limited to 5 as per coursework requirement
* Prompt engineering is applied to ensure responses are grounded in context
* Deterministic decoding is used for stable and reproducible outputs

9. Notes

* The system is designed to run on both CPU and GPU
* GPU is recommended for faster generation
* Ensure sufficient memory is available when loading the model

10. Conclusion
This project demonstrates a complete implementation of a RAG-based question answering system, integrating retrieval and generation to
produce context-aware and accurate responses within a specialised culinary domain.

End.
