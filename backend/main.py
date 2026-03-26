import argparse
import os

from backend.indexing import run_ingestion
from backend.generator import Generator, run_inference
from backend.retriever import Retriever
from backend.evaluation import evaluate


def main():
    parser = argparse.ArgumentParser(description="Run full RAG pipeline")

    # Input / Output files
    parser.add_argument("--input", required=True, help="Input queries JSON")
    parser.add_argument("--output", required=True, help="Output predictions JSON")
    parser.add_argument("--gold", required=False, help="Gold answers JSON")

    # Optional flags
    parser.add_argument("--skip_ingestion", action="store_true", help="Skip ingestion if index already exists")

    args = parser.parse_args()

    # -----------------------------
    # Step 1: Ingestion
    # -----------------------------
    if not args.skip_ingestion:
        print("\n🚀 Running ingestion pipeline...")
        run_ingestion()
    else:
        print("\n⏭️ Skipping ingestion (using existing index)")

    # -----------------------------
    # Step 2: Inference
    # -----------------------------
    print("\n🤖 Running inference pipeline...")

    retriever = Retriever()
    generator = Generator()

    run_inference(
        input_file=args.input,
        output_file=args.output,
        retriever=retriever,
        generator=generator
    )

    # -----------------------------
    # Step 3: Evaluation (optional)
    # -----------------------------
    if args.gold:
        print("\n📊 Running evaluation pipeline...")

        results = evaluate(args.output, args.gold)

        print("\n📊 Final Results:")
        for k, v in results.items():
            print(f"{k}: {v:.4f}")

    else:
        print("\n⚠️ No gold file provided — skipping evaluation")

    print("\n✅ Pipeline completed successfully!")


# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    main()