import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from Constants import EMBEDDING_MODEL_NAME


# -----------------------------
# Exact Match
# -----------------------------
def exact_match(pred, gold):
    return int(pred.strip().lower() == gold.strip().lower())


# -----------------------------
# Token-level F1
# -----------------------------
def f1_score(pred, gold):
    pred_tokens = pred.lower().split()
    gold_tokens = gold.lower().split()

    common = set(pred_tokens) & set(gold_tokens)

    if len(common) == 0:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)

    return 2 * (precision * recall) / (precision + recall)


# -----------------------------
# Semantic Similarity (bonus)
# -----------------------------
model = SentenceTransformer(EMBEDDING_MODEL_NAME)

def semantic_similarity(pred, gold):
    emb1 = model.encode([pred])
    emb2 = model.encode([gold])
    return cosine_similarity(emb1, emb2)[0][0]


# -----------------------------
# Main Evaluation Function
# -----------------------------
def evaluate(pred_file, gold_file):
    with open(pred_file, "r", encoding="utf-8") as f:
        preds = json.load(f)["answers"]

    with open(gold_file, "r", encoding="utf-8") as f:
        golds = json.load(f)["answers"]

    # Map gold answers by query_id
    gold_map = {g["query_id"]: g["gold_answer"] for g in golds}

    em_scores = []
    f1_scores = []
    sim_scores = []

    for p in preds:
        qid = p["query_id"]
        pred = p["answer"]
        gold = gold_map[qid]

        em_scores.append(exact_match(pred, gold))
        f1_scores.append(f1_score(pred, gold))
        sim_scores.append(semantic_similarity(pred, gold))

    results = {
        "Exact Match": sum(em_scores) / len(em_scores),
        "F1 Score": sum(f1_scores) / len(f1_scores),
        "Semantic Similarity": sum(sim_scores) / len(sim_scores)
    }

    return results


# -----------------------------
# CLI entry
# -----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True)
    parser.add_argument("--gold", required=True)

    args = parser.parse_args()

    results = evaluate(args.pred, args.gold)

    print("\n📊 Evaluation Results:")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")