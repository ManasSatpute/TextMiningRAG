from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from backend.Constants import transformer_model_name

model = SentenceTransformer(transformer_model_name)

def evaluate(predictions, references):
    scores = []

    for pred, ref in zip(predictions, references):
        emb1 = model.encode([pred])
        emb2 = model.encode([ref])

        sim = cosine_similarity(emb1, emb2)[0][0]
        scores.append(sim)

    avg_score = sum(scores) / len(scores)

    return {
        "average_similarity": avg_score,
        "individual_scores": scores
    }


# Example usage
if __name__ == "__main__":
    preds = ["Sushi is rice with fish"]
    refs = ["Sushi is a Japanese dish with vinegared rice"]

    print(evaluate(preds, refs))