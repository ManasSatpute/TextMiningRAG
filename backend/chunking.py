import json
import uuid
from typing import List, Dict
from Constants import CORPUS_PATH


# Method to create unique IDs
def generate_id():
    return str(uuid.uuid4())


# Main Method for Chunking in 3 parts: description, ingredients, steps
def chunk_recipe(recipe: Dict, max_steps_per_chunk: int = 4) -> List[Dict]:
    chunks = []

    name = recipe.get("name", "")
    cuisine = recipe.get("cuisine", "")
    subcuisine = recipe.get("subcuisine", "")
    description = recipe.get("description", "")
    ingredients = recipe.get("ingredients", [])
    steps = recipe.get("steps", [])
    url = recipe.get("url", "")

    # 1. Description Chunk
    desc_text = f"""Cuisine: {cuisine}
    Subcuisine: {subcuisine}
    Recipe: {name}

    Description:
    {description}
    """

    chunks.append({
        "id": generate_id(),
        "text": desc_text.strip(),
        "metadata": {
            "name": name,
            "cuisine": cuisine,
            "subcuisine": subcuisine,
            "chunk_type": "description",
            "url": url
        }
    })

    # 2. Ingredients Chunk
    ingredients_text = "\n".join([f"- {ing}" for ing in ingredients])

    ing_text = f"""Recipe: {name}

    Ingredients:
    {ingredients_text}
    """

    chunks.append({
        "id": generate_id(),
        "text": ing_text.strip(),
        "metadata": {
            "name": name,
            "cuisine": cuisine,
            "subcuisine": subcuisine,
            "chunk_type": "ingredients",
            "url": url
        }
    })

    # 3. Steps Chunk
    if steps:
        for i in range(0, len(steps), max_steps_per_chunk):
            step_group = steps[i:i + max_steps_per_chunk]

            step_text = "\n".join(
                [f"{idx+1}. {step}" for idx, step in enumerate(step_group, start=i)]
            )

            chunk_text = f"""Recipe: {name}

            Steps (Part {i // max_steps_per_chunk + 1}):
            {step_text}
            """

            chunks.append({
                "id": generate_id(),
                "text": chunk_text.strip(),
                "metadata": {
                    "name": name,
                    "cuisine": cuisine,
                    "subcuisine": subcuisine,
                    "chunk_type": "steps",
                    "url": url
                }
            })

    return chunks


# Method to load corpus and apply chunking to each recipe
def load_and_chunk_corpus(json_path: str) -> List[Dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_chunks = []

    for recipe in data:
        recipe_chunks = chunk_recipe(recipe)
        all_chunks.extend(recipe_chunks)

    return all_chunks


# Method to save chunks to a JSON file
if __name__ == "__main__":
    chunks = load_and_chunk_corpus(CORPUS_PATH)

    print(f"Total chunks created: {len(chunks)}\n")

    # print sample (TODO: remove this in production)
    for c in chunks[:3]:
        print("=" * 50)
        print(c["text"])
        print(c["metadata"])