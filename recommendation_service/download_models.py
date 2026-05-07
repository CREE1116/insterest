from sentence_transformers import SentenceTransformer
import os

models = [
    "paraphrase-multilingual-mpnet-base-v2",
    "clip-ViT-B-32"
]

print("🚀 Pre-downloading ML models...")
for model_name in models:
    print(f"  📥 Downloading {model_name}...")
    SentenceTransformer(model_name)
print("✅ All models downloaded.")
