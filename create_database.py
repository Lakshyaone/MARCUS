import os
import pickle
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm

DATA_FOLDER = "data"
if not os.path.exists(DATA_FOLDER):
    raise ValueError(f"{DATA_FOLDER} folder not found!")

model = SentenceTransformer('all-MiniLM-L6-v2')


documents = []
file_names = []
for file in os.listdir(DATA_FOLDER):
    if file.endswith(".md"):
        with open(os.path.join(DATA_FOLDER, file), "r", encoding="utf-8") as f:
            text = f.read().strip()
            if text:
                documents.append(text)
                file_names.append(file)

def chunk_text(text, chunk_size=300):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

chunks = []
sources = []
for i, doc in enumerate(documents):
    for chunk in chunk_text(doc):
        chunks.append(chunk)
        sources.append(file_names[i])

print(f"Total chunks created: {len(chunks)}")


embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)


dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)


faiss.write_index(index, "index.faiss")


with open("index.pkl", "wb") as f:
    pickle.dump({"chunks": chunks, "sources": sources}, f)

print("✅ FAISS index and metadata saved successfully!")
