import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from collections import Counter

# Load your dataset
df = pd.read_csv("data/dataset.csv")

df = df[["track_genre", "track_name"]].dropna().drop_duplicates()

# Group by genre to simulate playlists
playlists = df.groupby("track_genre")["track_name"].apply(list).to_dict()

# Train Word2Vec on track names
track_sequences = list(playlists.values())
w2v = Word2Vec(sentences=track_sequences, vector_size=64, epochs=10, window=5, min_count=1, sg=1)
print("Unique tracks in Word2Vec training:", len(w2v.wv.index_to_key))
track_embeddings = {t: w2v.wv[t] for t in w2v.wv.index_to_key}

# Create semantic IDs via sparse coding
X_raw = np.array(list(track_embeddings.values()))
X = StandardScaler().fit_transform(X_raw)
track_ids = list(track_embeddings.keys())
print("Word2Vec embedding matrix shape:", X.shape)

dict_learner = MiniBatchDictionaryLearning(n_components=64, alpha=1.0, batch_size=512, n_iter=200)
Z = dict_learner.fit_transform(X)

def semantic_id(vec, top_k=3):
    top = np.argsort(-np.abs(vec))[:top_k]
    return ''.join([f"<{i}>" for i in top])

semantic_ids = {t: semantic_id(z) for t, z in zip(track_ids, Z)}
print("Top 10 most common semantic IDs:", Counter(semantic_ids.values()).most_common(10))
print("Unique semantic IDs:", len(set(semantic_ids.values())))
# Build prompt to semantic ID pairs
data = []
for genre, tracks in playlists.items():
    vecs = [track_embeddings[t] for t in tracks if t in track_embeddings]
    if not vecs:
        continue
    prompt_vec = np.mean(vecs, axis=0)
    sims = cosine_similarity([prompt_vec], X)[0]
    top_indices = sims.argsort()[-5:][::-1]

    seen = set()
    for i in top_indices:
        tname = track_ids[i]
        sid = semantic_ids[tname]
        if sid not in seen:
            seen.add(sid)
            data.append({"prompt": genre, "target": sid})

pd.DataFrame(data).to_csv("training_data.csv", index=False)
print("Saved training_data.csv with prompt-semantic_id pairs.")