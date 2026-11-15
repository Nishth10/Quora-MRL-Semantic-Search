"""
Flask Web Application for Quora Semantic Search with Matryoshka Representation Learning (MRL).
Enables interactive querying, dimension selection (384d, 256d, 128d, 64d, 32d, 16d),
and precise real-time latency measurement.
"""

import os
import time
import json
import numpy as np

# Constants
SUPPORTED_DIMS = [384, 256, 128, 64, 32, 16]
DEFAULT_DIM = 64
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CORPUS_FILE = os.path.join(DATA_DIR, "quora_corpus.json")
EMB_FILE = os.path.join(DATA_DIR, "quora_embeddings.npy")

# Download and Cache the model
hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
hf_hub_cache = os.environ.get("HUGGINGFACE_HUB_CACHE", os.path.join(hf_home, "hub"))
model_dir_name = f"models--{MODEL_NAME.replace('/', '--')}"

if os.path.exists(os.path.join(hf_hub_cache, model_dir_name)):
    print("[Setup] Cached model found. Offline mode ENABLED.")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
else:
    print("[Setup] Model not found locally. Offline mode DISABLED for initial download.")

from flask import Flask, render_template, request, jsonify

app = Flask(__name__)


# Global state
model = None
corpus_metadata = []
corpus_embeddings_raw = None
corpus_normalized_cache = {}

def truncate_and_norm(emb, dim):
    """Slices first `dim` features and applies L2 normalization."""
    sliced = emb[:, :dim].copy()
    norms = np.linalg.norm(sliced, axis=1, keepdims=True)
    return sliced / np.clip(norms, 1e-12, None)

def get_normalized_corpus(dim):
    """Returns cached L2-normalized corpus embeddings sliced to the specified dimension."""
    global corpus_normalized_cache, corpus_embeddings_raw
    if dim not in corpus_normalized_cache:
        corpus_normalized_cache[dim] = truncate_and_norm(corpus_embeddings_raw, dim)
    return corpus_normalized_cache[dim]

def initialize_engine():
    """Initializes the SentenceTransformer model and loads the pre-computed corpus."""
    global model, corpus_metadata, corpus_embeddings_raw

    print(f"[Init] Loading SentenceTransformer: {MODEL_NAME}...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME)
    print("[Init] Model loaded successfully.")

    if not os.path.exists(CORPUS_FILE) or not os.path.exists(EMB_FILE):
        print("[Init] Corpus data files not found. Generating default seed corpus...")
        from prepare_data import main as prep_data
        prep_data()

    print(f"[Init] Loading corpus metadata from {CORPUS_FILE}...")
    with open(CORPUS_FILE, "r", encoding="utf-8") as f:
        corpus_metadata = json.load(f)

    print(f"[Init] Loading precomputed embeddings from {EMB_FILE}...")
    corpus_embeddings_raw = np.load(EMB_FILE)
    print(f"[Init] Loaded {len(corpus_metadata)} items with shape {corpus_embeddings_raw.shape}.")

    # Pre-warm cache for all supported dimensions
    print("[Init] Pre-caching L2-normalized vectors for MRL dimensions...")
    for d in SUPPORTED_DIMS:
        get_normalized_corpus(d)
    print("[Init] Engine ready for millisecond semantic retrieval!")

@app.route("/")
def index():
    return render_template(
        "index.html",
        supported_dims=SUPPORTED_DIMS,
        default_dim=DEFAULT_DIM,
        model_name=MODEL_NAME,
        corpus_size=len(corpus_metadata)
    )

@app.route("/api/search", methods=["POST"])
def search():
    """Performs semantic search with selected MRL dimension and latency tracking."""
    data = request.get_json(force=True) or {}
    query = data.get("query", "").strip()
    dim = int(data.get("dim", DEFAULT_DIM))
    top_k = min(int(data.get("top_k", 10)), 50)

    if not query:
        return jsonify({"error": "Query cannot be empty"}), 400

    if dim not in SUPPORTED_DIMS:
        return jsonify({"error": f"Invalid dimension {dim}. Supported: {SUPPORTED_DIMS}"}), 400

    # Start high-precision timer
    t_start = time.perf_counter()

    # 1. Encode query with SentenceTransformer (384-d raw)
    t_enc_start = time.perf_counter()
    q_emb = model.encode([query], normalize_embeddings=False, convert_to_numpy=True)
    t_enc = (time.perf_counter() - t_enc_start) * 1000

    # 2. MRL Slicing & L2-Normalization
    t_slice_start = time.perf_counter()
    q_sliced = truncate_and_norm(q_emb, dim)
    corpus_sliced = get_normalized_corpus(dim)
    t_slice = (time.perf_counter() - t_slice_start) * 1000

    # 3. Vector Similarity Search (Dot Product on normalized vectors = Cosine Similarity)
    t_sim_start = time.perf_counter()
    scores = np.dot(corpus_sliced, q_sliced.T).squeeze()
    
    # 4. Top-K Selection
    top_indices = np.argsort(-scores)[:top_k]
    t_sim = (time.perf_counter() - t_sim_start) * 1000

    # Total elapsed time
    t_total = time.perf_counter() - t_start

    results = []
    for rank, idx in enumerate(top_indices, start=1):
        score_val = float(scores[idx])
        item = corpus_metadata[idx]
        results.append({
            "rank": rank,
            "id": item.get("id"),
            "question": item.get("question"),
            "category": item.get("category", "General"),
            "score": round(score_val, 4),
            "percentage": max(0.0, min(100.0, round(score_val * 100, 1)))
        })

    # Memory and savings calculation
    bytes_per_vec = dim * 4 # FP32 = 4 bytes
    full_bytes = 384 * 4
    savings_pct = round(((full_bytes - bytes_per_vec) / full_bytes) * 100, 1)

    return jsonify({
        "query": query,
        "dim": dim,
        "latency_seconds": round(t_total, 4),
        "latency_ms": round(t_total * 1000, 2),
        "breakdown_ms": {
            "query_encoding": round(t_enc, 2),
            "mrl_slicing": round(t_slice, 2),
            "vector_search": round(t_sim, 2)
        },
        "compression": {
            "bytes_per_vector": bytes_per_vec,
            "savings_percent": savings_pct,
            "compression_factor": f"{384 / dim:.1f}x"
        },
        "results": results
    })

@app.route("/api/compare", methods=["POST"])
def compare():
    """Runs the same query across ALL MRL dimensions to demonstrate the speed/accuracy tradeoff."""
    data = request.get_json(force=True) or {}
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"error": "Query cannot be empty"}), 400

    # Encode query once
    q_emb = model.encode([query], normalize_embeddings=False, convert_to_numpy=True)

    comparisons = []
    baseline_top1_idx = None

    for dim in SUPPORTED_DIMS:
        t0 = time.perf_counter()
        q_sliced = truncate_and_norm(q_emb, dim)
        corpus_sliced = get_normalized_corpus(dim)
        scores = np.dot(corpus_sliced, q_sliced.T).squeeze()
        top_indices = np.argsort(-scores)[:5]
        latency = (time.perf_counter() - t0) * 1000

        top1_idx = top_indices[0]
        if dim == 384:
            baseline_top1_idx = top1_idx

        bytes_per_vec = dim * 4
        comparisons.append({
            "dim": dim,
            "latency_ms": round(float(latency), 2),
            "bytes_per_vec": int(bytes_per_vec),
            "savings_pct": round(float(((384 * 4 - bytes_per_vec) / (384 * 4)) * 100), 1),
            "top1_match": corpus_metadata[top1_idx]["question"],
            "top1_score": round(float(scores[top1_idx]), 4),
            "top1_matches_384d": bool(int(top1_idx) == int(baseline_top1_idx))
        })

    return jsonify({
        "query": query,
        "comparisons": comparisons
    })

@app.route("/api/stats", methods=["GET"])
def stats():
    """Returns backend configuration and index stats."""
    return jsonify({
        "model_name": MODEL_NAME,
        "corpus_size": len(corpus_metadata),
        "supported_dims": SUPPORTED_DIMS,
        "default_dim": DEFAULT_DIM,
        "full_dimension": 384
    })

if __name__ == "__main__":
    initialize_engine()
    print("Starting Flask WebApp on http://127.0.0.1:5000 ...")
    app.run(host="127.0.0.1", port=5000, debug=False)
