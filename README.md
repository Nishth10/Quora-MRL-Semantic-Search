# Quora Question Semantic Search with Matryoshka Representation Learning (MRL)

An interactive, high-performance Flask web application for semantic search over Quora questions using **Matryoshka Representation Learning (MRL)** and `sentence-transformers/all-MiniLM-L6-v2`.

---

## 📌 Project Overview
In modern vector search and Retrieval-Augmented Generation (RAG) systems, high-dimensional dense embeddings (e.g., 384d, 768d, 1536d) incur substantial RAM costs, network transfer latency, and slower vector similarity computation.

**Matryoshka Representation Learning (MRL)** trains embeddings such that the early dimensions (e.g., first 16, 32, 64, or 128 coordinates) pack the majority of semantic variance and information. This allows the system to dynamically **truncate embeddings to lower dimensions** at runtime:
- **No retraining needed** for lower dimensions.
- **Up to 95.8% reduction** in vector storage and RAM footprint.
- **Millisecond-level retrieval latency**.

This application directly showcases this mechanism interactively on real Quora questions.

---

## 🚀 Key Features

1. **Interactive Search Box**:
   - Enter any natural language question (or click quick test example prompts).
   - Retrieves the top-K semantically matching Quora questions.

2. **Dynamic MRL Dimension Selection**:
   - Choose between **384d** (Full baseline), **256d**, **128d**, **64d**, **32d**, and **16d**.
   - Displays real-time storage savings (e.g. 64d saves **83.3%** memory vs 384d).

3. **Precise Search Latency Measurement**:
   - High-precision timer (`time.perf_counter()`) tracks retrieval speed.
   - Outputs latency in **seconds** (e.g., `0.0082 s`) and **milliseconds** (`8.2 ms`).
   - Detailed breakdown: **Query Encoding**, **MRL Slicing & L2-Normalization**, and **Vector Similarity Search**.

4. **Multi-Dimension Comparison Mode**:
   - Click *"Compare Latency & Quality Across All Dimensions"* to benchmark the exact query across all 6 dimensions simultaneously.
   - Observes **rank retention** vs **rank drift** as dimension decreases.

---

## 🛠️ Mathematical Formulation of MRL Inference

Given a query embedding $\mathbf{e} \in \mathbb{R}^{384}$ produced by `all-MiniLM-L6-v2`:

1. **Truncation to Sub-Dimension $d \in \{16, 32, 64, 128, 256, 384\}$**:
   $$\mathbf{e}_{[:d]} = [e_1, e_2, \dots, e_d]$$

2. **L2 Normalization**:
   $$\mathbf{\hat{e}}^{(d)} = \frac{\mathbf{e}_{[:d]}}{\|\mathbf{e}_{[:d]}\|_2}$$

3. **Cosine Similarity (Dot Product on Normalized Vectors)**:
   $$\text{Sim}(\mathbf{q}^{(d)}, \mathbf{c}_i^{(d)}) = \mathbf{\hat{q}}^{(d)} \cdot \mathbf{\hat{c}}_i^{(d)}$$

---

## 📁 Project Structure

```
NLP_MRL_WebApp/
│
├── app.py                  # Main Flask application and MRL search engine
├── prepare_data.py         # Script to generate Quora corpus and pre-compute embeddings
├── requirements.txt        # Python package dependencies
├── README.md               # Complete project documentation
│
├── templates/
│   └── index.html          # Clean, modern single-page web UI
│
├── static/
│   ├── css/
│   │   └── style.css       # Responsive styling, modern cards, and badges
│   └── js/
│       └── main.js         # Frontend logic for API calls, latency timing, & UI updates
│
└── data/                   # (Generated on first run)
    ├── quora_corpus.json   # Quora question text & metadata
    └── quora_embeddings.npy# Precomputed 384-dimensional dense vectors
```

---

## 🏃 How to Run the Application

1. **Activate Virtual Environment**:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Flask App**:
   ```bash
   python app.py
   ```

4. **Open in Browser**:
   Navigate to `http://127.0.0.1:5000` in your web browser.
