"""
test_app.py
Unit and integration tests for the Flask MRL search application.
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import json
from app import app, initialize_engine

def test_engine():
    print("--- Initializing Engine ---")
    initialize_engine()

    with app.test_client() as client:
        # 1. Test Home Route
        print("\n[Test 1] Testing Home Page GET / ...")
        res = client.get("/")
        assert res.status_code == 200, f"Expected 200, got {res.status_code}"
        assert b"Quora Question Semantic Search" in res.data
        print("  ✓ Home page rendered successfully.")

        # 2. Test Stats Route
        print("\n[Test 2] Testing Stats GET /api/stats ...")
        res = client.get("/api/stats")
        assert res.status_code == 200
        stats = res.get_json()
        print(f"  ✓ Stats: {stats}")

        # 3. Test Search at different MRL dimensions
        test_query = "How can I improve my English speaking fluency?"
        dims = [384, 256, 128, 64, 32, 16]
        print(f"\n[Test 3] Testing Search for query: '{test_query}' across dimensions...")

        for dim in dims:
            payload = {"query": test_query, "dim": dim, "top_k": 5}
            res = client.post("/api/search", json=payload)
            assert res.status_code == 200, f"Failed at dim {dim}: {res.data}"
            data = res.get_json()
            latency_s = data["latency_seconds"]
            latency_ms = data["latency_ms"]
            top1 = data["results"][0]
            print(f"  ✓ Dim {dim:3d}d: Latency = {latency_s:.4f}s ({latency_ms:5.1f}ms) | Top-1: '{top1['question'][:45]}...' (Score: {top1['score']})")

        # 4. Test Multi-Dimension Compare
        print(f"\n[Test 4] Testing Multi-Dimension Comparison /api/compare ...")
        res = client.post("/api/compare", json={"query": test_query})
        assert res.status_code == 200
        comp_data = res.get_json()
        print(f"  ✓ Returned comparison for {len(comp_data['comparisons'])} dimensions.")
        for row in comp_data['comparisons']:
            print(f"    - {row['dim']:3d}d | Latency: {row['latency_ms']:4.2f}ms | Savings: -{row['savings_pct']}% | Top match score: {row['top1_score']}")

    print("\n==========================================")
    print("ALL TESTS PASSED PERFECTLY!")
    print("==========================================")

if __name__ == "__main__":
    test_engine()
