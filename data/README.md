# Data Directory

This directory stores bid documents and generated search indices.

- **`data/bids/`**: Contains bid folders (`Bid1/`, `Bid2/`). See [data/bids/README.md](bids/README.md) for complete file breakdowns and document type mappings.
- **`data/bm25_indices/`**: Stores generated BM25 inverted index files (`<bid_id>_bm25.pkl` and `<bid_id>_meta.json`) for fast keyword matching.
