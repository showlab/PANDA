#!/usr/bin/env python3
"""Build or verify rule indexes using a local all-MiniLM-L6-v2 model."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

TEXT_TEMPLATE = (
    "{Rule Description} (Scene: {Specific Applicable Scenes}, Type: {Event Type})"
)
ATOL = 1e-5
RTOL = 1e-4


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("build", "verify"))
    parser.add_argument("--rules-json", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--embedding-model-path", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.rules_json.is_file():
        parser.error("--rules-json must name an existing JSON file")
    if not args.embedding_model_path.is_dir():
        parser.error(
            "--embedding-model-path must name an existing local model directory"
        )
    if args.mode == "build" and args.index.exists():
        parser.error(
            "build does not overwrite an existing index; choose a new --index path"
        )
    if args.mode == "verify" and not args.index.is_file():
        parser.error("verify requires an existing --index file")

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from panda.rag.retriever import _load_and_validate_rules

    rules = _load_and_validate_rules(args.rules_json)
    if any("Event Type" not in rule for rule in rules):
        parser.error("index construction requires Event Type on every rule")
    texts = [TEXT_TEMPLATE.format_map(rule) for rule in rules]

    # Initialize the embedding runtime before loading FAISS.
    from sentence_transformers import SentenceTransformer

    # isort: split
    import faiss
    import numpy as np

    model = SentenceTransformer(
        str(args.embedding_model_path), device="cpu", local_files_only=True
    )
    embeddings = np.ascontiguousarray(
        model.encode(texts, convert_to_numpy=True), dtype=np.float32
    )
    if embeddings.ndim != 2 or embeddings.shape[0] != len(rules):
        raise ValueError(f"Invalid rule embedding shape: {embeddings.shape}")
    if not np.isfinite(embeddings).all():
        raise ValueError("Rule embeddings contain NaN or Inf")
    report = {
        "mode": args.mode,
        "rules_file": args.rules_json.name,
        "rules_sha256": hashlib.sha256(args.rules_json.read_bytes()).hexdigest(),
        "index_file": args.index.name,
        "embedding_model_id": "sentence-transformers/all-MiniLM-L6-v2",
        "embedding_device": "cpu",
        "text_template": TEXT_TEMPLATE,
        "rule_count": len(rules),
        "d": int(embeddings.shape[1]),
        "metric": "squared_l2",
    }
    if args.mode == "build":
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        args.index.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation also protects an index created after argument validation.
        with args.index.open("xb") as handle:
            handle.write(faiss.serialize_index(index).tobytes())
        passed = True
    else:
        index = faiss.read_index(str(args.index))
        if (
            not isinstance(index, faiss.IndexFlatL2)
            or int(index.metric_type) != int(faiss.METRIC_L2)
            or int(index.ntotal) != len(rules)
            or int(index.d) != embeddings.shape[1]
        ):
            raise ValueError(
                "Index type, metric, count or dimension does not match rule embeddings"
            )
        stored = index.reconstruct_n(0, index.ntotal)
        per_row = np.all(np.isclose(embeddings, stored, rtol=RTOL, atol=ATOL), axis=1)
        passed = bool(per_row.all())
        report.update(
            {
                "vector_order_pairing": passed,
                "row_pass_count": int(per_row.sum()),
                "max_absolute_error": float(np.abs(embeddings - stored).max()),
                "atol": ATOL,
                "rtol": RTOL,
            }
        )
    report.update(
        {
            "ntotal": int(index.ntotal),
            "index_sha256": hashlib.sha256(args.index.read_bytes()).hexdigest(),
            "success": passed,
        }
    )
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
