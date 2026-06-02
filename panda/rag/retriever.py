"""Validated local dataset rule retrieval."""

import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

RULES_DIR = Path(
    os.getenv("PANDA_RULES_DIR", str(Path(__file__).resolve().parent / "rules"))
)

DATASET_RULES: Dict[str, Dict[str, Optional[str]]] = {
    "ucf": {
        "json": "ucf_anomaly_rules_20.json",
        "index": "ucf_anomaly_rules_faiss_20.index",
    },
    "xd": {
        "json": "xd_anomaly_rules_20.json",
        "index": "xd_anomaly_rules_faiss_20.index",
    },
    "ub": {
        "json": "ub_train_anomaly_rules_20.json",
        "index": "ub_train_anomaly_rules_faiss_20.index",
    },
    "sht": {
        "json": "sht_anomaly_rules_20.json",
        "index": "sht_anomaly_rules_faiss_20.index",
    },
    "avenue": {
        "json": "avenue_anomaly_rules_10.json",
        "index": "avenue_anomaly_rules_faiss_10.index",
    },
    "ped2": {
        "json": "ped2_anomaly_rules_10.json",
        "index": "ped2_anomaly_rules_faiss_10.index",
    },
    "csad": {
        "json": "CSAD_anomaly_rules_20.json",
        "index": "CSAD_anomaly_rules_faiss_20.index",
    },
}

_REQUIRED_RULE_FIELDS = (
    "Rule ID",
    "Rule Description",
    "Specific Applicable Scenes",
)
_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


def _load_and_validate_rules(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required rules JSON does not exist: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid rules JSON at {path}: {exc.msg}") from exc
    except OSError as exc:
        raise OSError(f"Could not read rules JSON at {path}: {exc}") from exc

    if not isinstance(payload, list) or not payload:
        raise ValueError(f"rules JSON must contain a non-empty list: {path}")
    seen_rule_ids: Dict[str, set[Optional[str]]] = {}
    for index, rule in enumerate(payload):
        if not isinstance(rule, dict):
            raise ValueError(f"Invalid rule schema at index {index}: expected object")
        for field in ("Applicable Scenes", "Event Type"):
            if field in rule and (
                not isinstance(rule[field], str) or not rule[field].strip()
            ):
                raise ValueError(
                    f"Invalid rule schema at index {index}: {field!r} must be a non-empty string"
                )
        if "Specific Applicable Scenes" not in rule and "Applicable Scenes" in rule:
            rule["Specific Applicable Scenes"] = rule["Applicable Scenes"]
        for field in _REQUIRED_RULE_FIELDS:
            value = rule.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"Invalid rule schema at index {index}: {field!r} must be a non-empty string"
                )
        rule_id = rule["Rule ID"]
        event_type = rule.get("Event Type")
        previous_events = seen_rule_ids.setdefault(rule_id, set())
        if previous_events and (
            event_type is None
            or None in previous_events
            or event_type in previous_events
        ):
            raise ValueError(
                f"Duplicate rule identity (Event Type={event_type!r}, Rule ID={rule_id!r}) "
                f"at index {index}"
            )
        previous_events.add(event_type)
        if "Heuristics" in rule:
            heuristics = rule["Heuristics"]
            if (
                not isinstance(heuristics, list)
                or not heuristics
                or any(
                    not isinstance(item, str) or not item.strip() for item in heuristics
                )
            ):
                raise ValueError(
                    f"Invalid rule schema at index {index}: "
                    "'Heuristics' must be a non-empty list of non-empty strings"
                )
    return payload


def _tokens(text: str) -> set[str]:
    return {token.casefold() for token in _TOKEN_RE.findall(text)}


class RuleRetriever:
    """Retrieve validated local rules with an explicitly selected backend."""

    def __init__(
        self,
        dataset_name: str,
        *,
        backend: str = "lexical",
        rules_dir: Optional[os.PathLike[str] | str] = None,
        embedding_model_path: Optional[os.PathLike[str] | str] = None,
    ):
        if dataset_name not in DATASET_RULES:
            raise ValueError(
                f"Unsupported dataset '{dataset_name}'. Valid options: {sorted(DATASET_RULES)}"
            )
        if backend not in {"lexical", "faiss"}:
            raise ValueError("rag backend must be explicitly 'lexical' or 'faiss'")

        self.dataset_name = dataset_name
        self.backend = backend
        self.rules_dir = (
            Path(rules_dir)
            if rules_dir is not None
            else Path(os.getenv("PANDA_RULES_DIR", str(RULES_DIR)))
        )
        specification = DATASET_RULES[dataset_name]
        self.rules_path = self.rules_dir / str(specification["json"])
        self.rules = _load_and_validate_rules(self.rules_path)
        self.index = None
        self.embedder = None
        self._faiss_lower_is_better = True
        self.last_diagnostics: Dict[str, Any] = {}

        if backend == "faiss":
            self._initialize_faiss(specification, embedding_model_path)

    def _initialize_faiss(
        self,
        specification: Dict[str, Optional[str]],
        embedding_model_path: Optional[os.PathLike[str] | str],
    ) -> None:
        configured_model = embedding_model_path
        if configured_model is None:
            configured_model = os.getenv("EMBEDDING_MODEL_PATH", "")
        model_path = Path(configured_model) if configured_model else None
        if model_path is None or not model_path.exists():
            raise FileNotFoundError(
                "faiss backend requires EMBEDDING_MODEL_PATH to be an existing local path"
            )

        index_filename = specification.get("index")
        if not index_filename:
            raise FileNotFoundError(
                f"No FAISS index is configured for dataset {self.dataset_name}"
            )
        index_path = self.rules_dir / index_filename
        if not index_path.is_file():
            raise FileNotFoundError(
                f"Required local FAISS index does not exist: {index_path}"
            )

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError("faiss backend requires sentence-transformers") from exc

        try:
            import faiss
        except ImportError as exc:
            raise ImportError("faiss backend requires the local faiss package") from exc
        index = faiss.read_index(str(index_path))
        if int(index.ntotal) != len(self.rules):
            raise ValueError(
                f"FAISS index/rules mismatch: ntotal={index.ntotal}, rules={len(self.rules)}"
            )

        self.index = index
        self._faiss_lower_is_better = int(
            getattr(index, "metric_type", faiss.METRIC_L2)
        ) == int(faiss.METRIC_L2)
        self.embedder = SentenceTransformer(str(model_path), local_files_only=True)

    def retrieve(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Return top-k rules and expose deterministic backend diagnostics."""
        if not isinstance(query_text, str):
            raise ValueError("query_text must be a string")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        return (
            self._retrieve_lexical(query_text, top_k)
            if self.backend == "lexical"
            else self._retrieve_faiss(query_text, top_k)
        )

    def _retrieve_lexical(self, query_text: str, top_k: int) -> List[Dict[str, Any]]:
        query_tokens = _tokens(query_text)
        ranked = []
        for original_index, rule in enumerate(self.rules):
            searchable = " ".join(str(rule[field]) for field in _REQUIRED_RULE_FIELDS)
            overlap = len(query_tokens & _tokens(searchable))
            ranked.append((overlap, original_index, rule))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        max_overlap = ranked[0][0]
        selected = [item[2] for item in ranked[:top_k]]
        self.last_diagnostics = {
            "backend": "lexical",
            "zero_overlap": max_overlap == 0,
            "top_k": top_k,
            "rule_count": len(self.rules),
            "max_token_overlap": max_overlap,
        }
        return selected

    def _retrieve_faiss(self, query_text: str, top_k: int) -> List[Dict[str, Any]]:
        import numpy as np

        query_vector = np.asarray(self.embedder.encode([query_text]), dtype=np.float32)
        if query_vector.ndim != 2 or query_vector.shape[0] != 1:
            raise ValueError(
                f"Embedding backend returned invalid shape: {query_vector.shape}"
            )
        index_dimension = int(self.index.d)
        if query_vector.shape[1] != index_dimension:
            raise ValueError(
                "Embedding/FAISS dimension mismatch: "
                f"embedding={query_vector.shape[1]}, index={index_dimension}"
            )
        if not np.isfinite(query_vector).all():
            raise ValueError("Embedding backend returned NaN or Inf")
        scores, indices = self.index.search(query_vector, min(top_k, len(self.rules)))
        ranked_results = []
        for score, index in zip(scores[0], indices[0]):
            index_value = int(index)
            score_value = float(score)
            if index_value < 0 or index_value >= len(self.rules):
                raise ValueError(f"FAISS returned invalid rule index: {index_value}")
            if not math.isfinite(score_value):
                raise ValueError("FAISS returned NaN or Inf score")
            ranked_results.append((score_value, index_value))
        ranked_results.sort(
            key=lambda item: (
                item[0] if self._faiss_lower_is_better else -item[0],
                item[1],
            )
        )
        selected = [self.rules[index] for _, index in ranked_results]
        selected_scores = [score for score, _ in ranked_results]
        self.last_diagnostics = {
            "backend": "faiss",
            "zero_overlap": None,
            "top_k": top_k,
            "rule_count": len(self.rules),
            "index_ntotal": int(self.index.ntotal),
            "scores": selected_scores,
        }
        return selected
