"""Planning agent — generates detection strategy from env info + RAG rules."""

import json
import time

from panda.agents.utils import load_provider_object
from panda.paper_prompts import planning_prompt
from panda.providers import request_json_text
from panda.rag.retriever import RuleRetriever


def _validate_planning_contract(value: object) -> dict:
    if not isinstance(value, dict) or set(value) != {
        "preprocessing",
        "potential_anomalies",
        "heuristic_prompts",
    }:
        raise ValueError("Planning provider must return the complete planning object")
    preprocessing = value["preprocessing"]
    if not isinstance(preprocessing, str) or (
        preprocessing != "" and not preprocessing.strip()
    ):
        raise ValueError(
            "Planning provider preprocessing must be explicit empty or non-empty text"
        )
    if not isinstance(value["potential_anomalies"], str):
        raise ValueError("Planning provider potential_anomalies must be text")
    if not isinstance(value["heuristic_prompts"], dict):
        raise ValueError("Planning provider heuristic_prompts must be an object")
    return dict(value)


def build_planning_prompt(query: str, env_info: dict, anomaly_rules) -> tuple:
    """Build planning system/user prompts."""
    json_template = """{
  "preprocessing": "Step1 -> Step2 -> Step3",
  "potential_anomalies": "Fighting, Stealing, ...",
  "heuristic_prompts": {
    "Fighting": "Heuristic prompt with reasoning steps...",
    "Stealing": "Heuristic prompt with reasoning steps..."
  }
}"""
    example_output = """{
  "preprocessing": "Brightness Enhancement -> Video Denoising -> Super-Resolution",
  "potential_anomalies": "Fighting, Stealing",
  "heuristic_prompts": {
    "Fighting": "Observe the number of people, their movements, and interactions. If two individuals are repeatedly making aggressive contact, consider it a potential fight.",
    "Stealing": "Identify solitary individuals interacting with objects, especially if they conceal items or leave quickly without paying."
  }
}"""

    system_prompt = f"""You are a strategic expert in video anomaly detection, specializing in planning effective detection strategies based on user-defined requirements, video environment information, and scene-specific anomaly rules.

User Requirement: {query}

Video Environment Information:
Scene Overview: {env_info.get("Scene overview", "Unknown")}
Weather / Lighting: {env_info.get("Weather/Lighting", "Unknown")}
Video Quality: {env_info.get("Video quality", "Unknown")}
Potential anomalies: {env_info.get("Potential anomalies", "Unknown")}

Anomaly Detection Rules:
{anomaly_rules}

Based on the user requirement, the preliminary video environment information, and the provided anomaly rules, please design a strategy tailored for this video scenario. Your response must include:
1. Preprocessing Recommendations and Pipeline
2. Potential Anomaly Types
3. Heuristic Prompts for VLMs

The preprocessing field is an executable pixel-transform pipeline. It may contain only
these exact step names, joined with ` -> ` in execution order: Brightness Enhancement,
Video Denoising, Deblurring, Super-Resolution. If no pixel transform is needed, return
exactly `"preprocessing": ""`. Do not place Object Detection, Motion Detection, Object
Tracking, Web Search, retrieval, analysis, or any other operation in preprocessing.

Please return your output strictly in the following JSON format:
{json_template}
Example Output:
{example_output}
***Please return only the raw JSON object, without any markdown code block or explanation."""

    human = "Please formulate a tailored video anomaly detection strategy for this scenario."
    return system_prompt, human


def planning_agent(state: dict) -> dict:
    """Planning node: retrieve rules and generate detection strategy."""
    query = state["user_query"]
    env_info = state.get("env_info", {})
    num_rules = state.get("num_rules", 5)
    dataset_name = state.get("dataset_name", "ucf")
    rag_backend = state.get("rag_backend", "lexical")

    scene_overview = env_info.get("Scene overview", "Unknown")
    possible_anomaly = env_info.get("Potential anomalies", "Unknown")

    retriever = RuleRetriever(dataset_name, backend=rag_backend)
    rule_query = (
        f"Scene Overview: {scene_overview} Potential Anomalies: {possible_anomaly}"
    )
    anomaly_rules = retriever.retrieve(rule_query, top_k=num_rules)

    rules_dict = {}
    for r in anomaly_rules:
        key = json.dumps(
            [r.get("Event Type"), r["Rule ID"]],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        rules_dict[key] = dict(r)

    paper_mode = state.get("profile", "engineering").startswith("paper-")
    prompt_builder = planning_prompt if paper_mode else build_planning_prompt
    system_prompt, human = prompt_builder(query, env_info, rules_dict)

    start = time.time()
    provider_diagnostic = {}
    result_text = request_json_text(
        role="planning",
        provider=state.get("planning_provider"),
        system_prompt=system_prompt,
        user_prompt=human,
        diagnostics=provider_diagnostic,
    )
    value = load_provider_object(result_text, "Planning")
    if paper_mode and isinstance(value, dict):
        aliases = {
            "Preprocessing": "preprocessing",
            "Potential Anomalies": "potential_anomalies",
            "Heuristic Prompts": "heuristic_prompts",
        }
        if set(value) == set(aliases):
            value = {aliases[key]: item for key, item in value.items()}
    text = _validate_planning_contract(value)

    provider_diagnostics = dict(state.get("provider_diagnostics", {}))
    provider_diagnostics["planning"] = provider_diagnostic

    print(f"planning_time: {time.time() - start:.1f}s")
    return {
        **state,
        "planning_info": text,
        "anomaly_rules": rules_dict,
        "rag_backend": rag_backend,
        "rag_diagnostics": dict(retriever.last_diagnostics),
        "planning_prompt": system_prompt,
        "provider_diagnostics": provider_diagnostics,
    }
