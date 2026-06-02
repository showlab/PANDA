"""Appendix E prompts from PANDA's NeurIPS 2025 proceedings, Figures 7–10.

Line wrapping and JSON punctuation are normalized. Python state keys are mapped
to the paper's output field names. Human follow-up messages supply runtime context.
"""

from __future__ import annotations

import json


def perception_prompt(user_query: str) -> tuple[str, str]:
    prompt = f"""You are an expert in video anomaly perception. Your task is to perform an initial understanding and analysis of the provided video frames based on the user's specified requirements.
User Requirement:
{user_query}
Please respond by completing the following four aspects:
1. Scene Overview: Describe the environment shown in the video (e.g., shopping mall, office, street, surveillance corridor) and briefly summarize the main activities or events observed.
2. Weather Condition: Describe the visual lighting conditions (e.g., daytime/nighttime, sunny/overcast, bright/dim).
3. Video Quality: Comment on the overall quality of the video (e.g., clear, blurry, noisy, low frame rate, low light.etc.).
4. Potential Anomalies: Based on the visual content, what types of abnormal events are likely to occur in this video (e.g., Fighting, Stealing).
Please return your output strictly in the following JSON format:
{{
  "Scene Overview": "...",
  "Weather Condition": "...",
  "Video Quality": "...",
  "Potential Anomalies": "..."
}}"""
    return prompt, "Please analyze the input video frames."


def planning_prompt(user_query: str, env_info: dict, anomaly_rules) -> tuple[str, str]:
    prompt = f"""You are a strategic expert in video anomaly detection, specializing in planning effective detection strategies based on user-defined requirements, video environment information, and scene-specific anomaly rules. Your role is to generate an optimal plan that guides the analysis module in accurately detecting anomalies in the current video scene.
User Requirement:
{user_query}
Video Environment Information:
1. Scene Overview: {env_info.get("Scene overview", "Unknown")}
2. Weather Condition: {env_info.get("Weather/Lighting", "Unknown")}
3. Video Quality: {env_info.get("Video quality", "Unknown")}
4. Potential Anomalies: {env_info.get("Potential anomalies", "Unknown")}
Anomaly Detection Rules:
{anomaly_rules}
Based on the user’s requirement, the preliminary video environment information, and the provided anomaly rules, please design a strategy tailored for this video scenario. Your response must include the following three components:
1. Preprocessing Recommendations and Pipeline
Suggest a sequence of preprocessing steps (e.g., Image Deblurring, Brightness Enhancement, Image Denoising) that can help improve video quality and support better anomaly detection, especially if the video is of poor quality.
2. Potential Anomaly Types
Based on the preliminary video environment information and the given scene-related anomaly rules, further infer and list the most possible types of anomalies in this scenario.
3. Heuristic Prompts for VLM
Using the anomaly rules as guidance, craft chain-of-thought–style heuristic prompts for each potential anomaly type. These prompts are intended to assist Visual Language Model in performing accurate anomaly judgments.
Please return your output strictly in the following JSON format:
{{
  "Preprocessing": "Step1 -> Step2 -> ...",
  "Potential Anomalies": "Fighting, Stealing, ...",
  "Heuristic Prompts": {{
    "Fighting": "Heuristic prompt with reasoning steps...",
    "Stealing": "Heuristic prompt with reasoning steps..."
  }}
}}
Example Output:
{{
  "Preprocessing": "Image Brightness Enhancement -> Image Denoising",
  "Potential Anomalies": "Fighting, Stealing",
  "Heuristic Prompts": {{
    "Fighting": "Observe the number of people, their movements, and interactions. If two individuals are repeatedly making aggressive contact, consider it a potential fight.",
    "Stealing": "Identify solitary individuals interacting with objects, especially if they conceal items or leave quickly without paying."
  }}
}}"""
    return (
        prompt,
        "Please formulate a tailored video anomaly detection strategy for this scenario. "
        "Execution interface: Preprocessing is an executable pixel-transform pipeline, "
        "not explanatory prose. Use only these exact names, joined by ` -> `: "
        "Brightness Enhancement, Video Denoising, Deblurring, Super-Resolution. "
        'If no pixel transform is needed, return exactly "Preprocessing": "". '
        "Do not put Motion Detection, Object Detection, Object Tracking, Web Search, "
        "retrieval, analysis, or any other operation into Preprocessing. "
        "The executor rejects unsupported steps; keep detection reasoning in Heuristic Prompts.",
    )


def analysis_prompt(
    user_query,
    planning_info,
    anomaly_rules,
    history_result_prompt,
    enhancement_prompt_blocks,
    index,
):
    formatted = "\n\n".join(
        f"[Reflection #{i}]\n{block.strip()}"
        for i, block in enumerate(enhancement_prompt_blocks, 1)
    )
    formatted = (
        formatted or "No additional enhancement or reflection information available."
    )
    prompt = f"""You are a highly skilled expert in video anomaly detection, specializing in identifying abnormal events through temporal and spatial analysis of visual evidence.
Given a sequence of video frames, user requirements, potential anomalies, anomaly detection rules, heuristic prompts, and enhancement/reflection information, your task is to assess the likelihood of abnormal events in the current video clip.
You must output:
- A soft anomaly score between **0.0 (clearly normal)** and **1.0 (clearly abnormal)**.
- A status label from: **"normal"**, **"abnormal"**, or **"insufficient"**.
- A reason justifying your decision.
User Requirement:
{user_query}
Historical Detection Info:
{history_result_prompt if history_result_prompt.strip() else "No reliable historical detection information available."}
Current Video Clip Index:
Clip {index}
Potential Anomalies:
{planning_info.get("potential_anomalies", "")}
Anomaly Detection Rules:
{anomaly_rules}
Heuristic Prompt:
{planning_info.get("heuristic_prompts", "")}
Enhancement and Reflection Information:
{formatted}
Your analysis should follow three steps:
1. Describe the main visible actions and interactions between people or objects in the scene.
2. Assess how strongly these actions match any known abnormal event patterns using the provided rules and prompts. The provided anomaly rules may not be comprehensive, so you also apply your own expert reasoning.
3. Based on your assessment, assign a score and label, and explain your reasoning clearly.
4. If the [Enhancement and Reflection Information] section provides additional information, you should refer to it emphatically.
Scoring Guidelines:
- A score close to **1.0** indicates clear and confident abnormal behavior.
- A score close to **0.0** indicates clearly normal behavior.
- A score near **0.5** means uncertain, ambiguous behavior or mixed signals.
Examples of valid reasons for "insufficient":
- "The entire scene is too blurry or dark, making it difficult to distinguish any actions."
- "All persons are either occluded or out of frame."
- "Only partial limbs are visible and motion cues are unclear."
Please strictly output your response in the following JSON format:
{{
  "score": float,
  "status": "normal/abnormal/insufficient",
  "reason": "A detailed explanation of your reasoning..."
}}"""
    return (
        prompt,
        "Please analyze the input video frames and output your result in the specified JSON format.",
    )


def reflection_prompt(state: dict, memory_records: list, tool_description_text: str):
    env = state.get("env_info", {})
    planning = state.get("planning_info", {})
    current_index = state.get("window_index", 0)
    prior = [
        r
        for r in state.get("analysis_result", [])
        if r.get("index", current_index) < current_index
    ]
    latest = {r["index"]: r for r in prior}
    history = (
        "\n".join(
            f"  * Clip {i}: {latest[i].get('reason', '')}" for i in sorted(latest)[-5:]
        )
        or "No historical results."
    )
    memory = (
        json.dumps(memory_records, ensure_ascii=False, sort_keys=True)
        if memory_records
        else "No matching reflection memory."
    )
    # Section 3.4 additionally requires reflection outputs in Short CoM.
    # Keep the appendix's detection history, but do not substitute it for these.
    reflection_history = [
        {
            "window_index": record["window_index"],
            "reflections": record.get("reflections", []),
        }
        for record in state.get("long_memory", [])
        if current_index - 5 <= record.get("window_index", -1) < current_index
    ]
    current_attempts = [
        attempt
        for attempt in state.get("reflection_attempts", [])
        if attempt.get("window_index") == current_index
    ]
    prompt = f"""You are a reflection assistant within a video anomaly detection system.
The current VLM analysis module has returned "insufficient status" for determining whether an abnormal event occurred in the given video clip.
Your task is to critically analyze the situation based on the provided context and recommend solutions.
Here is the contextual information:
- User Requirement: {state.get("user_query", "")}
- Video Environment Information:
    **Scene Overview: {env.get("Scene overview", "Unknown")}
    **Weather Condition: {env.get("Weather/Lighting", "Unknown")}
    **Video Quality: {env.get("Video quality", "Unknown")}
- Anomaly Detection Rules:
{state.get("anomaly_rules", {})}
- Potential Anomalies: {planning.get("potential_anomalies", "")}
- Historical Detection Results:
{history}
- Current VLM Output Reason: {state.get("insufficient_reason", "")}
- Information Enhancement tools Already Used: {state.get("tools_already_used", [])}
- History Experience: {memory}
- Past Reflection Outputs (Short CoM): {json.dumps(reflection_history, ensure_ascii=False)}
- Current Window Reflection Attempts: {json.dumps(current_attempts, ensure_ascii=False)}
Based on this information, your tasks are:
1. Analyze and determine the primary reasons for the insufficient information.
2. Recommend which tools from the available options should be used to enhance the information for better anomaly detection.
{tool_description_text}
3. For any selected tool that requires a 'query' input (e.g., image_retrieve, web_search), generate an appropriate query based on the context; otherwise leave the 'query' field empty.
4. Propose a new representative anomaly detection rule derived from the current situation to better support future VLM analysis.
5. Propose an additional heuristic prompt based on the context and your analysis to better guide the VLM toward an accurate judgment.
Please output your response in the following structured JSON format:
{{
  "reason": "...your analysis of why the information is insufficient...",
  "tools_to_use": [{{"tool_name": "One of the most critical tools.", "query": "generated query if needed, otherwise leave empty", "params": {{}}}}],
  "new_anomaly_rule": "...a new representative anomaly rule derived from your analysis and context...",
  "new_heuristic_prompt": "...additional guidance to help the VLM make a more accurate judgment..."
}}
Important Notes:
- For zoom_images, set params.zoom_factor to an explicit positive numeric magnification factor.
- When calling tools, make sure you don't duplicate any of the information enhancement tools that have already been applied, and use only one of the most critical tools at a time.
- If all available information enhancement tools have been exhausted, you should directly suggest in the 'new_heuristic_prompt' how to guide the VLM analysis module to make the most reasonable judgment based on incomplete evidence.
- If the current context provides enough information to make a clear judgment, please directly guide the VLM analysis module in the 'new_heuristic_prompt' to conclude whether the event is abnormal or normal."""
    return prompt, "Please reflect on the insufficient evidence and provide a solution."
