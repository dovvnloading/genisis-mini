import json
import re
from pathlib import Path

import numpy as np


def get_next_filename(base_name="synthetic_dataset", extension=".jsonl"):
    output_dir = Path(".")
    base_path = output_dir / base_name
    initial_path = base_path.with_suffix(extension)
    if not initial_path.exists():
        return str(initial_path)

    counter = 1
    while True:
        versioned_path = output_dir / f"{base_name}_{counter:03d}{extension}"
        if not versioned_path.exists():
            return str(versioned_path)
        counter += 1


def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return dot_product / (norm_v1 * norm_v2)


def extract_json_from_response(content):
    fenced_match = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)
    if fenced_match:
        return fenced_match.group(1)

    start_index = content.find("{")
    if start_index == -1:
        return None

    depth = 0
    for index in range(start_index, len(content)):
        char = content[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return content[start_index : index + 1]

    return None


def parse_json_object(content, context="response"):
    json_str = extract_json_from_response(content)
    if not json_str:
        raise ValueError(f"No JSON object found in {context}.")

    payload = json.loads(json_str)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {context}.")
    return payload


def normalize_topic(candidate):
    return re.sub(r"\s+", " ", candidate.strip().lower())
