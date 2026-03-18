import json
import time

import numpy as np

from .config import (
    EMBEDDING_MODEL_NAME,
    MIN_SNIPPET_LENGTH,
    MODEL_REQUEST_OPTIONS,
    TOPIC_GENERATION_OPTIONS,
    VALIDATION_REQUEST_OPTIONS,
)
from .utils import cosine_similarity, normalize_topic, parse_json_object


class TopicRelevanceValidator:
    def __init__(self, model_name, progress_signal, client):
        self.model_name = model_name
        self.progress_signal = progress_signal
        self.client = client

    def is_relevant(self, main_topic, sub_topic_candidate):
        try:
            prompt = f"""
            You are a relevance validation AI.
            Main Topic: \"{main_topic}\"
            Sub-Topic Candidate: \"{sub_topic_candidate}\"
            Is the sub-topic candidate highly relevant to the main topic?
            Your response MUST be a single, raw JSON object and nothing else.
            Example: {{\"is_relevant\": true}}
            """
            response = self.client.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                options=VALIDATION_REQUEST_OPTIONS,
            )
            data = parse_json_object(response["message"]["content"], "relevance validator response")
            return data.get("is_relevant", False)
        except Exception as exc:
            self.progress_signal.emit((f"   - Relevance validation failed for '{sub_topic_candidate}': {exc}. Skipping.", "WARN"))
            return False


class TextSnippetValidator:
    def __init__(self, model_name, progress_signal, client):
        self.model_name = model_name
        self.progress_signal = progress_signal
        self.client = client

    def validate_relevance(self, topic, generated_text):
        self.progress_signal.emit((f"   - Validating snippet relevance for topic: '{topic}'...", "VALIDATION"))
        try:
            prompt = f"""
            You are a Validation AI.
            Topic: \"{topic}\"
            Text Snippet: \"{generated_text}\"
            Is the text snippet highly relevant and directly about the topic?
            Your response MUST be a single, raw JSON object and nothing else.
            Example: {{\"is_relevant\": true, \"reason\": \"The text directly addresses the topic.\"}}
            """
            response = self.client.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                options=VALIDATION_REQUEST_OPTIONS,
            )
            data = parse_json_object(response["message"]["content"], "snippet validator response")
            return data.get("is_relevant", False), data.get("reason", "No reason provided.")
        except Exception as exc:
            self.progress_signal.emit((f"   - Snippet validation error: {exc}", "ERROR"))
            return False, str(exc)


class TopicExpansionService:
    def __init__(self, client, generation_model_name, validation_model_name, progress_signal):
        self.client = client
        self.generation_model_name = generation_model_name
        self.progress_signal = progress_signal
        self.relevance_validator = TopicRelevanceValidator(validation_model_name, progress_signal, client)

    def generate_topics(self, main_topic, target_topic_count, similarity_threshold, batch_size):
        max_retries = 3
        for attempt in range(max_retries + 1):
            try:
                accepted_topics = []
                accepted_embeddings = []
                seen_topics = set()
                attempts = 0
                max_attempts = int(target_topic_count / batch_size * 5) + 20

                self.progress_signal.emit((f"Starting topic generation with batch size {batch_size}. Target: {target_topic_count} topics.", "STAGE"))

                while len(accepted_topics) < target_topic_count and attempts < max_attempts:
                    attempts += 1
                    needed = target_topic_count - len(accepted_topics)
                    current_batch_size = min(batch_size, needed)
                    self.progress_signal.emit((f"Starting batch {attempts}. Target: {target_topic_count}, Current: {len(accepted_topics)}, Requesting: {current_batch_size}", "INFO"))

                    avoid_list_str = ", ".join(f"'{topic}'" for topic in accepted_topics[-100:]) or "None yet"
                    prompt = f"""
                    You are a Curriculum Design AI specializing in creating diverse and conceptually unique educational modules.

                    Core Task: Generate a list of sub-topics for the main topic provided.
                    Main Topic: \"{main_topic}\"

                    Generation Guidelines & Constraints:
                    1. Be diverse: cover historical, technological, social, ethical, practical, and future-looking angles.
                    2. Be concise: each topic must be a concise phrase, maximum 7-9 words.
                    3. Quantity: generate exactly {current_batch_size} new, distinct sub-topics.
                    4. Avoid duplicates: do not repeat topics that differ only by punctuation, casing, or small wording changes.

                    Conceptual Areas to Avoid:
                    [{avoid_list_str}]

                    Output Format:
                    Your response MUST be a single raw JSON object and nothing else.
                    Example: {{\"topics\": [\"concise topic 1\", \"concise topic 2\"]}}
                    """
                    response = self.client.chat(
                        model=self.generation_model_name,
                        messages=[{"role": "user", "content": prompt}],
                        options=TOPIC_GENERATION_OPTIONS,
                    )
                    data = parse_json_object(response["message"]["content"], "topic generator response")
                    candidate_topics = data.get("topics", [])
                    if not candidate_topics:
                        self.progress_signal.emit(("   - LLM returned an empty topic list for the batch. Retrying.", "WARN"))
                        continue

                    newly_added_count = 0
                    for raw_candidate in candidate_topics:
                        if len(accepted_topics) >= target_topic_count:
                            break

                        if not isinstance(raw_candidate, str):
                            self.progress_signal.emit((f"   - REJECTED (Invalid Type): {raw_candidate}", "WARN"))
                            continue

                        candidate = " ".join(raw_candidate.strip().split())
                        normalized_candidate = normalize_topic(candidate)
                        if not candidate:
                            self.progress_signal.emit(("   - REJECTED (Empty Topic)", "WARN"))
                            continue
                        if normalized_candidate in seen_topics:
                            self.progress_signal.emit((f"   - REJECTED (Duplicate): '{candidate}'", "WARN"))
                            continue
                        if not self.relevance_validator.is_relevant(main_topic, candidate):
                            self.progress_signal.emit((f"   - REJECTED (Irrelevant): '{candidate}'", "WARN"))
                            continue

                        candidate_embedding_response = self.client.embeddings(model=EMBEDDING_MODEL_NAME, prompt=candidate)
                        candidate_embedding = np.array(candidate_embedding_response["embedding"])

                        is_too_similar = any(
                            cosine_similarity(candidate_embedding, embedding) > similarity_threshold
                            for embedding in accepted_embeddings
                        )
                        if is_too_similar:
                            self.progress_signal.emit((f"   - REJECTED (Similarity): '{candidate}'", "WARN"))
                            continue

                        self.progress_signal.emit((f"   - ACCEPTED ({len(accepted_topics) + 1}/{target_topic_count}): '{candidate}'", "SUCCESS"))
                        accepted_topics.append(candidate)
                        accepted_embeddings.append(candidate_embedding)
                        seen_topics.add(normalized_candidate)
                        newly_added_count += 1

                    self.progress_signal.emit((f"Batch complete. Added {newly_added_count} new unique topics.", "INFO"))

                if len(accepted_topics) < target_topic_count:
                    raise RuntimeError(
                        f"Topic generation stopped after producing {len(accepted_topics)} of {target_topic_count} requested topics."
                    )

                return accepted_topics

            except json.JSONDecodeError:
                self.progress_signal.emit(("   - Failed to parse extracted JSON. The model may have returned a malformed response. Retrying batch.", "WARN"))
                time.sleep(2)
            except Exception as exc:
                error_str = str(exc).lower()
                if "timed out" in error_str and attempt < max_retries:
                    self.progress_signal.emit((f"Network timeout detected. Retrying ({attempt + 1}/{max_retries})...", "WARN"))
                    time.sleep(3)
                    continue
                raise


class DatasetGenerationService:
    def __init__(self, client, generation_model_name, validation_model_name, progress_signal):
        self.client = client
        self.generation_model_name = generation_model_name
        self.progress_signal = progress_signal
        self.validator = TextSnippetValidator(validation_model_name, progress_signal, client)

    def generate_dataset(self, sub_topics, total_samples):
        dataset = []
        num_sub_topics = len(sub_topics)
        if num_sub_topics == 0:
            raise ValueError("Cannot generate data with an empty list of sub-topics.")

        attempts = 0
        max_attempts = total_samples * 3

        while len(dataset) < total_samples and attempts < max_attempts:
            sample_index = len(dataset)
            current_topic = sub_topics[sample_index % num_sub_topics]
            self.progress_signal.emit((f"Attempting sample {sample_index + 1}/{total_samples} for topic: '{current_topic}'", "INFO"))

            try:
                prompt = f"""
                You are a Data Generation AI. Write a single, high-quality text snippet for the given sub-topic.
                Sub-Topic: \"{current_topic}\"
                Your response MUST be a single, raw JSON object and nothing else.
                Format: {{\"text\": \"Your generated text here.\"}}
                """
                response = self.client.chat(
                    model=self.generation_model_name,
                    messages=[{"role": "user", "content": prompt}],
                    options=MODEL_REQUEST_OPTIONS,
                )
                text_entry = parse_json_object(response["message"]["content"], "data generation response")
                generated_text = text_entry.get("text", "").strip()
                if len(generated_text) < MIN_SNIPPET_LENGTH:
                    self.progress_signal.emit(("   - Generation failed: Text too short. Retrying.", "WARN"))
                    attempts += 1
                    continue

                is_relevant, reason = self.validator.validate_relevance(current_topic, generated_text)
                if not is_relevant:
                    self.progress_signal.emit((f"   - Snippet relevance FAILED. Reason: {reason}. Discarding.", "WARN"))
                    attempts += 1
                    continue

                self.progress_signal.emit((f"   - Generation and validation SUCCESSFUL for sample {sample_index + 1}.", "SUCCESS"))
                dataset.append({"input": current_topic, "output": generated_text})
            except Exception as exc:
                self.progress_signal.emit((f"   - Failed to generate or parse snippet: {exc}. Retrying.", "WARN"))
                attempts += 1
                time.sleep(2)

        if len(dataset) < total_samples:
            raise RuntimeError(f"Failed to generate {total_samples} samples after {attempts} attempts.")

        return dataset
