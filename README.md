# Genisis-Mini

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Ollama](https://img.shields.io/badge/Ollama-local%20LLM-111111.svg)](https://ollama.com/)
[![PySide6](https://img.shields.io/badge/UI-PySide6-41CD52.svg)](https://doc.qt.io/qtforpython-6/)

Genisis-Mini is a local desktop application for generating small, structured synthetic datasets from a single topic. It uses Ollama-hosted language models to expand a subject into sub-topics, filter them for relevance and diversity, and export validated input/output pairs as JSON or JSONL.

## What It Does

- Generates topic-focused synthetic dataset entries from a user-defined subject.
- Uses separate generation and validation steps instead of a single prompt.
- Filters similar topics with embedding-based cosine similarity checks.
- Runs locally through Ollama with no hosted API requirement.
- Exports results in `json` and `jsonl` formats.

## How It Works

1. Enter a main topic in the desktop application.
2. The generator model proposes candidate sub-topics.
3. A validator model removes off-topic candidates.
4. Embeddings are compared to reduce semantic duplicates.
5. The application generates text snippets for approved topics.
6. Final validated records are written to an output dataset file.

Each record follows a simple structure:

```json
{
  "input": "sub-topic",
  "output": "generated text"
}
```

## Requirements

- Python 3.8 or newer
- [Ollama](https://ollama.com/) installed and running
- The following Ollama models available locally:

```bash
ollama pull nomic-embed-text
ollama pull qwen3:4b
ollama pull qwen3:1.7b
```

## Installation

```bash
git clone https://github.com/dovvnloading/genisis-mini.git
cd genisis-mini
python -m venv .venv
source .venv/bin/activate
pip install PySide6 ollama numpy
```

## Run

```bash
python DataSource/DataSource/DataSource.py
```

## Configuration

The application allows you to tune:

- main topic
- target topic count
- similarity threshold
- batch size
- generation model
- validation model

Generated files are saved in the working directory with incrementing names such as `synthetic_dataset.jsonl` and `synthetic_dataset_001.jsonl`.

## Project Structure

```text
DataSource/
  DataSource/
    DataSource.py
README.md
LICENSE
```

## Use Cases

Genisis-Mini is best suited for:

- bootstrapping small domain datasets
- creating educational or prototype training data
- experimenting with topic decomposition workflows
- generating local data without external API costs

## License

This project is licensed under the Apache License 2.0. See `LICENSE` for details.
