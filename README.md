# Genisis-Mini

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![Powered by Ollama](https://img.shields.io/badge/Powered%20by-Ollama-232f3e)](https://ollama.com/)
[![Made with Python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)

Genisis-Mini is a desktop application designed to empower independent researchers, developers, and educators to create high-quality, specialized micro-datasets. By leveraging local large language models (LLMs) through Ollama, it provides a powerful, private, and cost-free tool for generating structured data suitable for educational purposes, fine-tuning smaller models, or preliminary research.

<img width="802" height="732" alt="Screenshot 2025-10-04 115020" src="https://github.com/user-attachments/assets/e4f340e3-c011-4d50-980e-0cac137449f2" />
<img width="802" height="732" alt="Screenshot 2025-10-04 115420" src="https://github.com/user-attachments/assets/442b7bb6-1e2a-4468-8d8d-a3829f7467bb" />


Our core mission is the **democratization of intelligence**—lowering the barrier to entry for custom AI development and research by addressing one of the most significant bottlenecks: the creation of curated, high-quality datasets.

---

## Table of Contents

- [Core Philosophy](#core-philosophy)
- [Features](#features)
- [How It Works: An Agentic Approach](#how-it-works-an-agentic-approach)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation & Execution](#installation--execution)
- [License](#license)
- [Commercial Use & Contact](#commercial-use--contact)
- [Credits & Acknowledgements](#credits--acknowledgements)
- [Contributing](#contributing)

---

## Core Philosophy

In the rapidly evolving field of artificial intelligence, access to diverse and well-structured data is paramount. However, dataset creation is often a resource-intensive process, placing it out of reach for many independent contributors. Genisis-Mini was built to bridge this gap. It operates on the principle that quality and conceptual diversity are more valuable than sheer volume, especially in specialized domains. By employing a sophisticated, multi-stage agentic workflow, the tool ensures that each generated data point is relevant, unique, and directly aligned with the user's specified domain.

## Features

- **Agentic Generation Workflow:** Utilizes a multi-step process involving separate generation and validation agents to ensure high-quality, relevant output.
- **Local First with Ollama:** Runs entirely on your local machine, guaranteeing data privacy and eliminating API costs.
- **Semantic Diversity Control:** Employs embedding models and cosine similarity thresholds to prevent semantic duplication and ensure a broad conceptual coverage of the main topic.
- **Intuitive Graphical User Interface:** A straightforward GUI built with PySide6 allows for easy configuration and monitoring of the generation process.
- **Configurable Parameters:** Provides granular control over the number of topics, sample size, diversity thresholds, batch sizes, and model selection.
- **Flexible Export Options:** Generates datasets in standard `JSONL` and `JSON` formats for immediate use in training and analysis pipelines.

<!--
    *** A screenshot of the application UI would be beneficial here ***
    ![Genisis-Mini UI](path/to/screenshot.png)
-->

## How It Works: An Agentic Approach

Genisis-Mini employs a pipeline of specialized agents to transform a single high-level topic into a structured dataset.

1.  **Stage 1: Topic Expansion:**
    - The user provides a central **Main Topic**.
    - A *Generation Agent* is prompted to brainstorm a batch of related sub-topics. To encourage novelty, it is provided with a list of recently accepted topics to avoid.

2.  **Stage 2: Topic Validation & Filtering:**
    - A *Relevance Validation Agent* assesses whether each candidate sub-topic is logically and thematically connected to the Main Topic.
    - For relevant candidates, an embedding is generated (using `nomic-embed-text`). This embedding is compared against the embeddings of all previously accepted topics.
    - If the **cosine similarity** is below a user-defined threshold, the new topic is deemed conceptually unique and is added to the pool. This step is critical for preventing a dataset of paraphrased ideas.

3.  **Stage 3: Data Generation:**
    - A *Data Generation Agent* cycles through the validated list of unique sub-topics.
    - For each sub-topic, it generates a concise, high-quality text snippet that explains or elaborates on that concept.

4.  **Stage 4: Snippet Validation:**
    - A *Snippet Validation Agent* reviews each generated text snippet against its corresponding sub-topic.
    - It confirms that the text is on-topic, coherent, and factually plausible. Only snippets that pass this final quality check are accepted.

5.  **Stage 5: Export:**
    - The final collection of validated `{"input": "sub-topic", "output": "generated_text"}` pairs is compiled and saved to a file in the user's chosen format (`.jsonl` or `.json`).

## Getting Started

### Prerequisites

Before running Genisis-Mini, ensure your system meets the following requirements:

1.  **Python:** Python 3.8 or newer.
2.  **Ollama:** You must have [Ollama](https://ollama.com/) installed and running on your system.
3.  **Ollama Models:** The application requires several models to be pulled from the Ollama library. Please pull them before launching the application by running the following commands in your terminal:

    ```bash
    ollama pull nomic-embed-text
    ollama pull granite4:tiny-h
    ollama pull granite4:micro-h
    ```
    
    > **Note:** The `granite` models are used by default for generation and validation but can be substituted with any other compatible model available in your Ollama instance via the application's UI. `nomic-embed-text` is required for the similarity calculations.

### Installation & Execution

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/dovvnloading/genisis-mini.git
    cd genisis-mini
    ```

2.  **Set up a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Dependencies:**
    The project relies on PySide6, Ollama, and NumPy. Install them using pip:
    ```bash
    pip install PySide6 ollama numpy
    ```
    
    *(Alternatively, if a `requirements.txt` file is provided: `pip install -r requirements.txt`)*

4.  **Run the Application:**
    Ensure Ollama is running in the background, then execute the main Python script:
    ```bash
    python genisis-mini.py
    ```

5.  **Configure and Generate:**
    - Fill in the "Main Topic" and adjust the parameters in the UI as needed.
    - Click "Start Generation Process" to begin.
    - Monitor the progress in the log window. Upon completion, a file (e.g., `synthetic_dataset_001.jsonl`) will be created in the project's root directory.

## License

This project is licensed under the **Apache License 2.0**. A copy of the license is available in the `LICENSE` file in this repository.

## Commercial Use & Contact

We encourage the use of this tool for all purposes, including commercial applications and inclusion in production pipelines. If you do so, we would appreciate a credit to the project.

For inquiries regarding commercial use, collaboration, or support, please contact the developer directly via email: **devaux.mail@gmail.com**

## Credits & Acknowledgements

This project was made possible by the efforts and contributions of the following:

-   **Matthew Wesney**
-   Conceptual assistance from **Google Gemini**

## Contributing

Contributions are welcome. If you have a suggestion, find a bug, or want to add a new feature, please follow these steps:

1.  Fork the repository.
2.  Create a new branch for your feature (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

Please open an issue first to discuss any major changes you would like to make.

---
