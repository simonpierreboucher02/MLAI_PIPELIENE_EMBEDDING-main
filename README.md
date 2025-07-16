# MLAI_PIPELINE_EMBEDDING

<div align="center">

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python Version](https://img.shields.io/badge/python-3.7%2B-blue.svg)
![OpenAI API](https://img.shields.io/badge/OpenAI-API-green.svg)
![GPT-4](https://img.shields.io/badge/GPT--4-Enabled-orange.svg)
![Embeddings](https://img.shields.io/badge/Embeddings-ADA--002-purple.svg)
![Concurrent Processing](https://img.shields.io/badge/Concurrent-Processing-yellow.svg)
![Rate Limiting](https://img.shields.io/badge/Rate%20Limiting-Supported-red.svg)
![YAML Config](https://img.shields.io/badge/YAML-Configuration-lightgrey.svg)

[![GitHub](https://img.shields.io/badge/GitHub-simonpierreboucher02-black?style=for-the-badge&logo=github)](https://github.com/simonpierreboucher02)
[![Made with Love](https://img.shields.io/badge/Made%20with-❤️-red.svg)](https://github.com/simonpierreboucher02)

**🚀 Advanced Text Processing & Embedding Pipeline with GPT-4 Contextualization**

</div>

## 📊 Repository Metrics

![GitHub repo size](https://img.shields.io/github/repo-size/simonpierreboucher02/MLAI_PIPELIENE_EMBEDDING-main)
![GitHub language count](https://img.shields.io/github/languages/count/simonpierreboucher02/MLAI_PIPELIENE_EMBEDDING-main)
![GitHub top language](https://img.shields.io/github/languages/top/simonpierreboucher02/MLAI_PIPELIENE_EMBEDDING-main)
![GitHub last commit](https://img.shields.io/github/last-commit/simonpierreboucher02/MLAI_PIPELIENE_EMBEDDING-main)
![GitHub issues](https://img.shields.io/github/issues/simonpierreboucher02/MLAI_PIPELIENE_EMBEDDING-main)
![GitHub pull requests](https://img.shields.io/github/issues-pr/simonpierreboucher02/MLAI_PIPELIENE_EMBEDDING-main)

## 🎯 Features Overview

- **🧠 AI-Powered Contextualization**: GPT-4 integration for intelligent text enhancement
- **⚡ Concurrent Processing**: Multi-threaded architecture for optimal performance
- **🔑 Multi-API Key Support**: Intelligent rate limiting with API key rotation
- **📝 Smart Text Chunking**: Overlapping chunks with context preservation
- **🎛️ Configurable Pipeline**: YAML-based configuration for easy customization
- **📊 Comprehensive Logging**: Detailed processing workflow tracking

## Table of Contents

- [Description](#description)
- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Requirements](#requirements)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Description

**MLAI_PIPELINE_EMBEDDING** is a modular and efficient pipeline designed to process textual data, contextualize it using OpenAI's GPT-4, and generate high-quality embeddings using OpenAI's Embedding API. This pipeline is ideal for tasks such as document analysis, semantic search, and machine learning model training.

The pipeline handles:

- **Text Chunking**: Splitting large documents into manageable chunks with overlapping regions to maintain context.
- **Contextualization**: Enhancing each chunk with contextual information using GPT-4 to improve the quality and relevance of embeddings.
- **Embedding Generation**: Producing numerical representations (embeddings) of the text for downstream applications.
- **Concurrency and Rate Limiting**: Efficiently managing API rate limits by cycling through multiple OpenAI API keys and supporting concurrent processing.

## Features

- **Modular Design**: Organized into separate modules for easy maintenance and scalability.
- **Configurable Parameters**: All settings are managed via `config.yaml`, allowing for flexible adjustments without modifying the code.
- **Contextualization with GPT-4**: Enhances text chunks with context to produce more meaningful embeddings.
- **Concurrent Processing**: Utilizes multithreading to speed up the embedding generation process.
- **API Key Management**: Supports multiple OpenAI API keys to handle rate limits efficiently.
- **Comprehensive Logging**: Tracks the processing workflow and errors through detailed logs.

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/simonpierreboucher02/MLAI_PIPELIENE_EMBEDDING-main.git
   cd MLAI_PIPELIENE_EMBEDDING-main
   ```

2. **Create a Virtual Environment (Optional but Recommended)**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## Configuration

All configurable parameters are managed via the `config.yaml` file. Ensure that this file is properly set up before running the pipeline.

### `config.yaml`

```yaml
input_dir: "path/to/your/input_txt_files"            # Path to the directory containing .txt files
output_dir: "path/to/your/output_embeddings"        # Path to save generated embeddings and metadata

# OpenAI API Keys. It's recommended to keep them in a separate file for security.
# Uncomment and fill in the keys below if you prefer not to use `api_keys.txt`.
# openai_api_keys:
#   - "your_openai_api_key_1"
#   - "your_openai_api_key_2"
api_keys_file: "api_keys.txt"                        # Path to the file containing your OpenAI API keys

# Text Chunking Parameters
chunk_size: 1200                                     # Maximum number of tokens per chunk
overlap_size: 100                                    # Number of overlapping tokens between chunks
embedding_model: "text-embedding-ada-002"            # OpenAI embedding model to use

# GPT-4 Contextualization Parameters
system_prompt: "You are an expert analyst. The following text is an excerpt from a larger document. Your task is to provide context for the next section by referencing the overall document content. Ensure the context helps in better understanding the excerpt."
llm_max_tokens: 200                                  # Maximum number of tokens for GPT-4's response

# Logging Configuration
logging:
  level: "INFO"                                      # Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL
  format: "%(asctime)s - %(levelname)s - %(message)s"  # Logging format
  file: "embedding_processor.log"                   # Log file name
  stream: true                                       # Whether to also log to the console

# Additional Parameters
verbose: true                                        # Enable verbose logging
max_workers: 10                                     # Maximum number of threads for concurrent processing
```

### `api_keys.txt`

Create a file named `api_keys.txt` in the root directory of the project and add your OpenAI API keys, one per line:

```
your_openai_api_key_1
your_openai_api_key_2
# Add more keys if needed
```

**Security Note:** Ensure that `api_keys.txt` is included in your `.gitignore` to prevent accidental exposure of your API keys.

## Usage

Once the installation and configuration are complete, you can execute the embedding pipeline using the following command:

```bash
python main.py
```

### Execution Steps

1. **Loading Configuration**: The pipeline reads configurations from `config.yaml`, including input/output directories, API keys, chunking parameters, and logging settings.
2. **Processing Files**: It processes each `.txt` file in the specified `input_dir`, splitting them into chunks with overlapping regions.
3. **Contextualization**: Each chunk is sent to GPT-4 for contextualization using the provided system prompt.
4. **Embedding Generation**: The contextualized text is then sent to OpenAI's Embedding API to generate embeddings.
5. **Saving Results**: Embeddings and their associated metadata are saved in the `output_dir` as `embeddings.npy` and `chunks.json`, respectively.
6. **Logging**: The entire process is logged both to the console and the `embedding_processor.log` file.

## File Structure

```
MLAI_PIPELIENE_EMBEDDING/
├── embedding_processor.py      # Main module for processing and generating embeddings
├── main.py                     # Entry point for executing the embedding pipeline
├── config.yaml                 # Configuration file containing all parameters
├── api_keys.txt                # File containing OpenAI API keys (one per line)
├── requirements.txt            # Python dependencies
├── embedding_processor.log     # Log file (generated after execution)
└── README.md                   # Project documentation
```

## Requirements

Ensure you have Python 3.7 or higher installed. The required Python packages are listed in `requirements.txt`:

- `requests`
- `numpy`
- `PyYAML`

Install them using:

```bash
pip install -r requirements.txt
```

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

### Steps for Contributing

1. **Fork the Repository**
2. **Create a Feature Branch**

   ```bash
   git checkout -b feature/YourFeature
   ```

3. **Commit Your Changes**

   ```bash
   git commit -m "Add YourFeature"
   ```

4. **Push to the Branch**

   ```bash
   git push origin feature/YourFeature
   ```

5. **Open a Pull Request**

## License

This project is licensed under the [MIT License](LICENSE). See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or support, please contact:

- **Author:** Simon Pierre Boucher
- **GitHub:** [@simonpierreboucher02](https://github.com/simonpierreboucher02)
- **Repository:** [MLAI_PIPELIENE_EMBEDDING-main](https://github.com/simonpierreboucher02/MLAI_PIPELIENE_EMBEDDING-main)

---

<div align="center">

**⭐ Star this repository if you find it useful!**

© 2023 Simon Pierre Boucher. All rights reserved.

</div>
