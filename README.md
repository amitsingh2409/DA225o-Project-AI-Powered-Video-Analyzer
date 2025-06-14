# AI-Powered Video Analyzer

## Project Overview
This project implements an AI-powered video analysis system that uses large language models to answer questions about video content. It's developed as part of the DA225o course.

## Features
- Video content analysis using state-of-the-art AI models
- Question-answering capabilities based on video content
- Integration with LangChain for improved context handling

## Installation

### Prerequisites
- Python 3.8 or higher
- Required packages (install via pip):
  ```
  pip install -r requirements.txt
  ```

## Usage

### Starting the LLM Server
```python
# Initialize and start the VLLM server
MODEL_DIR = "/path/to/your/model"
llm = VLLMServer(download_dir=MODEL_DIR)
llm.start()
```

### Using the LangChain Interface
```python
# Create a LangChain interface connected to the LLM
langchain_interface = LangchainInterface(llm)

# Ask questions about video content
QUESTION = "What happens in the video at the 2-minute mark?"
CONTEXT = "Video transcript or extracted features"
answer = langchain_interface.answer_question(QUESTION, CONTEXT)
print(answer)
```
