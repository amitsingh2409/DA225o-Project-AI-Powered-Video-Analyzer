# AI-Powered Video Analyzer

## Project Overview
This project implements an AI-powered video analysis system that uses large language models to answer questions about video content. It's developed as part of the DA225o course.

## Features
- Video content analysis using state-of-the-art AI models
- Question-answering capabilities based on video content
- Integration with LangChain for improved context handling
- Video summarization
- Automated quiz generation
- Persistent storage of analysis results

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

### Working with the Database
```python
# Initialize the database
from storage.database import VideoDatabase

# Create database with default path from config
db = VideoDatabase()

# Add a video to the database
video_id = db.add_video(
    video_id="unique_id_123",
    title="My Video Title",
    file_path="/path/to/video.mp4",
    duration=360.5,  # in seconds
    file_size=1024000  # in bytes
)

# List all videos
videos = db.list_videos()
```

### Using the QA Engine
```python
# Initialize the QA Engine
from modules.qa_engine import QAEngine
from llm.context_manager import ContextManager

context_manager = ContextManager(db)
qa_engine = QAEngine(db, context_manager, langchain_interface)

# Ask a question about a specific video
response = qa_engine.answer_question(
    video_id="unique_id_123",
    query="What is the main topic discussed in the video?"
)

# Access the answer
if response["success"]:
    print(response["answer"])
else:
    print(f"Error: {response.get('error')}")
```

### Generating Video Summaries
```python
# Initialize the Summarization Engine
from modules.summarization import SummarizationEngine

summarizer = SummarizationEngine(db, context_manager, langchain_interface)

# Get or generate a summary
summary_result = summarizer.get_summary(video_id="unique_id_123")

# Force regeneration of a summary
new_summary = summarizer.get_summary(video_id="unique_id_123", regenerate=True)

# Access the summary
if summary_result["success"]:
    print(summary_result["summary"])
```

### Creating Quizzes
```python
# Initialize the Quiz Generator
from modules.quiz_generator import QuizGenerator

quiz_generator = QuizGenerator(db, context_manager, langchain_interface)

# Get or generate a quiz for a video
quiz_result = quiz_generator.get_quiz(video_id="unique_id_123")

# Force generation of a new quiz
new_quiz = quiz_generator.get_quiz(video_id="unique_id_123", regenerate=True)

# Access the quiz questions
if quiz_result["success"]:
    for i, question in enumerate(quiz_result["questions"]):
        print(f"Q{i+1}: {question['question']}")
        for j, option in enumerate(question['options']):
            print(f"  {j+1}. {option}")
        correct_idx = question['correctAnswerIndex']
        print(f"  Correct answer: {correct_idx + 1}. {question['options'][correct_idx]}")
```
