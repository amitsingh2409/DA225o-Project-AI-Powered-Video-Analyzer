import logging
from typing import Dict, List, Any

from ..llm.context_manager import ContextManager
from ..llm.langchain_interface import LangchainInterface
from ..storage import VideoDatabase, cached

logger = logging.getLogger(__name__)


class QuizGenerator:
    """Generates quizzes based on video content."""

    def __init__(
        self, db: VideoDatabase, context_manager: ContextManager, langchain: LangchainInterface
    ):
        self.db = db
        self.context_manager = context_manager
        self.langchain = langchain

    def get_quiz(self, video_id: str, regenerate: bool = False) -> Dict[str, Any]:
        """Get or generate a quiz for a video."""
        logger.info(f"Getting quiz for video {video_id} (regenerate={regenerate})")

        # Check if quiz already exists in database
        if not regenerate:
            existing_quiz = self.db.get_quiz(video_id)
            if existing_quiz:
                logger.info(f"Retrieved existing quiz for video {video_id}")
                return {
                    "video_id": video_id,
                    "questions": existing_quiz,
                    "success": True,
                    "regenerated": False,
                }

        # Generate new quiz
        try:
            return self._generate_quiz(video_id)
        except Exception as e:
            logger.error(f"Error generating quiz: {str(e)}")
            return {"video_id": video_id, "questions": [], "success": False, "error": str(e)}

    @cached
    def _generate_quiz(self, video_id: str) -> Dict[str, Any]:
        """Generate a quiz for a video."""
        logger.info(f"Generating new quiz for video {video_id}")

        try:
            # Prepare context for quiz generation
            context = self.context_manager.prepare_quiz_context(video_id)

            # Generate quiz using LLM
            quiz_data = self.langchain.generate_quiz(context)

            # Validate quiz data
            validated_quiz = self._validate_quiz_data(quiz_data)

            # Save quiz to database
            self.db.save_quiz(video_id, validated_quiz)

            return {
                "video_id": video_id,
                "questions": validated_quiz,
                "success": True,
                "regenerated": True,
            }
        except Exception as e:
            logger.error(f"Error in quiz generation: {str(e)}")
            raise

    def _validate_quiz_data(self, quiz_data: List[Dict]) -> List[Dict]:
        """Validate and clean up quiz data."""
        validated_quiz = []

        for item in quiz_data:
            if not isinstance(item, dict):
                continue

            if "question" not in item or "options" not in item or "correctAnswerIndex" not in item:
                continue

            if not isinstance(item["options"], list) or len(item["options"]) < 2:
                continue

            if not isinstance(item["correctAnswerIndex"], int) or item["correctAnswerIndex"] >= len(
                item["options"]
            ):
                # Fix the index if it's out of bounds
                item["correctAnswerIndex"] = 0

            validated_quiz.append(
                {
                    "question": str(item["question"]),
                    "options": [str(opt) for opt in item["options"]],
                    "correctAnswerIndex": item["correctAnswerIndex"],
                }
            )

        return validated_quiz
