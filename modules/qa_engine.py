import logging
from typing import Dict, Any

from ..llm.context_manager import ContextManager
from ..llm.langchain_interface import LangchainInterface
from ..storage import VideoDatabase, cached

logger = logging.getLogger(__name__)


class QAEngine:
    """Handles question answering about video content."""

    def __init__(
        self, db: VideoDatabase, context_manager: ContextManager, langchain: LangchainInterface
    ):
        self.db = db
        self.context_manager = context_manager
        self.langchain = langchain

    @cached
    def answer_question(self, video_id: str, query: str) -> Dict[str, Any]:
        """Answer a question about a specific video."""
        logger.info(f"Processing question for video {video_id}: {query}")

        try:
            # Prepare context from video transcript
            context = self.context_manager.prepare_context(query, video_id)

            # Generate answer using LLM
            answer = self.langchain.answer_question(query, context)

            return {"video_id": video_id, "query": query, "answer": answer, "success": True}

        except Exception as e:
            logger.error(f"Error in QA engine: {str(e)}")
            return {
                "video_id": video_id,
                "query": query,
                "answer": "I'm sorry, I couldn't process your question due to an error.",
                "success": False,
                "error": str(e),
            }
