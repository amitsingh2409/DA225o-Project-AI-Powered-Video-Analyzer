import logging
from typing import Dict, Any

from ..llm.context_manager import ContextManager
from ..llm.langchain_interface import LangchainInterface
from ..storage import VideoDatabase, cached

logger = logging.getLogger(__name__)


class SummarizationEngine:
    """Handles video summarization."""

    def __init__(
        self, db: VideoDatabase, context_manager: ContextManager, langchain: LangchainInterface
    ):
        self.db = db
        self.context_manager = context_manager
        self.langchain = langchain

    def get_summary(self, video_id: str, regenerate: bool = False) -> Dict[str, Any]:
        """Get or generate a summary for a video."""
        logger.info(f"Getting summary for video {video_id} (regenerate={regenerate})")

        # Check if summary already exists in database
        if not regenerate:
            existing_summary = self.db.get_summary(video_id)
            if existing_summary:
                logger.info(f"Retrieved existing summary for video {video_id}")
                return {
                    "video_id": video_id,
                    "summary": existing_summary,
                    "success": True,
                    "regenerated": False,
                }

        # Generate new summary
        try:
            return self._generate_summary(video_id)
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return {
                "video_id": video_id,
                "summary": "Summary generation failed due to an error.",
                "success": False,
                "error": str(e),
            }

    @cached
    def _generate_summary(self, video_id: str) -> Dict[str, Any]:
        """Generate a summary for a video."""
        logger.info(f"Generating new summary for video {video_id}")

        try:
            # Prepare context for summarization
            context = self.context_manager.prepare_summary_context(video_id)

            # Generate summary using LLM
            summary = self.langchain.generate_summary(context)

            # Save summary to database
            self.db.save_summary(video_id, summary)

            return {"video_id": video_id, "summary": summary, "success": True, "regenerated": True}
        except Exception as e:
            logger.error(f"Error in summary generation: {str(e)}")
            raise
