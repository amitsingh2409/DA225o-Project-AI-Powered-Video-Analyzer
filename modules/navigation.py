import logging
from typing import Dict, Any

from ..llm.context_manager import ContextManager
from ..llm.langchain_interface import LangchainInterface
from ..storage import VideoDatabase, cached
from ..utils.timestamp_parser import TimestampParser

logger = logging.getLogger(__name__)


class NavigationEngine:
    """Handles natural language navigation of videos."""

    def __init__(
        self, db: VideoDatabase, context_manager: ContextManager, langchain: LangchainInterface
    ):
        self.db = db
        self.context_manager = context_manager
        self.langchain = langchain
        self.timestamp_parser = TimestampParser()

    @cached
    def navigate_to_position(self, video_id: str, query: str) -> Dict[str, Any]:
        """Navigate to a position in the video based on natural language query."""
        logger.info(f"Processing navigation query for video {video_id}: {query}")

        try:
            # First, try to directly extract timestamp if query contains one
            direct_timestamps = self.timestamp_parser.extract_timestamps_from_text(query)
            if direct_timestamps:
                timestamp_str, seconds = direct_timestamps[0]
                return {
                    "video_id": video_id,
                    "query": query,
                    "position": seconds,
                    "timestamp": timestamp_str,
                    "reason": f"Navigating to explicitly mentioned timestamp {timestamp_str}",
                    "success": True,
                }

            # If no direct timestamp, prepare context for LLM
            context = self.context_manager.prepare_navigation_context(query, video_id)

            # Get navigation point from LLM
            nav_result = self.langchain.get_navigation_point(query, context)

            # Parse the timestamp
            timestamp_str = nav_result.get("timestamp", "0:00")
            seconds = self.timestamp_parser.parse_timestamp(timestamp_str)

            return {
                "video_id": video_id,
                "query": query,
                "position": seconds,
                "timestamp": timestamp_str,
                "reason": nav_result.get("reason", "No specific reason provided"),
                "success": True,
            }

        except Exception as e:
            logger.error(f"Error in navigation engine: {str(e)}")
            return {
                "video_id": video_id,
                "query": query,
                "position": 0,
                "timestamp": "0:00",
                "reason": f"Error: {str(e)}",
                "success": False,
            }

    def search_transcript(self, video_id: str, search_text: str) -> Dict[str, Any]:
        """Search for text in transcript and return matching segments."""
        try:
            # Get transcript
            full_transcript, segments = self.db.get_transcript(video_id)
            if not segments:
                return {
                    "video_id": video_id,
                    "search_text": search_text,
                    "matches": [],
                    "success": False,
                    "error": "No transcript available for this video",
                }

            # Search for text in segments
            matches = self.timestamp_parser.find_text_in_segments(segments, search_text)

            return {
                "video_id": video_id,
                "search_text": search_text,
                "matches": matches,
                "success": True,
            }

        except Exception as e:
            logger.error(f"Error searching transcript: {str(e)}")
            return {
                "video_id": video_id,
                "search_text": search_text,
                "matches": [],
                "success": False,
                "error": str(e),
            }
