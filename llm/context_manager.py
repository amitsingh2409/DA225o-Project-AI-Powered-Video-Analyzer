import logging
from ..storage import VideoDatabase, VideoVectorStore

logger = logging.getLogger(__name__)


class ContextManager:
    """Manages context preparation for LLM queries."""

    def __init__(self, db: VideoDatabase, vector_store: VideoVectorStore):
        self.db = db
        self.vector_store = vector_store

    def prepare_context(self, query: str, video_id: str, context_size: int = 5) -> str:
        """Prepare context for a query about a specific video."""
        try:
            # Get relevant segments from vector store
            relevant_segments = self.vector_store.search(
                query=query, video_id=video_id, k=context_size
            )

            if not relevant_segments:
                # Fallback: get transcript from database
                full_transcript, segments = self.db.get_transcript(video_id)
                if not segments:
                    return "No transcript available for this video."

                # Just use first few segments if vector search failed
                relevant_segments = [
                    {
                        "text": segment["text"],
                        "start_time": segment["start"],
                        "end_time": segment["end"],
                    }
                    for segment in segments[:context_size]
                ]

            # Format context
            context_parts = ["Here are relevant parts of the video transcript:"]

            for i, segment in enumerate(relevant_segments):
                timestamp = self._format_timestamp(segment["start_time"])
                context_parts.append(f"[{timestamp}] {segment['text']}")

            return "\n\n".join(context_parts)

        except Exception as e:
            logger.error(f"Error preparing context: {str(e)}")
            return "Error preparing context from video transcript."

    def prepare_navigation_context(self, query: str, video_id: str) -> str:
        """Prepare context specifically for navigation queries."""
        # Get transcript
        full_transcript, segments = self.db.get_transcript(video_id)
        if not segments:
            return "No transcript available for this video."

        # Get relevant segments
        relevant_segments = self.vector_store.search(query=query, video_id=video_id, k=3)

        context_parts = [
            "The user wants to navigate to a specific part of the video.",
            "Your task is to identify the most relevant timestamp based on their query.",
            "Here are some relevant parts of the transcript:",
        ]

        # Add relevant segments to context
        for segment in relevant_segments:
            timestamp = self._format_timestamp(segment["start_time"])
            context_parts.append(f"[{timestamp}] {segment['text']}")

        # Add instructions
        context_parts.append(
            "\nRespond with a timestamp (in the format MM:SS or HH:MM:SS) "
            + "and a brief explanation of why this is the right part of the video."
        )

        return "\n\n".join(context_parts)

    def prepare_summary_context(self, video_id: str) -> str:
        """Prepare context for generating a video summary."""
        # Get transcript
        full_transcript, segments = self.db.get_transcript(video_id)
        if not full_transcript:
            return "No transcript available for this video."

        return (
            f"Here is the transcript of a video that needs to be summarized:\n\n{full_transcript}"
        )

    def prepare_quiz_context(self, video_id: str) -> str:
        """Prepare context for generating a quiz."""
        # Get transcript
        full_transcript, segments = self.db.get_transcript(video_id)
        if not full_transcript:
            return "No transcript available for this video."

        context = [
            "Here is the transcript of a video that you need to create a quiz for:",
            full_transcript,
            "\nCreate a quiz with 5 questions based on the video content.",
            "For each question, provide 4 possible answers with one correct answer.",
            "Format the output as a JSON array of objects with these fields: question, options (array), correctAnswerIndex",
        ]

        return "\n\n".join(context)

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        """Convert seconds to MM:SS format."""
        minutes, seconds = divmod(int(seconds), 60)
        return f"{minutes}:{seconds:02d}"
