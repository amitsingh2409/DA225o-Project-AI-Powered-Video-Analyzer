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
            relevant_segments = self.vector_store.search(
                query=query, video_id=video_id, k=context_size
            )

            if not relevant_segments:
                full_transcript, segments = self.db.get_transcript(video_id)
                if not segments:
                    return "No transcript available for this video."

                relevant_segments = [
                    {
                        "text": segment["text"],
                        "start_time": segment["start"],
                        "end_time": segment["end"],
                    }
                    for segment in segments[:context_size]
                ]

            context_parts = [
                "Here are relevant parts of the video transcript that address the user's question:",
                f"User question: \"{query}\"\n",
            ]

            for i, segment in enumerate(relevant_segments):
                timestamp = self._format_timestamp(segment["start_time"])
                context_parts.append(f"[{timestamp}] {segment['text']}")

            context_parts.append(
                "\nBased on these transcript segments, please provide a clear, concise answer to the user's question. "
                "Reference specific timestamps when appropriate using the [MM:SS] format. "
                "If the provided segments don't contain enough information to answer the question fully, "
                "acknowledge this limitation in your response."
            )

            return "\n\n".join(context_parts)

        except Exception as e:
            logger.error(f"Error preparing context: {str(e)}")
            return "Error preparing context from video transcript."

    def prepare_navigation_context(self, query: str, video_id: str) -> str:
        """Prepare context specifically for navigation queries."""
        full_transcript, segments = self.db.get_transcript(video_id)
        if not segments:
            return "No transcript available for this video."

        relevant_segments = self.vector_store.search(query=query, video_id=video_id, k=3)

        context_parts = [
            "The user wants to navigate to a specific part of the video with this request:",
            f"\"{query}\"",
            "\nYour task is to identify the most relevant timestamp based on their request.",
            "Here are some relevant parts of the transcript:",
        ]

        for segment in relevant_segments:
            timestamp = self._format_timestamp(segment["start_time"])
            context_parts.append(f"[{timestamp}] {segment['text']}")

        context_parts.append(
            "\nRespond with the following format:"
            "\n1. The exact timestamp (in the format MM:SS or HH:MM:SS) that best matches the user's request"
            "\n2. A brief explanation (1-2 sentences) of why this is the right part of the video"
            "\n3. If you're uncertain about the exact timestamp, state your confidence level and suggest an alternative approach"
        )

        return "\n\n".join(context_parts)

    def prepare_summary_context(self, video_id: str) -> str:
        """Prepare context for generating a video summary."""
        full_transcript, segments = self.db.get_transcript(video_id)
        if not full_transcript:
            return "No transcript available for this video."

        return (
            "Your task is to create a comprehensive summary of the following video transcript:\n\n"
            f"{full_transcript}\n\n"
            "Please structure your summary as follows:\n"
            "1. A brief overview (2-3 sentences) describing the main topic\n"
            "2. 3-5 key points or main ideas covered in the video\n"
            "3. A concise conclusion\n\n"
            "Keep the entire summary under 250 words while capturing the essential content and flow of the video."
        )

    def prepare_quiz_context(self, video_id: str) -> str:
        """Prepare context for generating a quiz."""
        full_transcript, segments = self.db.get_transcript(video_id)
        if not full_transcript:
            return "No transcript available for this video."

        context = [
            "Create a quiz based on the following video transcript:",
            full_transcript,
            "\nGenerate a balanced quiz with these specifications:",
            "- 5 questions of mixed difficulty (2 easy, 2 medium, 1 challenging)",
            "- Questions should cover different parts of the video, not just the beginning",
            "- Include a mix of factual recall and conceptual understanding questions",
            "- For each question, provide 4 plausible answer options with exactly one correct answer",
            "- Make incorrect options realistic and plausible to test true understanding",
            "\nFormat the output as a JSON array of objects with these fields:",
            "- question: The quiz question text",
            "- options: Array of 4 possible answers",
            "- correctAnswerIndex: Index (0-3) of the correct answer",
            "- difficulty: String indicating difficulty level ('easy', 'medium', or 'challenging')",
        ]

        return "\n\n".join(context)

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        """Convert seconds to MM:SS format."""
        minutes, seconds = divmod(int(seconds), 60)
        return f"{minutes}:{seconds:02d}"
