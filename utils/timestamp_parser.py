import re
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class TimestampParser:
    """Parser for handling and searching through timestamped transcripts."""

    @staticmethod
    def format_timestamp(seconds: float) -> str:
        """Convert seconds to HH:MM:SS format."""
        hours, remainder = divmod(int(seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02}:{minutes:02}:{seconds:02}"

    @staticmethod
    def parse_timestamp(timestamp_str: str) -> float:
        """Convert HH:MM:SS format to seconds."""
        try:
            # Handle both HH:MM:SS and MM:SS formats
            parts = timestamp_str.split(":")
            if len(parts) == 3:  # HH:MM:SS
                hours, minutes, seconds = map(int, parts)
                return hours * 3600 + minutes * 60 + seconds
            elif len(parts) == 2:  # MM:SS
                minutes, seconds = map(int, parts)
                return minutes * 60 + seconds
            else:
                raise ValueError(f"Invalid timestamp format: {timestamp_str}")
        except Exception as e:
            logger.error(f"Error parsing timestamp '{timestamp_str}': {str(e)}")
            return 0

    @staticmethod
    def find_text_in_segments(segments: List[Dict], query: str) -> List[Dict]:
        """Find segments containing the query text."""
        results = []
        query = query.lower()

        for segment in segments:
            if query in segment["text"].lower():
                results.append(
                    {
                        "start": segment["start"],
                        "end": segment["end"],
                        "text": segment["text"],
                        "timestamp": TimestampParser.format_timestamp(segment["start"]),
                    }
                )

        return results

    @staticmethod
    def segment_to_navigation_point(segment: Dict) -> Dict:
        """Convert a transcript segment to a navigation point."""
        return {
            "position": segment["start"],
            "display_time": TimestampParser.format_timestamp(segment["start"]),
            "text": segment["text"],
        }

    @staticmethod
    def extract_timestamps_from_text(text: str) -> List[Tuple[str, float]]:
        """Extract timestamps mentioned in text (like "at 5:30" or "around 1:20:15")."""
        # Match HH:MM:SS or MM:SS patterns
        timestamp_pattern = r"(\d{1,2}):(\d{2})(?::(\d{2}))?"
        matches = re.finditer(timestamp_pattern, text)

        results = []
        for match in matches:
            ts_str = match.group(0)
            ts_seconds = TimestampParser.parse_timestamp(ts_str)
            results.append((ts_str, ts_seconds))

        return results
