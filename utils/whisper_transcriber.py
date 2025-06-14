import os
import whisper
import torch
import ffmpeg
from pathlib import Path
from typing import Dict, List, Tuple
import logging

from ..config import WHISPER_MODEL, TEMP_DIR

logger = logging.getLogger(__name__)


class WhisperTranscriber:
    def __init__(self, model_name: str = WHISPER_MODEL):
        """Initialize Whisper transcription model."""
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading Whisper model '{model_name}' on {self.device}")
        self.model = whisper.load_model(model_name, device=self.device)
        logger.info(f"Whisper model loaded successfully")

    def extract_audio(self, video_path: str) -> str:
        """Extract audio from video file."""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Create temp file for audio
        audio_path = os.path.join(TEMP_DIR, f"{Path(video_path).stem}.mp3")

        try:
            # Extract audio using ffmpeg
            (
                ffmpeg.input(video_path)
                .output(audio_path, acodec="mp3", ac=1, ar="16k")
                .run(quiet=True, overwrite_output=True)
            )
            logger.info(f"Audio extracted to {audio_path}")
            return audio_path
        except Exception as e:
            logger.error(f"Error extracting audio: {str(e)}")
            raise

    def transcribe(self, video_path: str) -> Dict:
        """Transcribe video file and return timestamped transcript."""
        try:
            audio_path = self.extract_audio(video_path)

            # Run whisper on the audio file
            logger.info(f"Transcribing audio: {audio_path}")
            result = self.model.transcribe(
                audio_path, verbose=False, word_timestamps=True, language="en"
            )

            logger.info(f"Transcription complete for {video_path}")
            return result
        except Exception as e:
            logger.error(f"Transcription error: {str(e)}")
            raise

    def get_segments_with_timestamps(self, transcript_data: Dict) -> List[Dict]:
        """Extract segments with timestamps from transcript data."""
        segments = []

        for segment in transcript_data.get("segments", []):
            segments.append(
                {"start": segment["start"], "end": segment["end"], "text": segment["text"].strip()}
            )

        return segments

    def process_video(self, video_path: str) -> Tuple[str, List[Dict]]:
        """Process video file and return full transcript and segments with timestamps."""
        transcript_data = self.transcribe(video_path)
        full_transcript = transcript_data.get("text", "")
        segments = self.get_segments_with_timestamps(transcript_data)

        return full_transcript, segments
