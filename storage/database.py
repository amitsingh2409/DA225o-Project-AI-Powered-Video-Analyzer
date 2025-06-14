import sqlite3
import json
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from ..config import DB_PATH

logger = logging.getLogger(__name__)


class VideoDatabase:
    def __init__(self, db_path: str = DB_PATH):
        """Initialize database connection."""
        self.db_path = db_path
        self._create_tables()

    def _create_tables(self):
        """Create necessary tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Videos table
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS videos (
            video_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            file_path TEXT NOT NULL,
            uploaded_at TIMESTAMP NOT NULL,
            duration REAL,
            file_size REAL,
            thumbnail_path TEXT
        )
        """
        )

        # Transcripts table
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS transcripts (
            transcript_id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id TEXT NOT NULL,
            full_transcript TEXT NOT NULL,
            segments TEXT NOT NULL, -- JSON serialized segments with timestamps
            created_at TIMESTAMP NOT NULL,
            FOREIGN KEY (video_id) REFERENCES videos (video_id)
        )
        """
        )

        # Summaries table
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS summaries (
            summary_id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id TEXT NOT NULL,
            summary TEXT NOT NULL,
            created_at TIMESTAMP NOT NULL,
            FOREIGN KEY (video_id) REFERENCES videos (video_id)
        )
        """
        )

        # Quizzes table
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS quizzes (
            quiz_id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id TEXT NOT NULL,
            quiz_data TEXT NOT NULL, -- JSON serialized quiz questions and answers
            created_at TIMESTAMP NOT NULL,
            FOREIGN KEY (video_id) REFERENCES videos (video_id)
        )
        """
        )

        conn.commit()
        conn.close()

    def add_video(
        self,
        video_id: str,
        title: str,
        file_path: str,
        duration: Optional[float] = None,
        file_size: Optional[float] = None,
        thumbnail_path: Optional[str] = None,
    ) -> str:
        """Add a new video to the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute(
                """INSERT INTO videos (video_id, title, file_path, uploaded_at, duration, file_size, thumbnail_path) 
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (video_id, title, file_path, datetime.now(), duration, file_size, thumbnail_path),
            )
            conn.commit()
            logger.info(f"Added video {title} with ID {video_id}")
            return video_id
        except Exception as e:
            conn.rollback()
            logger.error(f"Error adding video: {str(e)}")
            raise
        finally:
            conn.close()

    def save_transcript(self, video_id: str, full_transcript: str, segments: List[Dict]) -> int:
        """Save transcript for a video."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute(
                """INSERT INTO transcripts (video_id, full_transcript, segments, created_at)
                   VALUES (?, ?, ?, ?)""",
                (video_id, full_transcript, json.dumps(segments), datetime.now()),
            )
            transcript_id = cursor.lastrowid
            conn.commit()
            logger.info(f"Saved transcript for video {video_id}")
            return transcript_id
        except Exception as e:
            conn.rollback()
            logger.error(f"Error saving transcript: {str(e)}")
            raise
        finally:
            conn.close()

    def get_transcript(self, video_id: str) -> Tuple[str, List[Dict]]:
        """Get transcript for a video."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute(
                """SELECT full_transcript, segments FROM transcripts
                   WHERE video_id = ? ORDER BY created_at DESC LIMIT 1""",
                (video_id,),
            )
            row = cursor.fetchone()

            if row:
                full_transcript, segments_json = row
                return full_transcript, json.loads(segments_json)
            else:
                logger.warning(f"No transcript found for video {video_id}")
                return "", []
        except Exception as e:
            logger.error(f"Error retrieving transcript: {str(e)}")
            return "", []
        finally:
            conn.close()

    def save_summary(self, video_id: str, summary: str) -> int:
        """Save summary for a video."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute(
                """INSERT INTO summaries (video_id, summary, created_at)
                   VALUES (?, ?, ?)""",
                (video_id, summary, datetime.now()),
            )
            summary_id = cursor.lastrowid
            conn.commit()
            logger.info(f"Saved summary for video {video_id}")
            return summary_id
        except Exception as e:
            conn.rollback()
            logger.error(f"Error saving summary: {str(e)}")
            raise
        finally:
            conn.close()

    def get_summary(self, video_id: str) -> str:
        """Get summary for a video."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute(
                """SELECT summary FROM summaries
                   WHERE video_id = ? ORDER BY created_at DESC LIMIT 1""",
                (video_id,),
            )
            row = cursor.fetchone()

            if row:
                return row[0]
            else:
                logger.warning(f"No summary found for video {video_id}")
                return ""
        except Exception as e:
            logger.error(f"Error retrieving summary: {str(e)}")
            return ""
        finally:
            conn.close()

    def save_quiz(self, video_id: str, quiz_data: List[Dict]) -> int:
        """Save quiz for a video."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute(
                """INSERT INTO quizzes (video_id, quiz_data, created_at)
                   VALUES (?, ?, ?)""",
                (video_id, json.dumps(quiz_data), datetime.now()),
            )
            quiz_id = cursor.lastrowid
            conn.commit()
            logger.info(f"Saved quiz for video {video_id}")
            return quiz_id
        except Exception as e:
            conn.rollback()
            logger.error(f"Error saving quiz: {str(e)}")
            raise
        finally:
            conn.close()

    def get_quiz(self, video_id: str) -> List[Dict]:
        """Get quiz for a video."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute(
                """SELECT quiz_data FROM quizzes
                   WHERE video_id = ? ORDER BY created_at DESC LIMIT 1""",
                (video_id,),
            )
            row = cursor.fetchone()

            if row:
                return json.loads(row[0])
            else:
                logger.warning(f"No quiz found for video {video_id}")
                return []
        except Exception as e:
            logger.error(f"Error retrieving quiz: {str(e)}")
            return []
        finally:
            conn.close()

    def list_videos(self) -> List[Dict]:
        """Get list of all videos."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        try:
            cursor.execute(
                """SELECT v.video_id, v.title, v.uploaded_at, v.duration, v.thumbnail_path,
                          s.summary
                   FROM videos v
                   LEFT JOIN summaries s ON v.video_id = s.video_id AND 
                          s.created_at = (SELECT MAX(created_at) FROM summaries WHERE video_id = v.video_id)
                   ORDER BY v.uploaded_at DESC"""
            )

            result = []
            for row in cursor.fetchall():
                result.append(
                    {
                        "video_id": row["video_id"],
                        "title": row["title"],
                        "uploaded_at": row["uploaded_at"],
                        "duration": row["duration"],
                        "thumbnail_path": row["thumbnail_path"],
                        "summary": row["summary"] if row["summary"] else "",
                    }
                )

            return result
        except Exception as e:
            logger.error(f"Error listing videos: {str(e)}")
            return []
        finally:
            conn.close()
