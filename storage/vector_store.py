import os
import logging
from typing import Dict, List, Optional
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from ..config import VECTOR_DB_PATH

logger = logging.getLogger(__name__)


class VideoVectorStore:
    def __init__(self, vector_db_path: str = VECTOR_DB_PATH):
        """Initialize vector store for semantic search of transcript segments."""
        self.vector_db_path = vector_db_path
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"}
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )

        # Create directory if it doesn't exist
        os.makedirs(vector_db_path, exist_ok=True)

        # Try to load existing DB, or create a new one
        try:
            self.db = Chroma(
                persist_directory=vector_db_path, embedding_function=self.embedding_model
            )
            logger.info(f"Loaded vector store from {vector_db_path}")
        except Exception as e:
            logger.warning(f"Could not load existing vector store: {str(e)}")
            logger.info("Creating new vector store")
            self.db = Chroma(
                persist_directory=vector_db_path, embedding_function=self.embedding_model
            )

    def add_transcript(self, video_id: str, transcript: str, segments: List[Dict]) -> None:
        """Add transcript segments to vector store."""
        try:
            # Create documents from segments
            documents = []
            for i, segment in enumerate(segments):
                text = segment["text"]
                metadata = {
                    "video_id": video_id,
                    "segment_id": i,
                    "start_time": segment["start"],
                    "end_time": segment["end"],
                }
                documents.append((text, metadata))

            # Split long documents if needed
            processed_docs = []
            for text, metadata in documents:
                chunks = self.text_splitter.split_text(text)
                for chunk in chunks:
                    processed_docs.append({"page_content": chunk, "metadata": metadata})

            # Add to vector store with video_id as collection name
            self.db.add_documents(processed_docs)
            self.db.persist()
            logger.info(
                f"Added {len(processed_docs)} segments to vector store for video {video_id}"
            )
        except Exception as e:
            logger.error(f"Error adding transcript to vector store: {str(e)}")
            raise

    def search(self, query: str, video_id: Optional[str] = None, k: int = 5) -> List[Dict]:
        """Search for relevant transcript segments using semantic search."""
        try:
            # Prepare filter if video_id is provided
            filter_dict = None
            if video_id:
                filter_dict = {"video_id": video_id}

            # Perform search
            results = self.db.similarity_search_with_score(query=query, k=k, filter=filter_dict)

            # Format results
            processed_results = []
            for doc, score in results:
                processed_results.append(
                    {
                        "text": doc.page_content,
                        "video_id": doc.metadata["video_id"],
                        "start_time": doc.metadata["start_time"],
                        "end_time": doc.metadata["end_time"],
                        "relevance_score": float(score),
                    }
                )

            return processed_results
        except Exception as e:
            logger.error(f"Error searching vector store: {str(e)}")
            return []
