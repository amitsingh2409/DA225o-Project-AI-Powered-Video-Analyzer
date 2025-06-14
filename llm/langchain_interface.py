import json
import logging
from typing import Dict, List
from .vllm_setup import VLLMServer

logger = logging.getLogger(__name__)


class LangchainInterface:
    """Interface for using LLM capabilities via vLLM server."""

    def __init__(self, llm_server: VLLMServer):
        """
        Initialize the LangchainInterface.

        Args:
            llm_server: An initialized and started VLLMServer instance
        """
        self.llm = llm_server

    def answer_question(self, query: str, context: str) -> str:
        """
        Answer a question based on provided context.

        Args:
            query: User's question about the video
            context: Video transcript or other contextual information

        Returns:
            A string containing the answer to the question
        """
        prompt_template = """
        You are a helpful AI assistant who provides information about videos.
        
        {context}
        
        User question: {query}
        
        Provide a helpful answer based on the video content. If the video content doesn't address the question, say so.
        
        Answer:
        """

        prompt = prompt_template.format(context=context, query=query)

        try:
            return self.llm.generate(prompt, max_tokens=512, temperature=0.7)
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return "Sorry, I couldn't process your question due to a system error."

    def get_navigation_point(self, query: str, context: str) -> Dict:
        """
        Get navigation point based on query.

        Args:
            query: User's navigation query
            context: Video transcript with timestamps

        Returns:
            Dict containing timestamp and reason
        """
        prompt_template = """
        {context}
        
        User query: "{query}"
        
        Based on the transcript segments, what is the most appropriate timestamp to navigate to?
        Respond with only a JSON object containing: timestamp (string) and reason (string).
        """

        prompt = prompt_template.format(context=context, query=query)

        try:
            response = self.llm.generate(prompt, max_tokens=256, temperature=0.3)

            # Extract JSON from the response
            try:
                # Find JSON-like content in the response
                response = response.strip()
                if response.startswith("```json"):
                    response = response.split("```json")[1].split("```")[0].strip()
                elif response.startswith("```"):
                    response = response.split("```")[1].split("```")[0].strip()

                result = json.loads(response)
                if "timestamp" not in result or "reason" not in result:
                    logger.warning(f"Incomplete navigation response: {result}")
                    return {
                        "timestamp": "00:00:00",
                        "reason": "Could not determine appropriate timestamp",
                    }
                return result
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse navigation JSON: {str(e)} - Raw: {response}")
                return {"timestamp": "00:00:00", "reason": "Could not parse timestamp information"}
        except Exception as e:
            logger.error(f"Error generating navigation point: {str(e)}")
            return {"timestamp": "00:00:00", "reason": "Error processing navigation request"}

    def generate_summary(self, context: str) -> str:
        """
        Generate a summary of a video.

        Args:
            context: Video transcript or other content to summarize

        Returns:
            A concise summary of the video
        """
        prompt_template = """
        {context}
        
        Create a concise summary of this video transcript in about 3-5 sentences.
        Focus on the main points and key insights.
        
        Summary:
        """

        prompt = prompt_template.format(context=context)

        try:
            return self.llm.generate(prompt, max_tokens=300, temperature=0.5)
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return "Unable to generate summary due to a system error."

    def generate_quiz(self, context: str) -> List[Dict]:
        """
        Generate a quiz based on video content.

        Args:
            context: Video transcript or content

        Returns:
            List of quiz questions with answers
        """
        prompt_template = """
        {context}
        
        Based on the video content, generate 3 multiple-choice quiz questions that test understanding of the key concepts.
        
        For each question:
        1. Provide the question text
        2. Provide 4 possible answers (A, B, C, D)
        3. Indicate the correct answer
        
        Return your response as a JSON array where each item has the format:
        {{"question": "Question text", "options": ["Option A", "Option B", "Option C", "Option D"], "correct": "A"}}
        """

        prompt = prompt_template.format(context=context)

        try:
            response = self.llm.generate(prompt, max_tokens=800, temperature=0.7)

            # Extract JSON from the response
            try:
                # Find JSON-like content in the response
                response = response.strip()
                if response.startswith("```json"):
                    response = response.split("```json")[1].split("```")[0].strip()
                elif response.startswith("```"):
                    response = response.split("```")[1].split("```")[0].strip()

                result = json.loads(response)
                # Validate the structure
                if not isinstance(result, list):
                    logger.warning("Quiz response is not a list")
                    result = []

                # Validate each question
                for item in result:
                    if not all(key in item for key in ["question", "options", "correct"]):
                        logger.warning(f"Invalid quiz question format: {item}")

                return result
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse quiz JSON: {str(e)} - Raw: {response}")
                return []
        except Exception as e:
            logger.error(f"Error generating quiz: {str(e)}")
            return []
