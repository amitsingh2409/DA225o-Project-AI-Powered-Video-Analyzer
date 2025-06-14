import logging
from typing import Dict, List, Optional
import json
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM

from .vllm_setup import VLLMServer

logger = logging.getLogger(__name__)


class LangchainInterface:
    """Interface to LangChain for LLM interactions."""

    def __init__(self, vllm_server: VLLMServer):
        self.vllm_server = vllm_server
        self.llm = self._setup_llm()

    def _setup_llm(self) -> LLM:
        """Setup LLM using vLLM."""

        # Define a custom LLM that uses the vLLM server
        class VLLMWrapper(LLM):
            vllm_instance: VLLMServer

            def __init__(self, vllm_instance):
                super().__init__()
                self.vllm_instance = vllm_instance

            @property
            def _llm_type(self) -> str:
                return "custom_vllm"

            def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
                # Call vLLM and return the generated text
                return self.vllm_instance.generate(prompt, stop=stop)

        return VLLMWrapper(self.vllm_server)

    def answer_question(self, query: str, context: str) -> str:
        """Answer a question based on provided context."""
        prompt_template = """
        You are a helpful AI assistant who provides information about videos.
        
        {context}
        
        User question: {query}
        
        Provide a helpful answer based on the video content. If the video content doesn't address the question, say so.
        
        Answer:
        """

        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "query"])

        chain = LLMChain(llm=self.llm, prompt=prompt)

        try:
            result = chain.run(context=context, query=query)
            return result.strip()
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return "I'm sorry, I encountered an error while processing your question."

    def get_navigation_point(self, query: str, context: str) -> Dict:
        """Get navigation point based on query."""
        prompt_template = """
        {context}
        
        User query: "{query}"
        
        Based on the transcript segments, what is the most appropriate timestamp to navigate to?
        Respond with only a JSON object containing: timestamp (string) and reason (string).
        """

        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "query"])

        chain = LLMChain(llm=self.llm, prompt=prompt)

        try:
            result = chain.run(context=context, query=query)

            # Try to extract JSON from the result
            result = result.strip()
            # Find JSON-like content between { and }
            start_idx = result.find("{")
            end_idx = result.rfind("}")
            if start_idx >= 0 and end_idx > start_idx:
                json_str = result[start_idx : end_idx + 1]
                navigation_data = json.loads(json_str)
                return {
                    "timestamp": navigation_data.get("timestamp", "0:00"),
                    "reason": navigation_data.get("reason", "No specific reason provided"),
                }
            else:
                # Fallback if JSON extraction fails
                logger.warning(f"Failed to extract JSON from LLM response: {result}")
                return {
                    "timestamp": "0:00",
                    "reason": "Could not determine a specific position in the video.",
                }
        except Exception as e:
            logger.error(f"Error generating navigation point: {str(e)}")
            return {"timestamp": "0:00", "reason": "Error processing your navigation request."}

    def generate_summary(self, context: str) -> str:
        """Generate a summary of a video."""
        prompt_template = """
        {context}
        
        Create a concise summary of this video transcript in about 3-5 sentences.
        Focus on the main points and key insights.
        
        Summary:
        """

        prompt = PromptTemplate(template=prompt_template, input_variables=["context"])

        chain = LLMChain(llm=self.llm, prompt=prompt)

        try:
            result = chain.run(context=context)
            return result.strip()
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return "Summary generation failed due to an error."

    def generate_quiz(self, context: str) -> List[Dict]:
        """Generate a quiz based on video content."""
        prompt_template = """
        {context}
        """

        prompt = PromptTemplate(template=prompt_template, input_variables=["context"])

        chain = LLMChain(llm=self.llm, prompt=prompt)

        try:
            result = chain.run(context=context)

            # Try to extract JSON from the result
            result = result.strip()

            # Find JSON-like content between [ and ]
            start_idx = result.find("[")
            end_idx = result.rfind("]")

            if start_idx >= 0 and end_idx > start_idx:
                json_str = result[start_idx : end_idx + 1]
                quiz_data = json.loads(json_str)
                return quiz_data
            else:
                logger.warning(f"Failed to extract JSON quiz data: {result}")
                # Return empty quiz if JSON extraction fails
                return []

        except Exception as e:
            logger.error(f"Error generating quiz: {str(e)}")
            return []
