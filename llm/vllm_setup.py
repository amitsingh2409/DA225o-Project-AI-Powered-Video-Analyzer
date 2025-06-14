import logging
import requests
import subprocess
import time
import sys
from typing import Optional, List
import threading

from ..config import VLLM_MODEL, VLLM_PORT, VLLM_HOST, VLLM_MAX_MODEL_LEN

logger = logging.getLogger(__name__)


class VLLMServer:
    """Manages a vLLM server for local LLM hosting."""

    def __init__(
        self,
        model_name: str = VLLM_MODEL,
        port: int = VLLM_PORT,
        host: str = VLLM_HOST,
        max_model_len: int = VLLM_MAX_MODEL_LEN,
    ):
        self.model_name = model_name
        self.port = port
        self.host = host
        self.max_model_len = max_model_len
        self.process = None
        self.api_base = f"http://{host}:{port}/v1"
        self._server_thread = None

    def is_server_running(self) -> bool:
        """Check if vLLM server is running by sending a test request."""
        try:
            response = requests.get(f"{self.api_base}/models")
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def _start_server_process(self):
        """Start the vLLM server process."""
        command = [
            sys.executable,
            "-m",
            "vllm.entrypoints.api_server",
            "--model",
            self.model_name,
            "--port",
            str(self.port),
            "--host",
            self.host,
            "--max-model-len",
            str(self.max_model_len),
        ]

        logger.info(f"Starting vLLM server with command: {' '.join(command)}")

        try:
            self.process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )

            # Wait for server to start (up to 60 seconds)
            max_wait = 60
            start_time = time.time()
            while not self.is_server_running() and time.time() - start_time < max_wait:
                time.sleep(1)

                # Check if process has exited
                if self.process.poll() is not None:
                    stdout, stderr = self.process.communicate()
                    logger.error(f"vLLM server process exited with code {self.process.returncode}")
                    logger.error(f"STDOUT: {stdout}")
                    logger.error(f"STDERR: {stderr}")
                    raise RuntimeError(f"vLLM server failed to start: {stderr}")

            if self.is_server_running():
                logger.info(f"vLLM server started successfully at {self.api_base}")
            else:
                logger.error("vLLM server failed to start within timeout")
                self.stop()
                raise TimeoutError("vLLM server failed to start within timeout")

        except Exception as e:
            logger.error(f"Error starting vLLM server: {str(e)}")
            self.stop()
            raise

    def start(self):
        """Start the vLLM server in a separate thread."""
        if self.is_server_running():
            logger.info("vLLM server is already running")
            return

        if self._server_thread is not None and self._server_thread.is_alive():
            logger.warning("Server thread already exists and is alive")
            return

        self._server_thread = threading.Thread(target=self._start_server_process)
        self._server_thread.daemon = True
        self._server_thread.start()

        # Wait for the thread to start the server
        max_wait = 10
        start_time = time.time()
        while not self.is_server_running() and time.time() - start_time < max_wait:
            time.sleep(1)

        if not self.is_server_running():
            logger.warning("vLLM server might still be starting")

    def stop(self):
        """Stop the vLLM server."""
        if self.process is not None:
            logger.info("Stopping vLLM server...")
            try:
                self.process.terminate()
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning("vLLM server didn't terminate, forcing kill")
                self.process.kill()

            self.process = None
            logger.info("vLLM server stopped")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop: Optional[List[str]] = None,
    ) -> str:
        """Generate text using the vLLM API."""
        if not self.is_server_running():
            logger.error("vLLM server is not running")
            raise RuntimeError("vLLM server is not running")

        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }

            if stop:
                payload["stop"] = stop

            response = requests.post(f"{self.api_base}/completions", json=payload)
            response.raise_for_status()

            result = response.json()
            return result["choices"][0]["text"]

        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling vLLM API: {str(e)}")
            raise
