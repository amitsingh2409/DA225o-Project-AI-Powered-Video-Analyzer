import os
import time
import logging
import threading
import requests
import subprocess
from typing import Optional, List

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
        download_dir: str = None,
    ):
        self.model_name = model_name
        self.port = port
        self.host = host
        self.max_model_len = max_model_len
        self.process = None
        self.api_base = f"http://{host}:{port}/v1"
        self._server_thread = None

        # Set a default download directory if None is provided
        if download_dir is None:
            self.download_dir = os.path.expanduser("~/.cache/huggingface")
            logger.info(f"No download directory provided, using default: {self.download_dir}")
        else:
            self.download_dir = download_dir

    def is_server_running(self) -> bool:
        """Check if vLLM server is running by sending a test request."""
        try:
            response = requests.get(f"{self.api_base}/models")
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def is_process_alive(self) -> bool:
        """Check if the server process is still running."""
        return self.process is not None and self.process.poll() is None

    def _start_server_process(self):
        """Start the vLLM server process."""
        command = [
            "vllm",
            "serve",
            self.model_name,
            "--port",
            str(self.port),
            "--host",
            self.host,
            "--max-model-len",
            str(self.max_model_len),
            "--download-dir",
            self.download_dir,
            "--load-format",
            "safetensors",
            "--tensor-parallel-size",
            "1",
        ]

        logger.info(f"Starting vLLM server with command: {' '.join(command)}")

        try:
            self.process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )

            # Wait for server to start (up to 60 seconds)
            max_wait = 60 * 20
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

    def wait_until_ready(self, timeout: int = 60 * 20) -> bool:
        """Wait until the server is ready to handle requests.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            bool: True if server is running, False if timed out
        """
        logger.info(f"Waiting for vLLM server to be ready (timeout: {timeout}s)...")
        start_time = time.time()

        while time.time() - start_time < timeout:
            if not self.is_process_alive():
                if self.process is not None:
                    stdout, stderr = self.process.communicate()
                    logger.error(f"vLLM server process exited with code {self.process.returncode}")
                    logger.error(f"STDOUT: {stdout}")
                    logger.error(f"STDERR: {stderr}")
                return False

            if self.is_server_running():
                logger.info(f"vLLM server is now ready at {self.api_base}")
                return True

            time.sleep(2)

        logger.warning(f"Timed out after {timeout}s waiting for vLLM server to become ready")
        return False

    def start(self, wait_ready: bool = True, timeout: int = 60 * 20):
        """Start the vLLM server in a separate thread.

        Args:
            wait_ready: Whether to block until server is ready
            timeout: Maximum time to wait for server readiness in seconds
        """
        if self.is_server_running():
            logger.info("vLLM server is already running")
            return True

        if self._server_thread is not None and self._server_thread.is_alive():
            logger.warning("Server thread already exists and is alive")
            if wait_ready:
                return self.wait_until_ready(timeout)
            return False

        # Launch the server in a thread
        self._server_thread = threading.Thread(target=self._start_server_process)
        self._server_thread.daemon = True
        self._server_thread.start()

        # Initial wait to give process time to start
        time.sleep(5)

        # Check if process has failed to start
        if not self.is_process_alive():
            logger.error("vLLM server process failed to start")
            return False

        # If requested, wait until the server is fully ready
        if wait_ready:
            return self.wait_until_ready(timeout)

        # Otherwise just check if it seems to be starting
        if self.is_process_alive():
            logger.info("vLLM server process is running, waiting for it to become ready")
            return True

        return False

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
        wait_if_starting: bool = True,
        timeout: int = 60,
    ) -> str:
        """Generate text using the vLLM API.

        Args:
            prompt: Input text prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stop: Optional stop sequences
            wait_if_starting: Whether to wait if server is still starting
            timeout: How long to wait for server to be ready

        Returns:
            Generated text string
        """
        # Check if server is running
        if not self.is_server_running():
            # If process is alive but server isn't accepting connections, it might still be loading
            if self.is_process_alive() and wait_if_starting:
                logger.info("vLLM server process is alive but not ready yet, waiting...")
                if not self.wait_until_ready(timeout):
                    raise RuntimeError("vLLM server failed to become ready within timeout")
            else:
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

            logger.debug(f"Sending request to vLLM API: {payload}")
            response = requests.post(f"{self.api_base}/completions", json=payload)
            response.raise_for_status()

            result = response.json()
            return result["choices"][0]["text"]

        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling vLLM API: {str(e)}")
            raise
