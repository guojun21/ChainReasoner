"""
LLM Client abstraction for ChainReasoner.

Provides a pluggable interface so the LLM backend can be swapped without
changing any business logic.  Current implementation uses the OpenAI-compatible
HTTP API.  For LangStudio migration, implement ``LangStudioLLMClient``.
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Optional

import requests

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class LLMClient(ABC):
    """Abstract LLM client interface.

    Subclass this and implement ``chat`` to plug in a new LLM backend.
    """

    @abstractmethod
    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 256,
        purpose: str = "",
    ) -> str:
        """Send a chat-completion request and return the assistant's reply.

        Parameters
        ----------
        system_prompt : str
            System instruction.
        user_prompt : str
            User message.
        temperature : float
            Sampling temperature.
        max_tokens : int
            Max tokens to generate.
        purpose : str
            Human-readable label used for logging.

        Returns
        -------
        str
            The assistant's text reply, or ``""`` on failure.
        """
        ...


# ---------------------------------------------------------------------------
# OpenAI-compatible implementation (current default)
# ---------------------------------------------------------------------------

class OpenAICompatibleClient(LLMClient):
    """LLM client that talks to any OpenAI-compatible ``/v1/chat/completions`` endpoint."""

    def __init__(
        self,
        api_url: str,
        api_key: str,
        model_id: str,
        max_retries: int = 3,
        timeout: int = 60,
    ):
        self.api_url = api_url
        self.api_key = api_key
        self.model_id = model_id
        self.max_retries = max_retries
        self.timeout = timeout

    # -- factory helpers ---------------------------------------------------

    @classmethod
    def from_config(cls, base_model_cfg: dict) -> Optional["OpenAICompatibleClient"]:
        """Create a client from the ``base_model`` section of config.yaml.

        Returns ``None`` if required fields are missing.
        """
        api_url = base_model_cfg.get("api_url")
        api_key = base_model_cfg.get("api_key")
        model_id = base_model_cfg.get("model_id")
        if not api_url or not api_key or not model_id:
            return None
        return cls(api_url=api_url, api_key=api_key, model_id=model_id)

    # -- core -------------------------------------------------------------

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 256,
        purpose: str = "",
    ) -> str:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        for attempt in range(self.max_retries):
            try:
                logger.info(
                    "LLM call: purpose=%s model=%s temp=%.1f max_tokens=%d "
                    "system_len=%d user_len=%d",
                    purpose or "generic",
                    self.model_id,
                    temperature,
                    max_tokens,
                    len(system_prompt),
                    len(user_prompt),
                )
                resp = requests.post(
                    self.api_url, headers=headers, json=payload, timeout=self.timeout
                )
                resp.raise_for_status()
                body = resp.json()

                if "choices" in body and body["choices"]:
                    return body["choices"][0]["message"]["content"].strip()

                # Rate-limit or empty
                status_str = str(body.get("status", ""))
                msg_str = str(body.get("msg", "")).lower()
                if status_str in ("449", "429") or "rate limit" in msg_str:
                    wait = 5 * (attempt + 1)
                    logger.warning("LLM rate-limited (attempt %d), retry in %ds", attempt + 1, wait)
                    time.sleep(wait)
                    continue

                logger.error("LLM missing choices: %s", str(body)[:400])
                return ""
            except Exception as exc:
                logger.error("LLM error (attempt %d): %s", attempt + 1, exc)
                if attempt < self.max_retries - 1:
                    time.sleep(5 * (attempt + 1))
                    continue
                return ""
        return ""
