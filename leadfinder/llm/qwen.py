"""Qwen structured-output client.

Targets any OpenAI-compatible /chat/completions endpoint (vLLM, llama.cpp,
LM Studio, Ollama's /v1 surface, TGI). The base URL, model and key come from
config or the environment - the prototype never hard-codes an endpoint.

JSON contract:
  * short, single-purpose prompts (entity resolution, query planning, person
    reconciliation, final synthesis) - never one giant research prompt;
  * strict JSON validated with Pydantic;
  * exactly one repair retry carrying the validation error, then give up.
"""

from __future__ import annotations

import json
import re
import threading
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

_JSON_BLOCK = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


class LLMUnavailable(RuntimeError):
    """Raised when the LLM cannot be used at all (not configured / unreachable)."""


class LLMCallFailed(RuntimeError):
    """Raised when a call succeeded at the transport level but not semantically."""


def extract_json(raw: str) -> dict[str, Any]:
    """Pull a JSON object out of a model response.

    Local models habitually wrap JSON in prose or fences, and sometimes append
    a trailing sentence. Tolerate that instead of failing the round.
    """
    if raw is None:
        raise LLMCallFailed("empty response")
    text = raw.strip()

    block = _JSON_BLOCK.search(text)
    if block:
        text = block.group(1).strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None

    if parsed is None:
        # Scan for the first balanced top-level object. A misbehaving model
        # can return megabytes of repeated prose, so cap the window and the
        # number of candidate starts to keep this O(n) and predictable.
        MAX_SCAN_CHARS = 100_000
        MAX_CANDIDATES = 200
        scan_text = text[:MAX_SCAN_CHARS]
        for opener, closer in (("{", "}"), ("[", "]")):
            start = scan_text.find(opener)
            candidates = 0
            while start != -1:
                candidates += 1
                if candidates > MAX_CANDIDATES:
                    break
                depth = 0
                in_string = False
                escape = False
                for index in range(start, len(scan_text)):
                    char = scan_text[index]
                    if escape:
                        escape = False
                        continue
                    if char == "\\":
                        escape = True
                        continue
                    if char == '"':
                        in_string = not in_string
                        continue
                    if in_string:
                        continue
                    if char == opener:
                        depth += 1
                    elif char == closer:
                        depth -= 1
                        if depth == 0:
                            candidate = scan_text[start : index + 1]
                            try:
                                parsed = json.loads(candidate)
                            except json.JSONDecodeError:
                                parsed = None
                            if parsed is not None:
                                break
                if parsed is not None:
                    break
                start = scan_text.find(opener, start + 1)
            if parsed is not None:
                break

    if parsed is None:
        raise LLMCallFailed(f"no parsable JSON in response: {raw[:300]!r}")
    if not isinstance(parsed, dict):
        if isinstance(parsed, list):
            return {"items": parsed}
        raise LLMCallFailed("response JSON is not an object")
    return parsed


class QwenClient:
    """Minimal, dependency-light chat client with Pydantic-validated output."""

    def __init__(self, config, *, timeout: float | None = None) -> None:
        self.config = config
        self.timeout = timeout or config.timeout
        self.calls = 0
        self.failures = 0
        self.total_seconds = 0.0
        self._lock = threading.Lock()
        self._client: httpx.Client | None = None

    # -- lifecycle -------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def _endpoint(self) -> str:
        base = self.config.base_url.rstrip("/")
        if self.config.provider == "ollama" and not base.endswith("/v1"):
            base = f"{base}/v1"
        if not base.endswith("/v1"):
            base = f"{base}/v1"
        return f"{base}/chat/completions"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        key = self.config.api_key.strip()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return headers

    def _payload(self, system: str, user: str) -> dict[str, Any]:
        return {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": False,
        }

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                timeout=self.timeout,
                headers=self._headers(),
                trust_env=False,
            )
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    # -- availability ----------------------------------------------------
    def health(self) -> tuple[bool, str]:
        """Used by `leadfinder doctor` so wiring the API is a one-liner check."""
        if not self.enabled:
            return False, "qwen.base_url / qwen.model not set (set in .env or config.yaml)"
        try:
            client = httpx.Client(timeout=15.0, headers=self._headers(), trust_env=False)
            base = self.config.base_url.rstrip("/")
            if not base.endswith("/v1"):
                base = f"{base}/v1"
            try:
                response = client.get(f"{base}/models")
                if response.status_code >= 400:
                    return False, f"{base}/models returned HTTP {response.status_code}"
                return True, f"reachable at {base} (model: {self.config.model})"
            finally:
                client.close()
        except Exception as exc:
            return False, f"cannot reach {self.config.base_url}: {type(exc).__name__}: {exc}"

    # -- core call -------------------------------------------------------
    def complete(self, system: str, user: str) -> str:
        if not self.enabled:
            raise LLMUnavailable("Qwen is not configured")
        client = self._get_client()
        last_error: Exception | None = None
        attempts = max(1, self.config.max_retries + 1)
        for attempt in range(attempts):
            try:
                response = client.post(
                    self._endpoint(), json=self._payload(system, user)
                )
                if response.status_code >= 400:
                    last_error = LLMCallFailed(
                        f"HTTP {response.status_code}: {response.text[:300]}"
                    )
                    continue
                data = response.json()
                choices = data.get("choices") or []
                if not choices:
                    last_error = LLMCallFailed("response contained no choices")
                    continue
                message = choices[0].get("message") or {}
                content = message.get("content")
                if not content:
                    last_error = LLMCallFailed("message content was empty")
                    continue
                with self._lock:
                    self.calls += 1
                return str(content)
            except LLMCallFailed as exc:
                last_error = exc
            except Exception as exc:
                last_error = exc
        with self._lock:
            self.failures += 1
        raise LLMCallFailed(str(last_error or "unknown error"))

    def structured(
        self,
        system: str,
        user: str,
        schema: type[T],
        *,
        repair: bool = True,
    ) -> T:
        """Call the model and validate the reply against `schema`.

        One repair attempt with the validation error attached, then stop. No
        unbounded repair loops.
        """
        current_user = user
        try:
            raw = self.complete(system, current_user)
        except LLMUnavailable:
            raise
        except LLMCallFailed:
            raise

        try:
            return schema.model_validate(extract_json(raw))
        except (ValidationError, LLMCallFailed) as first_error:
            if not repair:
                raise
            message = _describe_error(first_error)
            current_user = (
                f"{user}\n\n"
                f"YOUR PREVIOUS REPLY COULD NOT BE USED.\n"
                f"Validation error: {message}\n"
                f"Reply again with ONLY the corrected JSON object. "
                f"No prose, no markdown fence."
            )
            try:
                raw = self.complete(system, current_user)
                return schema.model_validate(extract_json(raw))
            except (ValidationError, LLMCallFailed, LLMUnavailable) as second_error:
                with self._lock:
                    self.failures += 1
                raise LLMCallFailed(
                    f"schema validation failed after repair: {_describe_error(second_error)}"
                ) from second_error


def _describe_error(error: Exception) -> str:
    if isinstance(error, ValidationError):
        parts = []
        for item in error.errors()[:5]:
            location = ".".join(str(piece) for piece in item.get("loc", ())) or "<root>"
            parts.append(f"{location}: {item.get('msg', 'invalid')}")
        return "; ".join(parts)
    return str(error)[:400]
