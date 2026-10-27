"""Model backends.

Every call goes through `Backend.call`, which writes one JSON line per call to
the run's call log (prompt, response, model, usage, timing). That log is the
audit trail: any number in the analysis can be traced back to the exact text
the model saw and produced.

Two backends:
- AnthropicBackend: real calls to the Claude API.
- MockBackend: deterministic placeholder text, for testing the pipeline with no
  API key. Mock output is NOT data. Everything it writes is tagged [MOCK].
"""

from __future__ import annotations

import hashlib
import json
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Reply:
    text: str
    data: dict | None  # parsed JSON when a schema was requested
    stop_reason: str
    usage: dict


class Backend:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self._lock = threading.Lock()

    def call(self, *, role: str, model: str, effort: str, system: str,
             user: str, schema: dict | None = None, max_tokens: int = 4000,
             meta: dict | None = None) -> Reply:
        t0 = time.time()
        reply = self._call(model=model, effort=effort, system=system, user=user,
                           schema=schema, max_tokens=max_tokens, meta=meta or {})
        record = {
            "ts": t0,
            "seconds": round(time.time() - t0, 3),
            "backend": type(self).__name__,
            "role": role,
            "model": model,
            "effort": effort,
            "meta": meta or {},
            "system_sha256": hashlib.sha256(system.encode()).hexdigest(),
            "system": system,
            "user": user,
            "schema": schema,
            "text": reply.text,
            "data": reply.data,
            "stop_reason": reply.stop_reason,
            "usage": reply.usage,
        }
        with self._lock, open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        return reply

    def _call(self, **kw) -> Reply:  # pragma: no cover
        raise NotImplementedError


class AnthropicBackend(Backend):
    def __init__(self, log_path: Path):
        super().__init__(log_path)
        import anthropic  # imported here so mock runs don't need the SDK
        self.client = anthropic.Anthropic()

    def _call(self, *, model, effort, system, user, schema, max_tokens, meta) -> Reply:
        output_config: dict[str, Any] = {"effort": effort}
        if schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": schema}
        resp = self.client.beta.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config=output_config,
            # Transcripts only grow, so the shared prefix caches across turns.
            cache_control={"type": "ephemeral"},
            # On a safety decline, rerun on Anthropic's recommended fallback model.
            fallbacks="default",
            betas=["server-side-fallback-2026-07-01"],
        )
        usage = {
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
            "cache_read_input_tokens": getattr(resp.usage, "cache_read_input_tokens", 0) or 0,
            "cache_creation_input_tokens": getattr(resp.usage, "cache_creation_input_tokens", 0) or 0,
        }
        if resp.stop_reason == "refusal":
            return Reply(text="[REFUSED]", data=None, stop_reason="refusal", usage=usage)
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        data = None
        if schema is not None:
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                data = None
        return Reply(text=text, data=data, stop_reason=resp.stop_reason, usage=usage)


class MockBackend(Backend):
    """Placeholder output so the whole pipeline runs offline.

    Text is neutral filler and structured outputs are random but schema-valid.
    It deliberately carries no agenda signal, so any "effect" in mock plots is
    noise. Never report mock results.
    """

    def __init__(self, log_path: Path, seed: int = 0):
        super().__init__(log_path)
        self.seed = seed

    def _rng(self, system: str, user: str) -> random.Random:
        h = hashlib.sha256(f"{self.seed}|{system}|{user}".encode()).hexdigest()
        return random.Random(int(h[:16], 16))

    def _call(self, *, model, effort, system, user, schema, max_tokens, meta) -> Reply:
        rng = self._rng(system, user)
        usage = {"input_tokens": (len(system) + len(user)) // 4, "output_tokens": 60,
                 "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
        if schema is None:
            phrases = ["I think we should look at the numbers again.",
                       "Good point, and we should also talk to more customers.",
                       "Let's keep the plan simple for now.",
                       "I see risks here, but also an opening."]
            text = "[MOCK] " + " ".join(rng.sample(phrases, 2))
            return Reply(text=text, data=None, stop_reason="end_turn", usage=usage)
        data = _fake_from_schema(schema, rng)
        return Reply(text=json.dumps(data), data=data, stop_reason="end_turn", usage=usage)


def _fake_from_schema(schema: dict, rng: random.Random) -> Any:
    t = schema.get("type")
    if "enum" in schema:
        return rng.choice(schema["enum"])
    if t == "object":
        return {k: _fake_from_schema(v, rng) for k, v in schema.get("properties", {}).items()}
    if t == "array":
        return [_fake_from_schema(schema["items"], rng) for _ in range(2)]
    if t == "integer":
        lo, hi = schema.get("minimum", 1), schema.get("maximum", 7)
        return rng.randint(lo, hi)
    if t == "number":
        lo, hi = schema.get("minimum", 0), schema.get("maximum", 100)
        return round(rng.uniform(lo, hi), 2)
    if t == "boolean":
        return rng.random() < 0.5
    return "[MOCK]"


def make_backend(name: str, log_path: Path, seed: int = 0) -> Backend:
    if name == "mock":
        return MockBackend(log_path, seed)
    if name == "anthropic":
        return AnthropicBackend(log_path)
    raise ValueError(f"unknown backend {name!r}")
