from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class AICleanClientError(RuntimeError):
    pass


class AICleanTimeoutError(AICleanClientError):
    pass


@dataclass(frozen=True)
class AICleanClientConfig:
    provider: str
    base_url: str
    model: str
    timeout_seconds: float
    temperature: float
    num_predict: int
    format_json: bool
    think: bool


class AICleanClient(Protocol):
    def clean_blocks(self, blocks: list[dict[str, object]], language: str) -> str:
        ...
