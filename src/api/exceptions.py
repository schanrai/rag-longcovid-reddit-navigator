"""Structured API errors with ``failed_stage`` for the frontend progress UI."""
from __future__ import annotations

from typing import Any


class ApiError(Exception):
    """Raised for user-facing JSON errors (4xx/5xx) with stable shape."""

    __slots__ = ("status_code", "body")

    def __init__(
        self,
        status_code: int,
        *,
        code: str,
        message: str,
        failed_stage: str | None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body: dict[str, Any] = {
            "error": {"code": code, "message": message},
            "failed_stage": failed_stage,
        }
