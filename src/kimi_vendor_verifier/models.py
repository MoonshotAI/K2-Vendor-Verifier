"""
Data models for K2 Vendor Verifier.

This module contains clean dataclasses for type-safe data handling.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FunctionDefinition:
    """Function definition for tool calls."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class ToolDefinition:
    """Tool definition in the request."""

    type: str
    function: FunctionDefinition


@dataclass
class Message:
    """Message in the conversation."""

    role: str
    content: str | None = None
    tool_calls: list[ToolCall] | None = None


@dataclass
class ToolCall:
    """Tool call in a message."""

    id: str
    type: str
    function: FunctionCall


@dataclass
class FunctionCall:
    """Function call details."""

    name: str
    arguments: str | dict[str, Any]


@dataclass
class Choice:
    """Response choice from the model."""

    index: int
    message: Message
    finish_reason: str | None = None
    text: str | None = None  # For completions endpoint
    delta: Message | None = None  # For streaming
    usage: dict[str, Any] | None = None


@dataclass
class Usage:
    """Token usage information."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class ModelResponse:
    """Complete response from the model."""

    id: str | None
    object: str
    created: int | None
    model: str
    choices: list[Choice]
    usage: Usage | None = None


@dataclass
class RequestData:
    """Request data with metadata."""

    data_index: int
    raw: dict[str, Any]
    prepared: dict[str, Any]
    hash: str


@dataclass
class ValidationResult:
    """Result of a single validation request."""

    data_index: int
    request: dict[str, Any]
    response: dict[str, Any] | None
    status: str
    finish_reason: str | None
    tool_calls_valid: bool | None
    last_run_at: str | None
    duration_ms: int
    hash: str


@dataclass
class SummaryStatistics:
    """Summary statistics for all validation results."""

    model: str
    success_count: int = 0
    failure_count: int = 0
    finish_stop: int = 0
    finish_tool_calls: int = 0
    finish_others: int = 0
    finish_others_detail: dict[str, int] = field(default_factory=dict)
    schema_validation_error_count: int = 0
    successful_tool_call_count: int = 0


@dataclass
class ToolCallInfo:
    """Extracted tool call information from text."""

    tool_calls: list[ToolCall]
