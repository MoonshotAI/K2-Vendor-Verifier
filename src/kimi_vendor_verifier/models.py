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
    validation_errors: list[str] | None = None
    last_run_at: str | None = None
    duration_ms: int = 0
    hash: str = ""


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


@dataclass
class FailureCategory:
    """Categorization of a validation failure."""

    category: str  # "wrong_tool", "missing_tool", "extra_tool", "schema_error"
    description: str
    expected_tools: list[str]
    actual_tools: list[str]
    count: int = 1


@dataclass
class ToolStats:
    """Statistics for a specific tool."""

    tool_name: str
    success_count: int = 0
    failure_count: int = 0
    schema_errors: list[str] = field(default_factory=lambda: [])

    @property
    def total_calls(self) -> int:
        return self.success_count + self.failure_count

    @property
    def success_rate(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return (self.success_count / self.total_calls) * 100


@dataclass
class ComplexityMetrics:
    """Metrics grouped by request complexity."""

    complexity_type: str  # e.g., "1-2 messages", "0 tools"
    request_count: int = 0
    total_duration_ms: int = 0

    @property
    def avg_duration_ms(self) -> int:
        if self.request_count == 0:
            return 0
        return self.total_duration_ms // self.request_count
