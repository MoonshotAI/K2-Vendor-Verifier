"""
Kimi Vendor Verifier - Tool Calls Validation Tool

A tool for validating LLM model's tool call functionality with concurrent request processing,
incremental mode, and real-time statistics updates.
"""

__version__ = "0.1.0"
__author__ = "Moonshot AI"
__description__ = "Tool Calls Validator for Kimi K2 model evaluation"

from .tool_calls_eval import ToolCallsValidator, compute_hash, extract_tool_call_info

__all__ = [
    "ToolCallsValidator",
    "extract_tool_call_info",
    "compute_hash",
]
