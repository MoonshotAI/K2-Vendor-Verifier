"""
Tool Calls Validator - Validates LLM model's tool call functionality

Features:
- Concurrent request processing
- Incremental mode (rerun only failed requests)
- Stream and non-stream responses
- Real-time statistics updates, rich UI output
- Support for both chat/completions and completions APIs
"""

import asyncio
import hashlib
import json
import os

# Suppress transformers warning about missing DL frameworks
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import megfile  # pyright: ignore[reportMissingTypeStubs]
from jsonschema import ValidationError, validate
from loguru import logger
from openai import AsyncOpenAI

# Rich imports for UI
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from transformers import AutoTokenizer

from .models import (
    ComplexityMetrics,
    FunctionCall,
    SummaryStatistics,
    ToolCall,
    ToolStats,
    ValidationResult,
)

DEFAULT_CONCURRENCY = 5
DEFAULT_TIMEOUT = 600
DEFAULT_MAX_RETRIES = 3

# Role constants
ROLE_INPUT = "_input"
ROLE_SYSTEM = "system"

# Tool call markers
TOOL_CALLS_BEGIN = "<|tool_calls_section_begin|>"
TOOL_CALLS_END = "<|tool_calls_section_end|>"
TOOL_CALL_BEGIN = "<|tool_call_begin|>"
TOOL_CALL_ARG_BEGIN = "<|tool_call_argument_begin|>"
TOOL_CALL_END = "<|tool_call_end|>"


class ChunksPerSecondColumn(ProgressColumn):
    """Custom column to display chunks per second rate for streaming requests."""

    def render(self, task):
        """Render the chunks per second based on custom field."""
        # Get total chunks from task fields
        total_chunks = task.fields.get("total_chunks", 0)

        # Calculate chunks per second: total_chunks / elapsed_time
        if task.elapsed and task.elapsed > 0:
            chunks_per_sec = total_chunks / task.elapsed
        else:
            chunks_per_sec = 0.0

        if chunks_per_sec == 0:
            return Text("-- chunks/s", style="progress.data.speed")
        return Text(f"{chunks_per_sec:.1f} chunks/s", style="progress.data.speed")


def extract_tool_call_info(tool_call_rsp: str) -> list[ToolCall]:
    """
    Extract tool call information from raw text response.

    Args:
        tool_call_rsp: Raw model response text

    Returns:
        List of tool calls, each containing id, type, and function fields
    """
    if TOOL_CALLS_BEGIN not in tool_call_rsp:
        return []

    # Extract tool calls section
    section_pattern = rf"{re.escape(TOOL_CALLS_BEGIN)}(.*?){re.escape(TOOL_CALLS_END)}"
    tool_calls_sections = re.findall(section_pattern, tool_call_rsp, re.DOTALL)

    if not tool_calls_sections:
        return []

    # Extract individual tool call details
    func_call_pattern = (
        rf"{re.escape(TOOL_CALL_BEGIN)}\s*"
        r"(?P<tool_call_id>[\w\.]+:\d+)\s*"
        rf"{re.escape(TOOL_CALL_ARG_BEGIN)}\s*"
        r"(?P<function_arguments>.*?)\s*"
        rf"{re.escape(TOOL_CALL_END)}"
    )

    tool_calls: list[ToolCall] = []
    for match in re.finditer(func_call_pattern, tool_calls_sections[0], re.DOTALL):
        function_id = match.group("tool_call_id")
        function_args = match.group("function_arguments")

        # Parse function_id: functions.get_weather:0
        try:
            function_name = function_id.split(".")[1].split(":")[0]
        except IndexError:
            logger.warning(f"Unable to parse function_id: {function_id}")
            continue

        tool_calls.append(
            ToolCall(
                id=function_id,
                type="function",
                function=FunctionCall(name=function_name, arguments=function_args),
            )
        )

    return tool_calls


def compute_hash(obj: dict[str, Any]) -> str:
    """
    Compute stable hash of request object for incremental mode.

    Args:
        obj: Request object dictionary

    Returns:
        MD5 hash string
    """
    serialized = json.dumps(obj, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(serialized.encode("utf-8")).hexdigest()


class ToolCallsValidator:
    """
    Tool Calls Validator.

    Responsibilities:
    1. Send concurrent API requests
    2. Validate tool call arguments against schema
    3. Collect and aggregate results
    4. Support incremental mode to avoid re-running successful requests
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        output_file: str,
        summary_file: str,
        api_key: str | None = None,
        concurrency: int = DEFAULT_CONCURRENCY,
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        extra_body: dict[str, Any] | None = None,
        incremental: bool = False,
        use_raw_completions: bool = False,
        tokenizer_model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        verbose: bool = True,  # Default to verbose as requested
        quiet: bool = False,
        sort_by: str = "none",
        limit: int | None = None,
    ):
        """
        Initialize validator.
        """
        # Validate parameters
        if not model or not model.strip():
            raise ValueError("model cannot be empty")
        if not base_url or not base_url.strip():
            raise ValueError("base_url cannot be empty")
        if concurrency <= 0:
            raise ValueError(f"concurrency must be positive, got {concurrency}")
        if timeout <= 0:
            raise ValueError(f"timeout must be positive, got {timeout}")
        if max_retries < 0:
            raise ValueError(f"max_retries cannot be negative, got {max_retries}")
        if temperature is not None and (temperature < 0 or temperature > 1):
            raise ValueError(f"temperature must be between 0 and 1, got {temperature}")
        if max_tokens is not None and max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")

        self.model = model
        self.base_url = base_url
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.concurrency = concurrency
        self.semaphore = asyncio.Semaphore(concurrency)
        self.timeout = timeout
        self.max_retries = max_retries
        self.extra_body = extra_body or {}
        self.output_file = output_file
        self.summary_file = summary_file
        self.incremental = incremental
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_raw_completions = use_raw_completions
        self.max_tokens = max_tokens
        self.use_raw_completions = use_raw_completions
        self.tokenizer_model = tokenizer_model
        # Default to verbose (detailed panels for all) unless quiet
        self.verbose = verbose
        self.quiet = quiet
        self.sort_by = sort_by
        self.limit = limit

        self.results: list[ValidationResult] = []
        self.finish_reason_stat: dict[str, int] = {}
        self.progress: Progress | None = None
        self.total_chunks: int = 0  # Track total streaming chunks across all requests
        self.chunks_lock = asyncio.Lock()  # Lock for updating chunk count

        # Initialize OpenAI client
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

        # Async locks
        self.file_lock = asyncio.Lock()
        self.stats_lock = asyncio.Lock()

        # Load tokenizer if using raw completions endpoint
        if use_raw_completions:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_model, trust_remote_code=True
            )
        else:
            self.tokenizer = None

        # Initialize rich console
        self.console = Console()

        # Ensure output directory exists
        output_dir = Path(self.output_file).parent
        if output_dir and not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created output directory: {output_dir}")

        # Ensure summary directory exists
        summary_dir = Path(self.summary_file).parent
        if summary_dir and not summary_dir.exists():
            summary_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created summary directory: {summary_dir}")

        # Log configuration
        logger.info(f"Model: {self.model}")
        logger.info(f"Results will be saved to: {self.output_file}")
        logger.info(f"Summary will be saved to: {self.summary_file}")
        logger.info(f"Concurrency: {self.concurrency}")
        endpoint = (
            "/v1/completions" if self.use_raw_completions else "/v1/chat/completions"
        )
        logger.info(f"Request endpoint: {endpoint}")
        if self.incremental:
            logger.info("Incremental mode: enabled")

    async def __aenter__(self):
        """
        Async context manager entry.
        """
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Async context manager exit, cleanup resources.
        """
        try:
            await self.client.close()
            logger.debug("AsyncOpenAI client closed successfully")
        except Exception as e:
            logger.warning(f"Error closing AsyncOpenAI client: {e}")
        return False

    def prepare_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """
        Preprocess request, set model and parameters.
        """
        req = request.copy()

        # Handle special _input role (convert to system)
        if "messages" in req:
            for message in req["messages"]:
                if message.get("role") == ROLE_INPUT:
                    message["role"] = ROLE_SYSTEM

        # Set model
        if self.model:
            req["model"] = self.model

        # Override temperature and max_tokens if specified at initialization
        if self.temperature is not None:
            req["temperature"] = self.temperature
        if self.max_tokens is not None:
            req["max_tokens"] = self.max_tokens

        # Convert messages to prompt if using completions endpoint
        if self.use_raw_completions and self.tokenizer:
            req["prompt"] = self.tokenizer.apply_chat_template(
                req["messages"],
                tokenize=False,
                tools=req.get("tools", None),
                add_generation_prompt=True,
            )
            req.pop("messages")
            if "tools" in req:
                req.pop("tools")

        return req

    def read_jsonl(self, file_path: str) -> list[dict[str, Any]]:
        """
        Read test set file in JSONL format.
        """
        # Check file existence
        if not megfile.smart_exists(file_path):
            raise FileNotFoundError(f"Test file not found: {file_path}")

        requests = []
        with megfile.smart_open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue

                try:
                    raw_req = json.loads(line)
                    prepared_req = self.prepare_request(raw_req)
                    requests.append({
                        "data_index": line_num,
                        "raw": raw_req,
                        "prepared": prepared_req,
                        "hash": compute_hash(prepared_req),
                    })
                except json.JSONDecodeError as e:
                    logger.error(f"JSON parse error at line {line_num}: {e}")
                except Exception as e:
                    logger.error(f"Error processing line {line_num}: {e}")

        logger.info(f"Successfully read {len(requests)} requests")

        # Apply sorting if specified
        if self.sort_by != "none":
            requests = self._sort_requests(requests)

        # Apply limit if specified
        if self.limit is not None and self.limit > 0:
            original_count = len(requests)
            requests = requests[: self.limit]
            logger.info(
                f"Limited dataset from {original_count} to {len(requests)} requests"
            )

        return requests

    def _sort_requests(self, requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Sort requests based on sort_by parameter.

        Args:
            requests: List of request objects

        Returns:
            Sorted list of requests
        """

        def count_messages(req: dict[str, Any]) -> int:
            """Count number of messages in request."""
            messages = req.get("raw", {}).get("messages", [])
            return len(messages) if messages else 0

        def count_available_tools(req: dict[str, Any]) -> int:
            """Count number of available tools in request."""
            tools = req.get("raw", {}).get("tools", [])
            return len(tools) if tools else 0

        if self.sort_by == "tool-calls-desc":
            # Sort by number of available tools (descending)
            requests.sort(key=count_available_tools, reverse=True)
            logger.info("Sorted requests by available tool count (most tools first)")
        elif self.sort_by == "tool-calls-asc":
            # Sort by number of available tools (ascending)
            requests.sort(key=count_available_tools, reverse=False)
            logger.info("Sorted requests by available tool count (fewest tools first)")
        elif self.sort_by == "messages-desc":
            requests.sort(key=count_messages, reverse=True)
            logger.info(
                "Sorted requests by message count (longest conversations first)"
            )
        elif self.sort_by == "messages-asc":
            requests.sort(key=count_messages, reverse=False)
            logger.info(
                "Sorted requests by message count (shortest conversations first)"
            )

        return requests

    def read_result_jsonl(self, file_path: str) -> list[dict[str, Any]]:
        """
        Read result file in JSONL format.
        """
        results = []
        with megfile.smart_open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.error(f"Parse error at line {line_num} in result file: {e}")
        return results

    async def send_request(
        self, request: dict[str, Any], log_cb: Any = None
    ) -> tuple[str, dict[str, Any]]:
        """
        Send API request (supports both stream and non-stream).
        """
        try:
            if request.get("stream", False):
                return await self._handle_stream_request(request, log_cb)
            else:
                # Non-stream request
                if log_cb:
                    log_cb("Waiting for full response...")

                if not self.use_raw_completions:
                    response = await self.client.chat.completions.create(
                        **request, extra_body=self.extra_body
                    )
                else:
                    response = await self.client.completions.create(
                        **request, extra_body=self.extra_body
                    )
                return "success", response.model_dump()
        except Exception as e:
            logger.error(f"Request failed: {e}")
            return "failed", {"error": str(e)}

    async def _handle_stream_request(
        self, request: dict[str, Any], log_cb: Any = None
    ) -> tuple[str, dict[str, Any]]:
        """
        Handle stream request.
        """
        try:
            if log_cb:
                log_cb("Initiating stream...")

            # Create stream request
            if not self.use_raw_completions:
                stream = await self.client.chat.completions.create(
                    **request, extra_body=self.extra_body
                )
            else:
                stream = await self.client.completions.create(
                    **request, extra_body=self.extra_body
                )

            if log_cb:
                log_cb("Stream connected. receiving chunks...")

            # Initialize accumulation variables
            request_id = None
            created = None
            full_content = []
            tool_calls: dict[int, dict[str, Any]] = {}
            finish_reason = None
            usage = None
            chunk_count = 0

            # Process stream events
            async for event in stream:
                chunk_count += 1
                # Update global chunk counter and progress bar in real-time
                async with self.chunks_lock:
                    self.total_chunks += 1
                    # Update progress bar every 10 chunks for smooth real-time feedback
                    if self.progress and chunk_count % 10 == 0:
                        # Find the task and update it with current chunk count
                        for task_id in self.progress.task_ids:
                            self.progress.update(
                                task_id, total_chunks=self.total_chunks, refresh=True
                            )

                if log_cb and chunk_count % 20 == 0:
                    log_cb(f"Streaming... ({chunk_count} chunks received)")
                # Extract metadata
                if hasattr(event, "id") and event.id:
                    request_id = event.id
                if hasattr(event, "created") and event.created:
                    created = event.created

                # Check choices
                if not hasattr(event, "choices") or not event.choices:
                    # logger.warning("Empty choices in stream event")
                    continue

                choice = event.choices[0]

                # Handle delta content (chat.completions)
                if hasattr(choice, "delta") and choice.delta:
                    # Accumulate text content
                    if hasattr(choice.delta, "content") and choice.delta.content:
                        full_content.append(choice.delta.content)

                    # Accumulate tool calls
                    if hasattr(choice.delta, "tool_calls") and choice.delta.tool_calls:
                        self._accumulate_tool_calls(choice.delta.tool_calls, tool_calls)

                # Handle text content (completions)
                elif hasattr(choice, "text") and choice.text:
                    full_content.append(choice.text)

                # Extract finish_reason
                if hasattr(choice, "finish_reason") and choice.finish_reason:
                    finish_reason = choice.finish_reason

                # Extract usage
                if hasattr(choice, "usage") and choice.usage:
                    usage = choice.usage

            # Extract tool calls from text if using completions endpoint
            content_text = "".join(full_content)

            # DEBUG OUTPUT: Show raw response and tool call detection logic
            original_finish_reason = finish_reason  # Save original before modification

            if self.use_raw_completions:
                # Show we're in raw completions mode
                if log_cb:
                    log_cb(
                        "⚙️  Using raw completions mode - checking for tool call markers in response"
                    )

                # Show raw content preview
                content_preview = (
                    content_text[:500] if len(content_text) > 500 else content_text
                )
                if log_cb:
                    log_cb(
                        f"📄 Raw Response Content (first 500 chars):\n{content_preview}"
                    )

                # Check for tool call markers
                has_markers = TOOL_CALLS_BEGIN in content_text
                if log_cb:
                    log_cb(f"🔍 Tool call markers detected: {has_markers}")
                    if has_markers:
                        log_cb(f"   ✓ Found '{TOOL_CALLS_BEGIN}' in response")
                    else:
                        log_cb(f"   ✗ No '{TOOL_CALLS_BEGIN}' found in response")

                # Extract tool calls
                extracted_tool_calls = extract_tool_call_info(content_text)

                if log_cb:
                    log_cb(
                        f"🔧 extract_tool_call_info() returned: {len(extracted_tool_calls) if extracted_tool_calls else 0} tool calls"
                    )

                if extracted_tool_calls:
                    tool_calls = {
                        i: tc.__dict__ for i, tc in enumerate(extracted_tool_calls)
                    }
                    finish_reason = "tool_calls"

                    if log_cb:
                        log_cb("✅ Tool calls extracted successfully:")
                        for i, tc in enumerate(extracted_tool_calls):
                            log_cb(f"   Tool {i}: {tc.function.name}")
                        log_cb(
                            f"📝 Updated finish_reason: '{original_finish_reason}' → '{finish_reason}'"
                        )
                else:
                    if log_cb:
                        log_cb("⚠️  No tool calls extracted from response")
                        log_cb(f"📝 finish_reason remains: '{finish_reason}'")
            else:
                # Show we're in chat completions mode
                if log_cb:
                    log_cb(
                        "⚙️  Using chat/completions mode - tool calls should be structured"
                    )
                    if tool_calls:
                        log_cb(
                            f"✅ Received {len(tool_calls)} structured tool calls from API"
                        )
                    else:
                        log_cb("ℹ️  No structured tool calls in API response")

            # Convert tool_calls to list of dicts for response format
            # entries in tool_calls.values() are already dicts
            tool_calls_list = list(tool_calls.values()) if tool_calls else None

            # Construct response
            response = {
                "id": request_id,
                "object": "chat.completion",
                "created": created,
                "model": request.get("model", ""),
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": content_text,
                            "tool_calls": tool_calls_list,
                        },
                        "finish_reason": finish_reason or "stop",
                    }
                ],
                "usage": usage,
            }
            return "success", response
        except Exception as e:
            logger.error(f"Stream request handling failed: {e}")
            return "failed", {"error": str(e)}

    def _accumulate_tool_calls(
        self, delta_tool_calls: list[Any], tool_calls: dict[int, dict[str, Any]]
    ) -> None:
        """
        Accumulate tool call information from stream response.
        """
        for tc in delta_tool_calls:
            idx = tc.index if tc.index is not None else 0

            # Initialize tool call
            if idx not in tool_calls:
                tool_calls[idx] = {
                    "id": tc.id if hasattr(tc, "id") else None,
                    "type": tc.type if hasattr(tc, "type") else "function",
                    "function": {"name": "", "arguments": ""},
                }

            # Accumulate function information
            if hasattr(tc, "function") and tc.function:
                if hasattr(tc.function, "name") and tc.function.name:
                    tool_calls[idx]["function"]["name"] = tc.function.name
                if hasattr(tc.function, "arguments") and tc.function.arguments:
                    tool_calls[idx]["function"]["arguments"] += tc.function.arguments

    async def process_request(
        self, prepared_req: dict[str, Any], data_index: int
    ) -> ValidationResult:
        """
        Process a single request, record duration and status.

        Args:
            prepared_req: Preprocessed request (containing raw, prepared, hash)
            data_index: Data index

        Returns:
            Result dictionary
        """

        def live_log(msg: str):
            if self.verbose and not self.quiet:
                if self.progress:
                    self.progress.console.print(msg)
                else:
                    self.console.print(msg)

        # Live status: Queued
        live_log(f"[dim]#{data_index} Queued...[/]")

        async with self.semaphore:
            # Live status: Sending
            live_log(f"[cyan]#{data_index} Sending Request...[/]")

            start_time = time.time()

            # Define scoped callback that prefixes ID
            def request_logger(msg):
                live_log(f"[dim]#{data_index} {msg}[/]")

            status, response = await self.send_request(
                prepared_req["prepared"], log_cb=request_logger
            )
            duration_ms = int((time.time() - start_time) * 1000)

            # Live status: Received
            live_log(f"[blue]#{data_index} Received Response ({duration_ms}ms)[/]")

            finish_reason: str | None = None
            tool_calls_valid: bool | None = None
            validation_errors: list[str] = []

            # Extract finish_reason and validate tool calls
            if response and "choices" in response and response["choices"]:
                choice = response["choices"][0]
                finish_reason = choice.get("finish_reason")

                # Validate parameters if tool call
                if finish_reason == "tool_calls":
                    # Handle case where tools might be explicitly None
                    tools = prepared_req["raw"].get("tools") or []
                    # Handle case where tool_calls might be explicitly None
                    tool_calls = choice.get("message", {}).get("tool_calls") or []
                    if tool_calls:
                        # Live status: Validating
                        live_log(
                            f"[yellow]#{data_index} Validating {len(tool_calls)} tool calls...[/]"
                        )

                        results_and_errors = [
                            self.validate_tool_call(tc, tools) for tc in tool_calls
                        ]
                        tool_calls_valid = all(r[0] for r in results_and_errors)
                        validation_errors = [r[1] for r in results_and_errors if r[1]]

            return ValidationResult(
                data_index=data_index,
                request=prepared_req["prepared"],
                response=response,
                status=status,
                finish_reason=finish_reason,
                tool_calls_valid=tool_calls_valid,
                validation_errors=validation_errors,
                last_run_at=datetime.now().isoformat(),
                duration_ms=duration_ms,
                hash=prepared_req["hash"],
            )

    def validate_tool_call(
        self, tool_call: dict[str, Any], tools: list[dict[str, Any]]
    ) -> tuple[bool, str | None]:
        """
        Validate tool call arguments against JSON Schema.

        Args:
            tool_call: Tool call object
            tools: Available tools list

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            tool_name = tool_call["function"]["name"]

            # Find corresponding tool schema
            schema = next(
                (
                    t["function"]["parameters"]
                    for t in tools
                    if t["function"]["name"] == tool_name
                ),
                None,
            )

            if not schema:
                msg = f"No schema found for tool '{tool_name}'"
                logger.warning(msg)
                return False, msg

            # Parse arguments (may be string or dict)
            args = tool_call["function"]["arguments"]
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError as e:
                    msg = f"JSON parse failed for tool '{tool_name}' arguments: {e}"
                    logger.warning(msg)
                    return False, msg

            # Validate using jsonschema
            validate(instance=args, schema=schema)
            return True, None

        except ValidationError as e:
            # e.message contains the specific validation error
            msg = f"Schema validation failed for tool '{tool_name}': {e.message}"
            # Also e.instance is useful but message is usually enough
            logger.warning(msg)
            return False, msg
        except KeyError as e:
            msg = f"Tool call format error, missing field: {e}"
            logger.warning(msg)
            return False, msg
        except Exception as e:
            msg = f"Unexpected error during validation: {e}"
            logger.warning(msg)
            return False, msg

    async def validate_file(self, file_path: str) -> None:
        """
        Validate all requests from test file.

        Args:
            file_path: Test set file path (JSONL format)
        """
        # Read all requests
        all_requests = self.read_jsonl(file_path)
        if not all_requests:
            logger.warning("Test set is empty, no requests to process")
            return

        existing_hash_map = {}

        # Incremental mode: load existing results
        if self.incremental and megfile.smart_exists(self.output_file):
            existing_results = self.read_result_jsonl(self.output_file)
            for r in existing_results:
                existing_hash_map[r["hash"]] = r
            logger.info(
                f"Incremental mode: loaded {len(existing_results)} existing results"
            )
        else:
            # Non-incremental mode: clear output file with lock protection
            async with self.file_lock:
                with megfile.smart_open(self.output_file, "w", encoding="utf-8"):
                    pass
            logger.info(f"Initialized output file: {self.output_file}")

        # Initialize summary file
        await self.update_summary_file()

        # Prepare tasks to process
        self.results = []

        # First pass: load existing successful results in incremental mode
        for req in all_requests:
            h = req["hash"]
            data_index = req["data_index"]

            # Incremental mode: skip successful requests
            if self.incremental and h in existing_hash_map:
                r = existing_hash_map[h]
                if isinstance(r, dict) and r.get("status") == "success":
                    # Convert dict to ValidationResult for consistency
                    validation_result = ValidationResult(
                        data_index=r.get("data_index", 0),
                        request=r.get("request", {}),
                        response=r.get("response"),
                        status=r.get("status", ""),
                        finish_reason=r.get("finish_reason"),
                        tool_calls_valid=r.get("tool_calls_valid"),
                        validation_errors=r.get("validation_errors", []),
                        last_run_at=r.get("last_run_at", ""),
                        duration_ms=r.get("duration_ms", 0),
                        hash=r.get("hash", ""),
                    )
                    self.results.append(validation_result)

        # Initialize rich progress object (assigned to self for access in process_request)
        # Note: We use a custom RequestsPerSecondColumn that calculates rate ourselves
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            ChunksPerSecondColumn(),  # Custom column that calculates chunks/s
            console=self.console,
        )

        # Second pass: build list of coroutines for concurrent processing
        tasks = []
        for req in all_requests:
            h = req["hash"]
            data_index = req["data_index"]

            # Incremental mode: skip successful requests
            if self.incremental and h in existing_hash_map:
                r = existing_hash_map[h]
                if isinstance(r, dict) and r.get("status") == "success":
                    continue

            # Append the coroutine (not started yet)
            tasks.append(self.process_request(req, data_index))

        if not tasks:
            logger.info("All requests already processed successfully, no need to rerun")
            await self.update_summary_file()
            return

        logger.info(f"Preparing to process {len(tasks)} requests")

        # Process all tasks concurrently
        with self.progress:
            task_id = self.progress.add_task(
                f"[cyan]Processing {len(tasks)} requests...",
                total=len(tasks),
                total_chunks=0,  # Initialize chunks field for real-time updates
            )

            completed_count = 0
            for task in asyncio.as_completed(tasks):
                try:
                    res = await task
                    # Update statistics
                    finish_reason = res.finish_reason
                    if finish_reason is not None:
                        self.finish_reason_stat[finish_reason] = (
                            self.finish_reason_stat.get(finish_reason, 0) + 1
                        )

                    self.results.append(res)
                    completed_count += 1
                    # Save result immediately and update stats/UI (pass completed_count)
                    await self.save_result_and_update_stats(
                        res, self.progress, task_id, completed_count
                    )
                except Exception as e:
                    logger.error(f"Task execution failed: {e}")
                    completed_count += 1
                    # Still update progress even on error
                    self.progress.update(task_id, completed=completed_count)

        # Cleanup progress reference
        self.progress = None

        # Final processing: deduplicate and sort results
        await self.deduplicate_and_sort_results()

        # Final summary update
        await self.update_summary_file()

        # Export failures for re-testing
        await self.export_failures()

        # Print final summary table
        self.print_final_summary()

        # Print tool stats heat map
        self.print_tool_stats()

        logger.info(f"Results saved to: {self.output_file}")
        logger.info(f"Summary saved to: {self.summary_file}")

    async def save_result_and_update_stats(
        self,
        result: ValidationResult,
        progress: Progress | None = None,
        task_id: TaskID | None = None,
        completed_count: int | None = None,
    ) -> None:
        """
        Save single result to file and update statistics in real-time.
        Allows reporting failures to UI.
        """
        # Write to file
        async with self.file_lock:
            with megfile.smart_open(self.output_file, "a", encoding="utf-8") as f:
                # Convert ValidationResult to dict for JSON serialization
                result_dict = {
                    "data_index": result.data_index,
                    "request": result.request,
                    "response": result.response,
                    "status": result.status,
                    "finish_reason": result.finish_reason,
                    "tool_calls_valid": result.tool_calls_valid,
                    "validation_errors": result.validation_errors,
                    "last_run_at": result.last_run_at,
                    "duration_ms": result.duration_ms,
                    "hash": result.hash,
                }
                f.write(json.dumps(result_dict, ensure_ascii=False) + "\n")

        # Update statistics (UI)
        summary = self.compute_summary()

        # Enhanced descriptive logging with clear success/failure indicators
        total_requests = summary.success_count + summary.failure_count
        network_failures = summary.failure_count

        # Build detailed status message
        status_parts = [
            f"📊 [PROGRESS UPDATE #{result.data_index}]",
            f"Total Processed: {total_requests}",
        ]

        # Network/API level failures (these are BAD)
        if network_failures > 0:
            status_parts.append(
                f"❌ Network/API Failures: {network_failures} (These requests failed to complete)"
            )

        # Successful API responses
        status_parts.append(f"✅ Successful API Responses: {summary.success_count}")

        # Break down finish reasons
        if summary.finish_stop > 0:
            status_parts.append(
                f"   📝 Normal Text Responses (finish='stop'): {summary.finish_stop}"
            )

        if summary.finish_tool_calls > 0:
            status_parts.append(
                f"   🔧 Tool Call Attempts (finish='tool_calls'): {summary.finish_tool_calls}"
            )

            # Show validation results for tool calls
            if summary.successful_tool_call_count > 0:
                status_parts.append(
                    f"     ✅ VALID Tool Calls (passed schema validation): {summary.successful_tool_call_count}"
                )

            if summary.schema_validation_error_count > 0:
                status_parts.append(
                    f"     ❌ INVALID Tool Calls (FAILED schema validation): {summary.schema_validation_error_count}"
                )
                status_parts.append(
                    "        ⚠️  Check detailed panels above for validation errors!"
                )

        # Log with newlines for readability
        logger.info("\n" + "\n".join(status_parts))

        if progress and task_id is not None and completed_count is not None:
            count_total = summary.success_count + summary.failure_count
            count_success = summary.success_count
            count_failed = summary.failure_count
            count_stop = summary.finish_stop
            count_tool_calls = summary.finish_tool_calls
            count_valid = summary.successful_tool_call_count
            count_invalid = summary.schema_validation_error_count

            # Show ALL tracked stats like the original output
            desc = (
                f"[cyan]Total: {count_total}[/cyan] | "
                f"[green]Success: {count_success}[/green] | "
                f"[red]Failed: {count_failed}[/red] | "
                f"[yellow]Stop: {count_stop}[/yellow] | "
                f"[magenta]ToolCalls: {count_tool_calls}[/magenta] | "
                f"[green]ToolCallValid: {count_valid}[/green] | "
                f"[red]ToolCallInvalid: {count_invalid}[/red]"
            )
            # Update with description, completed count, and total chunks for chunks/s calculation
            progress.update(
                task_id,
                description=desc,
                completed=completed_count,
                total_chunks=self.total_chunks,
                refresh=True,
            )

        # Print Live Status for EVERY request (unless quiet)
        if not self.quiet:
            self.print_result_summary(result, progress)

        # Report Validation Details (ONLY for successful API responses)
        # Don't show validation panel for network/API failures - there's nothing to validate!
        if result.status == "success":
            # Check if this is a validation failure
            has_failure = bool(
                result.tool_calls_valid is False
                or (result.validation_errors and len(result.validation_errors) > 0)
            )

            # Only show panel if:
            # 1. There was a validation failure, OR
            # 2. There were actual tool calls that were validated (tool_calls_valid is True)
            # DO NOT show panel when tool_calls_valid is None (no tool calls to validate)
            has_tool_calls_to_show = result.tool_calls_valid is not None

            should_show_panel = has_failure or (
                has_tool_calls_to_show and self.verbose and not self.quiet
            )

            if should_show_panel:
                self.print_failure_report(result, progress, is_failure=has_failure)

    def print_result_summary(
        self, result: ValidationResult, progress: Progress | None = None
    ) -> None:
        """
        Print a detailed summary with request/response info for debugging.
        Shows EXPECTED vs ACTUAL for clear debugging.
        """
        # Create a detailed panel for each request
        details = []

        # Request summary - show what tools were provided
        req_tools = result.request.get("tools", [])
        req_tool_names = [t["function"]["name"] for t in req_tools] if req_tools else []

        # Request context
        details.append(Text("━━━ REQUEST ━━━", style="bold cyan"))

        # Show available tools
        if req_tool_names:
            details.append(
                Text(
                    f"Available Tools: {', '.join(req_tool_names)} ({len(req_tools)} tools)",
                    style="cyan",
                )
            )
        else:
            details.append(
                Text(
                    "Available Tools: None",
                    style="dim cyan",
                )
            )

        # Response
        details.append(Text("\n━━━ ACTUAL ━━━", style="bold magenta"))

        # Response summary based on status
        if result.status != "success":
            error_msg = (
                result.response.get("error", "Unknown error")
                if result.response
                else "Unknown error"
            )
            details.append(
                Text(f"❌ API REQUEST FAILED: {error_msg}", style="bold red")
            )
            details.append(
                Text("Result: Network/API error prevented completion", style="red")
            )

        elif result.finish_reason == "stop":
            if result.response and "choices" in result.response:
                content = (
                    result.response["choices"][0].get("message", {}).get("content", "")
                )
                preview = content[:100] + "..." if len(content) > 100 else content
                details.append(
                    Text("Response Type: Text (finish_reason='stop')", style="yellow")
                )
                details.append(Text(f"Content Preview: {preview}", style="dim"))

        elif result.finish_reason == "tool_calls":
            # Tool call response
            details.append(
                Text(
                    "Response Type: Tool Calls (finish_reason='tool_calls')",
                    style="magenta",
                )
            )

            if result.response and "choices" in result.response:
                tcs = result.response["choices"][0]["message"].get("tool_calls") or []
                details.append(Text(f"Tool Calls Made: {len(tcs)}", style="magenta"))

                # Show validation result clearly
                if result.tool_calls_valid is True:
                    details.append(
                        Text(
                            "✅ VALIDATION: All tool calls are VALID",
                            style="bold green",
                        )
                    )
                elif result.tool_calls_valid is False:
                    details.append(
                        Text("❌ VALIDATION: Tool calls are INVALID", style="bold red")
                    )
                    if result.validation_errors:
                        details.append(Text("Validation Errors:", style="red"))
                        for err in result.validation_errors:
                            details.append(Text(f"  • {err}", style="red"))

                # Show each tool call
                for i, tc in enumerate(tcs, 1):
                    func_name = tc.get("function", {}).get("name", "unknown")
                    func_args = tc.get("function", {}).get("arguments", "{}")
                    # Parse args for display
                    try:
                        args_obj = (
                            json.loads(func_args)
                            if isinstance(func_args, str)
                            else func_args
                        )
                        args_str = json.dumps(args_obj, ensure_ascii=False)
                    except Exception:
                        args_str = str(func_args)

                    style = "green" if result.tool_calls_valid else "red"
                    icon = "✓" if result.tool_calls_valid else "✗"
                    details.append(
                        Text(f"  {icon} Tool {i}: {func_name}({args_str})", style=style)
                    )
        else:
            # Other finish reasons
            details.append(
                Text(
                    f"Response Type: finish_reason='{result.finish_reason}'",
                    style="yellow",
                )
            )
            details.append(Text("⚠️  Unexpected finish_reason value", style="yellow"))

        # Duration
        details.append(Text(f"\nDuration: {result.duration_ms}ms", style="dim"))

        # Determine panel style based on result
        is_success = result.status == "success" and result.tool_calls_valid is not False
        border_style = "green" if is_success else "red"
        title_style = "bold green" if is_success else "bold red"

        # Create panel
        panel = Panel(
            "\n".join([str(d) for d in details]),
            title=f"[{title_style}]Request #{result.data_index}[/{title_style}]",
            border_style=border_style,
            padding=(0, 1),
        )

        if progress:
            progress.console.print(panel)
        else:
            self.console.print(panel)

    def print_failure_report(
        self,
        result: ValidationResult,
        progress: Progress | None = None,
        is_failure: bool = True,
    ) -> None:
        """
        Print a detailed report to the console.
        """
        # We extract the tools needed
        tools = result.request.get("tools", [])

        # We extract the model's tool calls
        response_msg = {}
        if (
            result.response
            and "choices" in result.response
            and result.response["choices"]
        ):
            response_msg = result.response["choices"][0].get("message", {})

        tool_calls = response_msg.get("tool_calls") or []

        # Construct details header
        subtitle_text = Text()
        if is_failure:
            if result.validation_errors:
                for err in result.validation_errors:
                    subtitle_text.append(f"❌ {err}\n", style="bold red")
            else:
                subtitle_text.append(
                    "❌ Tool call validation failed (reason unknown)\n",
                    style="bold red",
                )
        else:
            subtitle_text.append("✅ Validation Successful\n", style="bold green")

        # Syntax highlighted JSON for input tools (schema)
        relevant_schemas = []
        called_tool_names = [tc.get("function", {}).get("name") for tc in tool_calls]
        for t in tools:
            if t["function"]["name"] in called_tool_names:
                relevant_schemas.append(t)

        schema_json = json.dumps(relevant_schemas, indent=2, ensure_ascii=False)
        schema_syntax = Syntax(schema_json, "json", theme="monokai", word_wrap=True)

        # Syntax highlighted JSON for actual output
        output_json = json.dumps(tool_calls, indent=2, ensure_ascii=False)
        output_syntax = Syntax(output_json, "json", theme="monokai", word_wrap=True)

        # Create a grid or table for comparison
        grid = Table.grid(expand=True)
        grid.add_column(ratio=1)
        grid.add_column(ratio=1)

        grid.add_row(
            Text("Expected Schema (Relevant Tools)", style="bold cyan"),
            Text("Actual Tool Calls Output", style="bold magenta"),
        )
        grid.add_row(schema_syntax, output_syntax)

        title_style = "bold red" if is_failure else "bold green"
        border_style = "red" if is_failure else "green"
        title_text = "Validation Failure" if is_failure else "Validation Success"

        panel = Panel(
            grid,
            title=f"[{title_style}]{title_text} (Index: {result.data_index})[/]",
            subtitle=subtitle_text,
            border_style=border_style,
            padding=(1, 2),
        )

        if progress:
            progress.console.print(panel)
        else:
            self.console.print(panel)

    def print_final_summary(self) -> None:
        """
        Print the final summary table.
        """
        summary = self.compute_summary()

        table = Table(title="Validation Summary", border_style="blue")
        table.add_column("Metric", style="cyan")
        table.add_column("Count", justify="right", style="green")

        table.add_row(
            "Total Requests", str(summary.success_count + summary.failure_count)
        )
        table.add_row("Successful Requests", str(summary.success_count))
        table.add_row("Failed Requests (Network/API)", str(summary.failure_count))
        table.add_section()
        table.add_row("Finish Reason: stop", str(summary.finish_stop))
        table.add_row("Finish Reason: tool_calls", str(summary.finish_tool_calls))

        for reason, count in summary.finish_others_detail.items():
            table.add_row(f"Finish Reason: {reason}", str(count))

        table.add_section()
        table.add_row("Valid Tool Calls", str(summary.successful_tool_call_count))
        table.add_row(
            "Invalid Tool Calls (Schema Error)",
            f"[red]{summary.schema_validation_error_count}[/red]",
        )

        self.console.print(table)

    async def deduplicate_and_sort_results(self) -> None:
        """
        Deduplicate and sort results by data_index.
        For records with the same data_index, keep the one with the latest last_run_at.
        """
        # Read all results from file
        if not megfile.smart_exists(self.output_file):
            logger.warning(f"Output file does not exist: {self.output_file}")
            return

        all_results = self.read_result_jsonl(self.output_file)
        if not all_results:
            # logger.info("No results to process")
            return

        # logger.info(f"Processing {len(all_results)} results for deduplication and sorting")

        # Group by data_index and keep the latest one for each index
        results_by_index: dict[int, ValidationResult] = {}
        for result_dict in all_results:
            # Convert dict to ValidationResult
            result = ValidationResult(
                data_index=result_dict.get("data_index", 0),
                request=result_dict.get("request", {}),
                response=result_dict.get("response"),
                status=result_dict.get("status", ""),
                finish_reason=result_dict.get("finish_reason"),
                tool_calls_valid=result_dict.get("tool_calls_valid"),
                validation_errors=result_dict.get("validation_errors", []),
                last_run_at=result_dict.get("last_run_at", ""),
                duration_ms=result_dict.get("duration_ms", 0),
                hash=result_dict.get("hash", ""),
            )

            data_index = result.data_index
            last_run_at = result.last_run_at
            if last_run_at is None:
                # logger.warning(f"Result missing last_run_at: {result}")
                continue

            # If this index is new, or this result is newer, keep it
            if data_index not in results_by_index:
                results_by_index[data_index] = result
            else:
                existing_last_run = results_by_index[data_index].last_run_at
                if existing_last_run is None or last_run_at > existing_last_run:
                    results_by_index[data_index] = result

        # Convert to list and sort by data_index
        deduplicated_results = list(results_by_index.values())
        deduplicated_results.sort(key=lambda x: x.data_index)

        # logger.info(f"Deduplicated from {len(all_results)} to {len(deduplicated_results)} results")

        # Rewrite the file with deduplicated and sorted results
        async with self.file_lock:
            with megfile.smart_open(self.output_file, "w", encoding="utf-8") as f:
                for result in deduplicated_results:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")

        # Update self.results
        self.results = deduplicated_results  # This is already List[ValidationResult]

        # logger.info(f"Results deduplicated, sorted, and saved to: {self.output_file}")

    def compute_tool_stats(self) -> dict[str, ToolStats]:
        """
        Compute per-tool statistics.

        Returns:
            Dictionary mapping tool names to ToolStats objects
        """
        tool_stats: dict[str, ToolStats] = {}

        for result in self.results:
            if result.status != "success":
                continue

            # Get actual tool calls
            if not result.response or "choices" not in result.response:
                continue

            tool_calls = (
                result.response["choices"][0].get("message", {}).get("tool_calls")
            )
            if not tool_calls:
                continue

            for tc in tool_calls:
                tool_name = tc.get("function", {}).get("name", "unknown")

                if tool_name not in tool_stats:
                    tool_stats[tool_name] = ToolStats(tool_name=tool_name)

                # Determine if this specific tool call was valid
                if result.tool_calls_valid is True:
                    tool_stats[tool_name].success_count += 1
                else:
                    tool_stats[tool_name].failure_count += 1
                    # Add schema error if present
                    if result.validation_errors:
                        for err in result.validation_errors:
                            if tool_name in err:
                                tool_stats[tool_name].schema_errors.append(err)

        return tool_stats

    def compute_complexity_metrics(self) -> dict[str, ComplexityMetrics]:
        """
        Compute metrics grouped by request complexity.

        Returns:
            Dictionary mapping complexity types to ComplexityMetrics objects
        """
        metrics: dict[str, ComplexityMetrics] = {}

        for result in self.results:
            # Group by message count
            msg_count = len(result.request.get("messages", []))
            if msg_count <= 2:
                msg_key = "1-2 messages"
            elif msg_count <= 5:
                msg_key = "3-5 messages"
            elif msg_count <= 10:
                msg_key = "6-10 messages"
            else:
                msg_key = "11+ messages"

            if msg_key not in metrics:
                metrics[msg_key] = ComplexityMetrics(complexity_type=msg_key)

            metrics[msg_key].request_count += 1
            metrics[msg_key].total_duration_ms += result.duration_ms

        return metrics

    async def export_failures(self) -> None:
        """
        Export failed cases to separate JSONL files for re-testing.
        """
        output_dir = Path(self.output_file).parent

        # Export all failures
        failures = [r for r in self.results if r.status != "success"]
        if failures:
            failures_file = output_dir / "failures.jsonl"
            async with self.file_lock:
                with megfile.smart_open(str(failures_file), "w", encoding="utf-8") as f:
                    for r in failures:
                        # Write the original request (from raw request data)
                        f.write(json.dumps(r.request, ensure_ascii=False) + "\n")
            logger.info(
                f"Exported {len(failures)} network/API failures to {failures_file}"
            )

        # Export schema validation failures
        schema_failures = [r for r in self.results if r.tool_calls_valid is False]
        if schema_failures:
            schema_file = output_dir / "schema_failures.jsonl"
            async with self.file_lock:
                with megfile.smart_open(str(schema_file), "w", encoding="utf-8") as f:
                    for r in schema_failures:
                        f.write(json.dumps(r.request, ensure_ascii=False) + "\n")
            logger.info(
                f"Exported {len(schema_failures)} schema failures to {schema_file}"
            )

    async def update_summary_file(self) -> None:
        """
        Update summary file.
        """
        summary = self.compute_summary()
        # Convert SummaryStatistics to dict for JSON serialization
        summary_dict = {
            "model": summary.model,
            "success_count": summary.success_count,
            "failure_count": summary.failure_count,
            "finish_stop": summary.finish_stop,
            "finish_tool_calls": summary.finish_tool_calls,
            "finish_others": summary.finish_others,
            "finish_others_detail": summary.finish_others_detail,
            "schema_validation_error_count": summary.schema_validation_error_count,
            "successful_tool_call_count": summary.successful_tool_call_count,
        }
        with megfile.smart_open(self.summary_file, "w", encoding="utf-8") as f:
            json.dump(summary_dict, f, ensure_ascii=False, indent=4)

    def compute_summary(self) -> SummaryStatistics:
        """
        Compute summary statistics from results list.

        Returns:
            Summary statistics
        """
        summary = SummaryStatistics(model=self.model)

        for r in self.results:
            status = r.status
            finish_reason = r.finish_reason
            tool_calls_valid = r.tool_calls_valid

            if status == "success":
                summary.success_count += 1
            else:
                summary.failure_count += 1

            if finish_reason == "stop":
                summary.finish_stop += 1
            elif finish_reason == "tool_calls":
                summary.finish_tool_calls += 1
                # Explicitly check True/False, not just truthy/falsy
                # (tool_calls_valid can be None if no tool calls were validated)
                if tool_calls_valid is True:
                    summary.successful_tool_call_count += 1
                elif tool_calls_valid is False:
                    summary.schema_validation_error_count += 1
                # If None, don't count as either (no validation occurred)
            elif finish_reason:
                summary.finish_others += 1
                summary.finish_others_detail.setdefault(finish_reason, 0)
                summary.finish_others_detail[finish_reason] += 1

        self.summary = summary
        return summary

    def print_tool_stats(self) -> None:
        """Print per-tool success rates (heat map)."""
        tool_stats = self.compute_tool_stats()

        if not tool_stats:
            return

        table = Table(title="Tool Call Success Rates", border_style="blue")
        table.add_column("Tool Name", style="cyan")
        table.add_column("Success", justify="right", style="green")
        table.add_column("Failed", justify="right", style="red")
        table.add_column("Success Rate", justify="right")
        table.add_column("", width=20)  # Bar column

        # Sort by success rate
        sorted_tools = sorted(
            tool_stats.values(), key=lambda x: x.success_rate, reverse=True
        )

        for stats in sorted_tools:
            rate = stats.success_rate
            bar_length = int(rate / 5)  # 20 char = 100%
            bar = "█" * bar_length + "░" * (20 - bar_length)

            # Color code the rate
            if rate >= 95:
                rate_style = "green"
            elif rate >= 80:
                rate_style = "yellow"
            else:
                rate_style = "red"

            table.add_row(
                stats.tool_name,
                str(stats.success_count),
                str(stats.failure_count),
                f"[{rate_style}]{rate:.1f}%[/]",
                bar,
            )

        self.console.print(table)


if __name__ == "__main__":
    pass
