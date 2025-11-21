"""
Tool Calls Validator - Validates LLM model's tool call functionality

Features:
- Concurrent request processing
- Incremental mode (rerun only failed requests)
- Stream and non-stream responses
- Real-time statistics updates
- Support for both chat/completions and completions APIs
"""

import argparse
import asyncio
import hashlib
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import megfile
from jsonschema import ValidationError, validate
from loguru import logger
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
from transformers import AutoTokenizer

DEFAULT_CONCURRENCY = 5
DEFAULT_TIMEOUT = 600
DEFAULT_MAX_RETRIES = 3
DEFAULT_OUTPUT_FILE = "results/results.jsonl"
DEFAULT_SUMMARY_FILE = "results/summary.json"

# Role constants
ROLE_INPUT = "_input"
ROLE_SYSTEM = "system"

# Tool call markers
TOOL_CALLS_BEGIN = "<|tool_calls_section_begin|>"
TOOL_CALLS_END = "<|tool_calls_section_end|>"
TOOL_CALL_BEGIN = "<|tool_call_begin|>"
TOOL_CALL_ARG_BEGIN = "<|tool_call_argument_begin|>"
TOOL_CALL_END = "<|tool_call_end|>"


def extract_tool_call_info(tool_call_rsp: str) -> list[dict[str, Any]]:
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

    tool_calls = []
    for match in re.finditer(func_call_pattern, tool_calls_sections[0], re.DOTALL):
        function_id = match.group("tool_call_id")
        function_args = match.group("function_arguments")

        # Parse function_id: functions.get_weather:0
        try:
            function_name = function_id.split(".")[1].split(":")[0]
        except IndexError:
            logger.warning(f"Unable to parse function_id: {function_id}")
            continue

        tool_calls.append({
            "id": function_id,
            "type": "function",
            "function": {"name": function_name, "arguments": function_args},
        })

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
        api_key: str | None = None,
        concurrency: int = DEFAULT_CONCURRENCY,
        output_file: str = DEFAULT_OUTPUT_FILE,
        summary_file: str = DEFAULT_SUMMARY_FILE,
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        extra_body: dict[str, Any] | None = None,
        incremental: bool = False,
        use_raw_completions: bool = False,
        tokenizer_model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        """
        Initialize validator.

        Args:
            model: Model name
            base_url: API base URL
            api_key: API key (optional, defaults to env var)
            concurrency: Number of concurrent requests
            output_file: Detailed results output file
            summary_file: Aggregated summary output file
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            extra_body: Extra request body parameters
            incremental: Whether to enable incremental mode
            use_raw_completions: Whether to use /v1/completions endpoint
            tokenizer_model: Tokenizer model name for raw completions
            temperature: Generation temperature
            max_tokens: Maximum token count
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
        self.tokenizer_model = tokenizer_model

        self.results: list[dict[str, Any]] = []
        self.finish_reason_stat: dict[str, int] = {}

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

        Returns:
            Self for use in 'async with' statement
        """
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Async context manager exit, cleanup resources.

        Args:
            exc_type: Exception type if an exception was raised
            exc_val: Exception value if an exception was raised
            exc_tb: Exception traceback if an exception was raised

        Returns:
            False to propagate exceptions
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

        Args:
            request: Raw request dictionary

        Returns:
            Processed request dictionary
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

        Args:
            file_path: File path

        Returns:
            List of requests, each containing raw request, prepared request, and hash

        Raises:
            FileNotFoundError: If the test file does not exist
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
        return requests

    def read_result_jsonl(self, file_path: str) -> list[dict[str, Any]]:
        """
        Read result file in JSONL format.

        Args:
            file_path: File path

        Returns:
            List of results
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

    async def send_request(self, request: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """
        Send API request (supports both stream and non-stream).

        Args:
            request: Request parameters dictionary

        Returns:
            Tuple of (status, response dict), status is either "success" or "failed"
        """
        try:
            if request.get("stream", False):
                return await self._handle_stream_request(request)
            else:
                # Non-stream request
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
        self, request: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        """
        Handle stream request.

        Args:
            request: Request parameters dictionary

        Returns:
            Tuple of (status, response dict)
        """
        try:
            # Create stream request
            if not self.use_raw_completions:
                stream = await self.client.chat.completions.create(
                    **request, extra_body=self.extra_body
                )
            else:
                stream = await self.client.completions.create(
                    **request, extra_body=self.extra_body
                )

            # Initialize accumulation variables
            request_id = None
            created = None
            full_content = []
            tool_calls: dict[int, dict[str, Any]] = {}
            finish_reason = None
            usage = None

            # Process stream events
            async for event in stream:
                # Extract metadata
                if hasattr(event, "id") and event.id:
                    request_id = event.id
                if hasattr(event, "created") and event.created:
                    created = event.created

                # Check choices
                if not hasattr(event, "choices") or not event.choices:
                    logger.warning("Empty choices in stream event")
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
            if self.use_raw_completions:
                extracted_tool_calls = extract_tool_call_info(content_text)
                if extracted_tool_calls:
                    tool_calls = dict(enumerate(extracted_tool_calls))
                    finish_reason = "tool_calls"

            # Convert tool_calls to list
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

        Args:
            delta_tool_calls: Delta tool calls list
            tool_calls: Accumulated tool calls dictionary (will be modified)
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
    ) -> dict[str, Any]:
        """
        Process a single request, record duration and status.

        Args:
            prepared_req: Preprocessed request (containing raw, prepared, hash)
            data_index: Data index

        Returns:
            Result dictionary
        """
        async with self.semaphore:
            start_time = time.time()
            status, response = await self.send_request(prepared_req["prepared"])
            duration_ms = int((time.time() - start_time) * 1000)

            finish_reason = None
            tool_calls_valid = None

            # Extract finish_reason and validate tool calls
            if response and "choices" in response and response["choices"]:
                choice = response["choices"][0]
                finish_reason = choice.get("finish_reason")

                # Validate parameters if tool call
                if finish_reason == "tool_calls":
                    tools = prepared_req["raw"].get("tools", [])
                    tool_calls = choice.get("message", {}).get("tool_calls", [])
                    if tool_calls:
                        tool_calls_valid = all(
                            self.validate_tool_call(tc, tools) for tc in tool_calls
                        )

            result = {
                "data_index": data_index,
                "request": prepared_req["prepared"],
                "response": response,
                "status": status,
                "finish_reason": finish_reason,
                "tool_calls_valid": tool_calls_valid,
                "last_run_at": datetime.now().isoformat(),
                "duration_ms": duration_ms,
                "hash": prepared_req["hash"],
            }
            return result

    def validate_tool_call(
        self, tool_call: dict[str, Any], tools: list[dict[str, Any]]
    ) -> bool:
        """
        Validate tool call arguments against JSON Schema.

        Args:
            tool_call: Tool call object
            tools: Available tools list

        Returns:
            Whether validation passed
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
                logger.warning(f"No schema found for tool '{tool_name}'")
                return False

            # Parse arguments (may be string or dict)
            args = tool_call["function"]["arguments"]
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError as e:
                    logger.warning(
                        f"JSON parse failed for tool '{tool_name}' arguments: {e}"
                    )
                    return False

            # Validate using jsonschema
            validate(instance=args, schema=schema)
            return True

        except ValidationError as e:
            logger.warning(
                f"Schema validation failed for tool '{tool_name}': {e.message}"
            )
            return False
        except KeyError as e:
            logger.warning(f"Tool call format error, missing field: {e}")
            return False
        except Exception as e:
            logger.warning(f"Unexpected error during validation: {e}")
            return False

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
        tasks = []
        self.results = []

        for req in all_requests:
            h = req["hash"]
            data_index = req["data_index"]

            # Incremental mode: skip successful requests
            if self.incremental and h in existing_hash_map:
                r = existing_hash_map[h]
                if r.get("status") == "success":
                    self.results.append(r)
                    continue

            tasks.append(self.process_request(req, data_index))

        if not tasks:
            logger.info("All requests already processed successfully, no need to rerun")
            return

        logger.info(f"Preparing to process {len(tasks)} requests")

        # Process all tasks concurrently
        with tqdm_asyncio(total=len(tasks), desc="Processing", unit="req") as pbar:
            for task in asyncio.as_completed(tasks):
                try:
                    res = await task
                    # Update statistics
                    finish_reason = res.get("finish_reason")
                    self.finish_reason_stat[finish_reason] = (
                        self.finish_reason_stat.get(finish_reason, 0) + 1
                    )

                    self.results.append(res)
                    # Save result immediately and update stats
                    await self.save_result_and_update_stats(res)
                except Exception as e:
                    logger.error(f"Task execution failed: {e}")
                finally:
                    pbar.update(1)

        # Final processing: deduplicate and sort results
        await self.deduplicate_and_sort_results()

        # Final summary update
        await self.update_summary_file()

        logger.info(f"Results saved to: {self.output_file}")
        logger.info(f"Summary saved to: {self.summary_file}")

    async def save_result_and_update_stats(self, result: dict[str, Any]) -> None:
        """
        Save single result to file and update statistics in real-time.

        Args:
            result: Result dictionary
        """
        # Write to file
        async with self.file_lock:
            with megfile.smart_open(self.output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

        # Update statistics
        async with self.stats_lock:
            summary = self.compute_summary()
            logger.info(
                f"[Stats] Total: {summary['success_count'] + summary['failure_count']}, "
                f"Success: {summary['success_count']}, "
                f"Failed: {summary['failure_count']}, "
                f"Stop: {summary['finish_stop']}, "
                f"ToolCalls: {summary['finish_tool_calls']}, "
                f"ToolCallValid: {summary['successful_tool_call_count']}, "
                f"ToolCallInvalid: {summary['schema_validation_error_count']}"
            )

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
            logger.info("No results to process")
            return

        logger.info(
            f"Processing {len(all_results)} results for deduplication and sorting"
        )

        # Group by data_index and keep the latest one for each index
        results_by_index: dict[int, dict[str, Any]] = {}
        for result in all_results:
            data_index = result.get("data_index")
            if data_index is None:
                logger.warning(f"Result missing data_index: {result}")
                continue

            last_run_at = result.get("last_run_at")
            if last_run_at is None:
                logger.warning(f"Result missing last_run_at: {result}")
                continue

            # If this index is new, or this result is newer, keep it
            if data_index not in results_by_index:
                results_by_index[data_index] = result
            else:
                existing_last_run = results_by_index[data_index].get("last_run_at")
                if existing_last_run is None or last_run_at > existing_last_run:
                    results_by_index[data_index] = result

        # Convert to list and sort by data_index
        deduplicated_results = list(results_by_index.values())
        deduplicated_results.sort(key=lambda x: x.get("data_index", 0))

        logger.info(
            f"Deduplicated from {len(all_results)} to {len(deduplicated_results)} results"
        )

        # Rewrite the file with deduplicated and sorted results
        async with self.file_lock:
            with megfile.smart_open(self.output_file, "w", encoding="utf-8") as f:
                for result in deduplicated_results:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")

        # Update self.results
        self.results = deduplicated_results

        logger.info(f"Results deduplicated, sorted, and saved to: {self.output_file}")

    async def update_summary_file(self) -> None:
        """
        Update summary file.
        """
        summary = self.compute_summary()
        with megfile.smart_open(self.summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=4)

    def compute_summary(self) -> dict[str, Any]:
        """
        Compute summary statistics from results list (backward compatibility).

        Returns:
            Summary dictionary
        """
        summary = {
            "model": self.model,
            "success_count": 0,
            "failure_count": 0,
            "finish_stop": 0,
            "finish_tool_calls": 0,
            "finish_others": 0,
            "finish_others_detail": {},
            "schema_validation_error_count": 0,
            "successful_tool_call_count": 0,
        }

        for r in self.results:
            status = r.get("status")
            finish_reason = r.get("finish_reason")
            tool_calls_valid = r.get("tool_calls_valid")

            if status == "success":
                summary["success_count"] += 1
            else:
                summary["failure_count"] += 1

            if finish_reason == "stop":
                summary["finish_stop"] += 1
            elif finish_reason == "tool_calls":
                summary["finish_tool_calls"] += 1
                if tool_calls_valid:
                    summary["successful_tool_call_count"] += 1
                else:
                    summary["schema_validation_error_count"] += 1
            elif finish_reason:
                summary["finish_others"] += 1
                summary["finish_others_detail"].setdefault(finish_reason, 0)
                summary["finish_others_detail"][finish_reason] += 1

        self.summary = summary
        return summary


async def main() -> None:
    """
    Main function: parse command-line arguments and execute validation.
    """
    parser = argparse.ArgumentParser(
        description=(
            "LLM Tool Calls Validator\n\n"
            "Validate LLM tool call functionality via HTTP API with concurrency support "
            "and optional incremental re-run.\n"
            "Each line in the test set file must be a complete LLM request body (JSON format)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "file_path",
        help="Test set file path (JSONL format)",
    )

    parser.add_argument(
        "--base-url",
        required=True,
        help="API endpoint URL, e.g., https://api.moonshot.cn/v1",
    )

    parser.add_argument(
        "--api-key", help="API key (can also be set via OPENAI_API_KEY env var)"
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Model name, e.g., kimi-k2-0905-preview",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Generation temperature (overrides request temperature)",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum token count (overrides request max_tokens)",
    )

    parser.add_argument(
        "--extra-body",
        type=str,
        help="Extra request body parameters (JSON string)",
    )

    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"Maximum concurrent requests (default: {DEFAULT_CONCURRENCY})",
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_FILE,
        help=f"Detailed results output file path (default: {DEFAULT_OUTPUT_FILE})",
    )
    parser.add_argument(
        "--summary",
        default=DEFAULT_SUMMARY_FILE,
        help=f"Aggregated summary output file path (default: {DEFAULT_SUMMARY_FILE})",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Request timeout in seconds (default: {DEFAULT_TIMEOUT})",
    )

    parser.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help=f"Number of retries on failure (default: {DEFAULT_MAX_RETRIES})",
    )

    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Incremental mode: only rerun failed or new requests, preserve successful results",
    )

    parser.add_argument(
        "--use_raw_completions",
        action="store_true",
        help="Use /v1/completions endpoint (requires tokenizer)",
    )

    parser.add_argument(
        "--tokenizer-model",
        type=str,
        help="Tokenizer model name for raw completions",
    )

    args = parser.parse_args()

    extra_body = {}
    if args.extra_body:
        try:
            extra_body = json.loads(args.extra_body)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse --extra-body JSON: {e}")
            return

    async with ToolCallsValidator(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        concurrency=args.concurrency,
        output_file=args.output,
        summary_file=args.summary,
        timeout=args.timeout,
        max_retries=args.retries,
        extra_body=extra_body,
        incremental=args.incremental,
        use_raw_completions=args.use_raw_completions,
        tokenizer_model=args.tokenizer_model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    ) as validator:
        await validator.validate_file(args.file_path)


if __name__ == "__main__":
    asyncio.run(main())
