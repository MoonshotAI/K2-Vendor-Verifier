import argparse
import asyncio
import hashlib
import json
import os
import time
from datetime import datetime
from typing import Optional
from collections import defaultdict

import megfile
from jsonschema import ValidationError, validate
from loguru import logger
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio


def compute_hash(obj: dict) -> str:
    """Compute a stable hash of the request dict."""
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(s.encode("utf-8")).hexdigest()


class ToolCallsValidator:
    """Validator for tool calls."""

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: Optional[str] = None,
        concurrency: int = 5,
        output_file: str = "results.jsonl",
        summary_file: str = "summary.json",
        timeout: int = 600,
        max_retries: int = 3,
        extra_body: Optional[dict] = None,
        incremental: bool = False,
    ):
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

        self.results: list[dict] = []

        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

        logger.info(f"Results will be saved to {self.output_file}")
        logger.info(f"Summary will be saved to {self.summary_file}")

    def prepare_request(self, request: dict) -> dict:
        """Process request messages and set model."""
        req = request.copy()
        if "messages" in req:
            for message in req["messages"]:
                if message.get("role") == "_input":
                    message["role"] = "system"
        if self.model:
            req["model"] = self.model
        return req

    def read_jsonl(self, file_path: str) -> list[dict]:
        """Load and prepare JSONL requests, compute hash."""
        requests = []
        with megfile.smart_open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    raw_req = json.loads(line.strip())
                    prepared_req = self.prepare_request(raw_req)
                    requests.append(
                        {
                            "data_index": line_num,
                            "raw": raw_req,
                            "prepared": prepared_req,
                            "hash": compute_hash(prepared_req),
                        }
                    )
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing line {line_num}: {e}")
        return requests

    def read_result_jsonl(self, file_path: str) -> list[dict]:
        results = []
        with megfile.smart_open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                results.append(json.loads(line))
        return results

    async def send_request(self, request: dict) -> tuple[str, dict]:
        try:
            if request.get("stream", False):
                return await self._handle_stream_request(request)
            else:
                response = await self.client.chat.completions.create(**request, extra_body=self.extra_body)
                return "success", response.model_dump()
        except Exception as e:
            logger.error(f"Request failed: {e}")
            return "failed", {"error": str(e)}

    async def _handle_stream_request(self, request: dict) -> tuple[str, dict]:
        try:
            stream = await self.client.chat.completions.create(**request, extra_body=self.extra_body)

            request_id = None
            created = None
            full_content = []
            tool_calls: dict[int, dict] = {}
            finish_reason = None
            usage = None

            async for event in stream:
                if hasattr(event, 'id') and event.id:
                    request_id = event.id
                if hasattr(event, 'created') and event.created:
                    created = event.created

                if not hasattr(event, 'choices') or not event.choices:
                    logger.warning("Empty choices in stream event")
                    continue

                choice = event.choices[0]

                if hasattr(choice, 'delta') and choice.delta:
                    if hasattr(choice.delta, 'content') and choice.delta.content:
                        full_content.append(choice.delta.content)

                    if hasattr(choice.delta, 'tool_calls') and choice.delta.tool_calls:
                        for tc in choice.delta.tool_calls:
                            idx = tc.index if tc.index is not None else 0

                            if idx not in tool_calls:
                                tool_calls[idx] = {
                                    "id": tc.id,
                                    "type": tc.type,
                                    "function": {"name": "", "arguments": ""},
                                }

                            if hasattr(tc, 'function') and tc.function:
                                if hasattr(tc.function, 'name') and tc.function.name:
                                    tool_calls[idx]["function"]["name"] = tc.function.name
                                if hasattr(tc.function, 'arguments') and tc.function.arguments:
                                    tool_calls[idx]["function"]["arguments"] += tc.function.arguments

                if hasattr(choice, 'finish_reason') and choice.finish_reason:
                    finish_reason = choice.finish_reason

                if hasattr(choice, 'usage') and choice.usage:
                    usage = choice.usage

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
                            "content": "".join(full_content),
                            "tool_calls": (
                                list(tool_calls.values()) if tool_calls else None
                            ),
                        },
                        "finish_reason": finish_reason or "stop",
                    }
                ],
                "usage": usage,
            }
            return "success", response
        except Exception as e:
            logger.error(f"Stream request failed: {e}")
            return "failed", {"error": str(e)}

    async def process_request(self, prepared_req: dict, data_index: int) -> dict:
        """Process a single request, record duration and status."""
        async with self.semaphore:
            start_time = time.time()
            status, response = await self.send_request(prepared_req["prepared"])
            duration_ms = int((time.time() - start_time) * 1000)

            finish_reason = None
            tool_calls_valid = None

            if response and "choices" in response:
                choice = response["choices"][0] if response["choices"] else {}
                finish_reason = choice.get("finish_reason")
                if finish_reason == "tool_calls":
                    tools = prepared_req["prepared"].get("tools", [])
                    tool_calls = choice.get("message", {}).get("tool_calls", [])
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

    def validate_tool_call(self, tool_call: dict, tools: list[dict]) -> bool:
        """Validate tool call arguments against schema."""
        try:
            tool_name = tool_call["function"]["name"]
            schema = next(
                (
                    t["function"]["parameters"]
                    for t in tools
                    if t["function"]["name"] == tool_name
                ),
                None,
            )
            if not schema:
                logger.warning(f"No schema for tool {tool_name}")
                return False
            args = tool_call["function"]["arguments"]
            if isinstance(args, str):
                args = json.loads(args)
            validate(instance=args, schema=schema)
            return True
        except (json.JSONDecodeError, ValidationError) as e:
            logger.warning(f"Schema validation failed: {e}")
            return False
        except Exception as e:
            logger.warning(f"Unexpected validation error: {e}")
            return False

    async def validate_file(self, file_path: str):
        """Validate all requests from a file, supports incremental mode."""
        all_requests = self.read_jsonl(file_path)
        existing_results = []
        existing_hash_map = {}

        if self.incremental and megfile.smart_exists(self.output_file):
            existing_results = self.read_result_jsonl(self.output_file)
            for r in existing_results:
                existing_hash_map[r["hash"]] = r
            logger.info(f"Loaded {len(existing_results)} existing results")

        tasks = []
        self.results = []

        for req in all_requests:
            h = req["hash"]
            data_index = req["data_index"]
            if self.incremental and h in existing_hash_map:
                r = existing_hash_map[h]
                if r.get("status") == "success":
                    self.results.append(r)
                    continue  # skip successful
            tasks.append(self.process_request(req, data_index))

        with tqdm_asyncio(total=len(tasks), desc="Processing", unit="req") as pbar:
            for task in asyncio.as_completed(tasks):
                try:
                    res = await task
                    with megfile.smart_open(self.output_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(res, ensure_ascii=False) + "\n")
                    self.results.append(res)
                except Exception as e:
                    logger.error(f"Task failed: {e}")
                finally:
                    pbar.update(1)

        self.results.sort(key=lambda r: r["data_index"])

        # Compute summary
        self.compute_summary()
        with megfile.smart_open(self.summary_file, "w", encoding="utf-8") as f:
            json.dump(self.summary, f, ensure_ascii=False, indent=4)

        logger.info(f"Results saved to {self.output_file}")
        logger.info(f"Summary saved to {self.summary_file}")

    def compute_summary(self):
        """Compute summary from all results."""
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


async def main():
    parser = argparse.ArgumentParser(
        description="Validate LLM tool calls via HTTP API with concurrency and optional incremental re-run.\n\n"
        "Each line in the JSONL test set must be a complete LLM request body, e.g., including messages and optional tools.\n"
        "Project tip: a typical test set file is named `samples.jsonl` in the repo path."
    )

    parser.add_argument(
        "file_path",
        help=(
            "Path to the test set file in JSONL format.\n"
            "Example line in JSONL:\n"
            '  {"messages":[{"role":"system","content":"You are a helpful assistant."},\n'
            '               {"role":"user","content":"Find info about company X"}],\n'
            '   "tools":[{"type":"function","function":{"name":"search","parameters":{"query":"company X"}}}]}\n\n'
        ),
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Evaluation model name, e.g., kimi-k2-0905-preview",
    )
    parser.add_argument(
        "--base-url",
        required=True,
        help="API endpoint, e.g., https://api.moonshot.cn/v1",
    )
    parser.add_argument(
        "--api-key", help="API key for authentication (or set OPENAI_API_KEY in env)"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Maximum number of concurrent requests (default: 5)",
    )
    parser.add_argument(
        "--output",
        default="results.jsonl",
        help="Path to save detailed results (default: results.jsonl)",
    )
    parser.add_argument(
        "--summary",
        default="summary.json",
        help="Path to save aggregated summary (default: summary.json)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Per-request timeout in seconds (default: 600)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of retries on failure (default: 3)",
    )
    parser.add_argument(
        "--extra-body",
        type=str,
        help=(
            "Extra JSON body as string.\n"
        ),
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help=(
            "Incremental mode: only rerun previously failed or new requests, merge results into existing output file.\n"
            "Existing successful results are preserved, summary will be recalculated."
        ),
    )

    args = parser.parse_args()

    extra_body = {}
    if args.extra_body:
        try:
            extra_body = json.loads(args.extra_body)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON for --extra-body: {e}")
            return

    validator = ToolCallsValidator(
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
    )
    await validator.validate_file(args.file_path)


if __name__ == "__main__":
    asyncio.run(main())
