"""
Command-line interface for Kimi Vendor Verifier.

Provides a clean entry point for running the tool calls validation tool.
"""

import argparse
import asyncio
import sys
from pathlib import Path

from .tool_calls_eval import ToolCallsValidator


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        prog="kimi-vendor-verifier",
        description="Tool Calls Validator - Validates LLM model's tool call functionality",
        epilog="Example: kimi-vendor-verifier samples.jsonl --model kimi-k2-0905-preview --base-url https://api.moonshot.cn/v1 --api-key YOUR_API_KEY",
    )

    # Required arguments
    parser.add_argument(
        "test_file",
        type=str,
        help="Path to the test set file in JSONL format",
    )

    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (e.g., kimi-k2-0905-preview)",
    )

    parser.add_argument(
        "--base-url",
        type=str,
        required=True,
        help="API endpoint URL",
    )

    parser.add_argument(
        "--api-key",
        type=str,
        help="API key for authentication (or set OPENAI_API_KEY environment variable)",
    )

    # Optional arguments
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Maximum number of concurrent requests (default: 5)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="results.jsonl",
        help="Path to save detailed results (default: results.jsonl)",
    )

    parser.add_argument(
        "--summary",
        type=str,
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
        help="Extra JSON body as string to merge into each request payload (e.g., '{\"temperature\":0.6}')",
    )

    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Incremental mode to only rerun failed requests",
    )

    parser.add_argument(
        "--use-raw-completions",
        action="store_true",
        help="Use /v1/completions endpoint instead of /v1/chat/completions",
    )

    parser.add_argument(
        "--tokenizer-model",
        type=str,
        help="Tokenizer model name for raw completions",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        help="Generation temperature (0-1)",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        help="Maximum token count",
    )

    return parser


def parse_extra_body(extra_body_str: str | None) -> dict | None:
    """Parse extra body JSON string into dictionary."""
    if not extra_body_str:
        return None

    try:
        import json

        return json.loads(extra_body_str)
    except json.JSONDecodeError as e:
        print(f"Error parsing extra-body JSON: {e}", file=sys.stderr)
        sys.exit(1)


async def main_async(args: argparse.Namespace) -> None:
    """Run the main async logic."""
    # Parse extra body if provided
    extra_body = parse_extra_body(args.extra_body)

    # Create validator
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
        # Run validation
        await validator.validate_file(args.test_file)


def main() -> None:
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args()

    # Validate test file exists
    test_file_path = Path(args.test_file)
    if not test_file_path.exists():
        print(f"Error: Test file not found: {args.test_file}", file=sys.stderr)
        sys.exit(1)

    # Run the async main function
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
