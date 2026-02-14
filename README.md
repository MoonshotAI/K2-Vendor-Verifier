# K2 Vendor Verifier

## We've updated the evaluation approach for kimi-vendor-verifier. Click [here](https://www.kimi.com/blog/kimi-vendor-verifier.html) for more details.

## What's K2VV

Since the release of the Kimi K2 model, we have received numerous feedback on the precision of Kimi K2 in toolcall. Given that K2 focuses on the agentic loop, the reliability of toolcall is of utmost importance.

We have observed significant differences in the toolcall performance of various open-source solutions and vendors. When selecting a provider, users often prioritize lower latency and cost, but may inadvertently overlook more subtle yet critical differences in model accuracy.

These inconsistencies not only affect user experience but also impact K2's performance in various benchmarking results.
To mitigate these problems, we launch K2 Vendor Verifier to monitor and enhance the quality of all K2 APIs.

We hope K2VV can help ensuring that everyone can access a consistent and high-performing Kimi K2 model.

## K2-thinking  Evaluation Results

**Test Time**: 2025-11-15
- temperature=1.0
- max_tokens=64000

<table>
  <thead>
    <tr>
      <th rowspan="2">Model Name</th>
      <th rowspan="2">Provider</th>
      <th rowspan="2">Api Source</th>
      <th rowspan="2">ToolCall-Trigger Similarity</th>
      <th colspan="3" style="text-align: center;">ToolCall-Schema Accuracy</th>
    </tr>
    <tr>
      <th>count_finish_reason_tool_calls</th>
      <th>count_successful_tool_call</th>
      <th>schema_accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="17">kimi-k2-thinking</td>
      <td><a href="https://platform.moonshot.ai/">MoonshotAI</a></td>
      <td>https://platform.moonshot.ai</td>
      <td>-</td>
      <td>1958</td>
      <td>1958</td>
      <td>100.00%</td>
    </tr>
    <tr>
      <td><a href="https://platform.moonshot.ai/">Moonshot AI Turbo</a></td>
      <td>https://platform.moonshot.ai</td>
      <td rowspan="12">>=73%</td>
      <td>1984</td>
      <td>1984</td>
      <td>100.00%</td>
    </tr>
    <tr>
      <td><a href="https://fireworks.ai/">Fireworks</a></td>
      <td>https://fireworks.ai</td>
      <td>1703</td>
      <td>1703</td>
      <td>100.00%</td>
    </tr>
    <tr>
      <td><a href="https://cloud.infini-ai.com/">InfiniAI</a></td>
      <td>https://cloud.infini-ai.com</td>
      <td>1827</td>
      <td>1825</td>
      <td>99.89%</td>
    </tr>
    <tr>
      <td><a href="https://siliconflow.cn/">SiliconFlow</a></td>
      <td>https://siliconflow.cn</td>
      <td>2119</td>
      <td>2097</td>
      <td>98.96%</td>
    </tr>
    <tr>
      <td><a href="https://openrouter.ai/provider/gmicloud">GMICloud</a></td>
      <td>https://openrouter.ai</td>
      <td>1850</td>
      <td>1775</td>
      <td>95.95%</td>
    </tr>
    <tr>
      <td><a href="https://openrouter.ai/provider/atlas-cloud">AtlasCloud</a></td>
      <td>https://openrouter.ai</td>
      <td>1878</td>
      <td>1798</td>
      <td>95.74%</td>
    </tr>
    <tr>
      <td><a href="https://github.com/sgl-project/sglang">SGLang</a></td>
      <td>https://github.com/sgl-project/sglang</td>
      <td>1874</td>
      <td>1790</td>
      <td>95.52%</td>
    </tr>
    <tr>
      <td><a href="https://github.com/vllm-project/vllm">vLLM</a></td>
      <td>https://github.com/vllm-project/vllm</td>
      <td>2128</td>
      <td>1856</td>
      <td>87.22%</td>
    </tr>
    <tr>
      <td><a href="https://openrouter.ai/provider/parasail">Parasail</a></td>
      <td>https://openrouter.ai</td>
      <td>2108</td>
      <td>1837</td>
      <td>87.14%</td>
    </tr>
    <tr>
      <td><a href="https://openrouter.ai/provider/deepinfra">DeepInfra</a></td>
      <td>https://openrouter.ai</td>
      <td>2071</td>
      <td>1800</td>
      <td>86.91%</td>
    </tr>
    <tr>
      <td><a href="https://openrouter.ai/provider/google-vertex">GoogleVertex</a></td>
      <td>https://openrouter.ai</td>
      <td>1945</td>
      <td>1668</td>
      <td>85.76%</td>
    </tr>
    <tr>
      <td><a href="https://openrouter.ai/provider/together">Together</a></td>
      <td>https://openrouter.ai</td>
      <td>1893</td>
      <td>1602</td>
      <td>84.63%</td>
    </tr>
    <tr>
      <td><a href="https://openrouter.ai/provider/novita">NovitaAI</a></td>
      <td>https://openrouter.ai</td>
      <td>72.22%</td>
      <td>1778</td>
      <td>1715</td>
      <td>96.46%</td>
    </tr>
    <tr>
      <td><a href="https://openrouter.ai/provider/chutes">Chutes</a></td>
      <td>https://openrouter.ai</td>
      <td>68.10%</td>
      <td>3657</td>
      <td>3037</td>
      <td>83.05%</td>
    </tr>
  </tbody>
</table>

##### We ran the official API multiple times to test the fluctuation of `tool_call_f1`. The lowest score was **75.81%**, and the average was **76%**. Given the inherent randomness of the model, we believe that an `tool_call_f1` score above **73%** is acceptable and can be used as a reference.

## K2 0905 Evaluation Results

**Test Time**: 2025-11-15
- temperature=0.6

<table>
  <thead>
    <tr>
      <th rowspan="2">Model Name</th>
      <th rowspan="2">Provider</th>
      <th rowspan="2">Api Source</th>
      <th rowspan="2">ToolCall-Trigger Similarity</th>
      <th colspan="3" style="text-align: center;">ToolCall-Schema Accuracy</th>
    </tr>
    <tr>
      <th>count_finish_reason_tool_calls</th>
      <th>count_successful_tool_call</th>
      <th>schema_accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="16">kimi-k2-0905-preview</td>
      <td><a href="https://platform.moonshot.ai/">MoonshotAI</a></td>
      <td>https://platform.moonshot.ai</td>
      <td>-</td>
      <td>1274</td>
      <td>1274</td>
      <td>100.00%</td>
    </tr>
    <tr>
      <td><a href="https://platform.moonshot.ai/">Moonshot AI Turbo</a></td>
      <td>https://platform.moonshot.ai</td>
      <td rowspan="13">>=80%</td>
      <td>1398</td>
      <td>1398</td>
      <td>100.00%</td>
    </tr>
    <tr>
      <td><a href="https://openrouter.ai/provider/deepinfra">DeepInfra</a></td>
      <td>https://openrouter.ai</td>
      <td>1365</td>
      <td>1365</td>
      <td>100.00%</td>
    </tr>
    <tr>
      <td><a href="https://openrouter.ai/provider/fireworks">Fireworks</a></td>
      <td>https://openrouter.ai</td>
      <td>1453</td>
      <td>1453</td>
      <td>100.00%</td>
    </tr>
    <tr>
      <td><a href="https://cloud.infini-ai.com/">Infinigence</a></td>
      <td>https://cloud.infini-ai.com</td>
      <td>1257</td>
      <td>1257</td>
      <td>100.00%</td>
    </tr>
    <tr>
      <td><a href="https://openrouter.ai/provider/novita">NovitaAI</a></td>
      <td>https://openrouter.ai</td>
      <td>1299</td>
      <td>1299</td>
      <td>100.00%</td>
    </tr>
    <tr>
      <td><a href="https://siliconflow.cn/">SiliconFlow</a></td>
      <td>https://siliconflow.cn</td>
      <td>1305</td>
      <td>1302</td>
      <td>99.77%</td>
    </tr>
    <tr>
      <td><a href="https://openrouter.ai/provider/chutes">Chutes</a></td>
      <td>https://openrouter.ai</td>
      <td>1271</td>
      <td>1229</td>
      <td>96.70%</td>
    </tr>
    <tr>
      <td><a href="https://github.com/vllm-project/vllm">vLLM</a></td>
      <td>https://github.com/vllm-project/vllm</td>
      <td>1325</td>
      <td>1007</td>
      <td>76.00%</td>
    </tr>
    <tr>
      <td><a href="https://github.com/sgl-project/sglang">SGLang</a></td>
      <td>https://github.com/sgl-project/sglang</td>
      <td>1269</td>
      <td>928</td>
      <td>73.13%</td>
    </tr>
    <tr>
      <td><a href="https://www.volcengine.com/">Volc</a></td>
      <td>https://www.volcengine.com</td>
      <td>1330</td>
      <td>969</td>
      <td>72.86%</td>
    </tr>
    <tr>
      <td><a href="https://openrouter.ai/provider/baseten">Baseten</a></td>
      <td>https://openrouter.ai</td>
      <td>1243</td>
      <td>901</td>
      <td>72.49%</td>
    </tr>
    <tr>
      <td><a href="https://openrouter.ai/provider/atlas-cloud">AtlasCloud</a></td>
      <td>https://openrouter.ai</td>
      <td>1277</td>
      <td>925</td>
      <td>72.44%</td>
    </tr>
    <tr>
      <td><a href="https://openrouter.ai/provider/together">Together</a></td>
      <td>https://openrouter.ai</td>
      <td>1266</td>
      <td>911</td>
      <td>71.96%</td>
    </tr>
    <tr>
      <td><a href="https://groq.com/">Groq</a></td>
      <td>https://groq.com</td>
      <td>69.52%</td>
      <td>1042</td>
      <td>1042</td>
      <td>100.00%</td>
    </tr>
    <tr>
      <td><a href="https://nebius.ai/">Nebius</a></td>
      <td>https://nebius.ai</td>
      <td>50.60%</td>
      <td>644</td>
      <td>544</td>
      <td>84.47%</td>
    </tr>
  </tbody>
</table>

##### We ran the official API multiple times to test the fluctuation of `tool_call_f1`. The lowest score was **82.71%**, and the average was **84%**. Given the inherent randomness of the model, we believe that an `tool_call_f1` score above **80%** is acceptable and can be used as a reference.

### Evaluation Metrics

#### ToolCall-Trigger Similarity
We use `tool_call_f1` to determine whether the model deployment is correct.

| Label / Metric | Formula | Meaning |
| --- | --- | --- |
| `TP` (True Positive) | — | Both model & official have `finish_reason == "tool_calls"`. |
| `FP` (False Positive) | — | Model `finish_reason == "tool_calls"` while official is `"stop"` or `"others"`. |
| `FN` (False Negative) | — | Model `finish_reason == "stop"` or `"others"` while official is `"tool_calls"`. |
| `TN` (True Negative) | — | Both model & official have `finish_reason == "stop"` or `"others"`. |
| `tool_call_precision` | `TP / (TP + FP)` | Proportion of triggered tool calls that should have been triggered. |
| `tool_call_recall` | `TP / (TP + FN)` | Proportion of tool calls that should have been triggered and were. |
| **`tool_call_f1`** | **`2*`tool_call_precision`*`tool_call_recall` / (`tool_call_precision`+`tool_call_recall`)`** | **Harmonic mean of precision and recall (primary metric for deployment check).** |



#### ToolCall-Schema Accuracy
We use `schema_accuracy` to measure the robustness of the engineering.

| Label / Metric | Formula / Condition | Description |
| --- | --- | --- |
| `count_finish_reason_tool_calls` | — | Number of responses with `finish_reason == "tool_calls"`. |
| `count_successful_tool_call` | — | Number of **tool_calls** responses that passed schema validation. |
| **`schema_accuracy`** | **`count_successful_tool_call / count_finish_reason_tool_calls`** | **Proportion of triggered tool calls whose JSON payload satisfies the schema.** |

## How we do the test

We test toolcall's response over a set of 4,000 requests. Each provider's responses are collected and compared against the official Moonshot AI API.

K2 vendors are periodically evaluated. If you are not on the list and would like to be included, feel free to contact us.

**Sample Data**: Detailed samples and MoonshotAI results are available in [tool-calls-dataset](https://statics.moonshot.cn/k2vv/tool-calls.tar.gz) (50% of the test set).

## Suggestions to Vendors

1. **Use the Correct Versions**  
Some vendors may not meet the requirements due to using incorrect versions. We recommend using the following versions and newer versions:
- K2-0905:
    - [vllm v0.11.0](https://github.com/vllm-project/vllm/releases/tag/v0.11.0)
    - [sglang v0.5.3rc0](https://github.com/sgl-project/sglang/releases/tag/v0.5.3rc0)
    - [moonshotai/Kimi-K2-Instruct-0905](https://huggingface.co/moonshotai/Kimi-K2-Instruct-0905) (commit: 94a4053eb8863059dd8afc00937f054e1365abbd)
- K2-thinking: 
    - [vllm v0.11.1rc6](https://github.com/vllm-project/vllm/releases/tag/v0.11.1rc6)
    - [sglang v0.5.5.post2](https://github.com/sgl-project/sglang/releases/tag/v0.5.5.post2)
    - [moonshotai/Kimi-K2-Thinking latest](https://huggingface.co/moonshotai/Kimi-K2-Thinking)

2. **Rename Tool Call IDs**  
The Kimi-K2 model expects all tool call IDs in historical messages to follow the format `functions.func_name:idx`. However, previous test cases may contain malformed tool IDs like `serach:0`*, which could mislead Kimi-K2 into generating incorrect tool call IDs, resulting in parsing failures.   
In this version, we manually add the `functions.` prefix to all previous tool calls to make Kimi-K2 happy :). We recommend that users and vendors adopt this fix in practice as well.   
This type of tool ID was generated by our official API. Before invoking the K2 model, our official API automatically renames all tool call IDs to the format `functions.func_name:idx`, so this is not an issue for us.

3. **Add Guided Encoding**  
Large language models generate text token-by-token according to probability; they have no built-in mechanism to enforce a hard JSON schema. Even with careful prompting, the model may omit fields, add extra ones, or nest them incorrectly. So please add guided encoding to ensure the correct schema.

## Verify by yourself

To run the evaluation tool with sample data, use the following command:

```bash
python tool_calls_eval.py samples.jsonl \
    --model kimi-k2-0905-preview \
    --base-url https://api.moonshot.cn/v1 \
    --api-key YOUR_API_KEY \
    --concurrency 5 \
    --output results.jsonl \
    --summary summary.json
```

- `samples.jsonl`: Path to the test set file in JSONL format
- `--model`: Model name (e.g., kimi-k2-0905-preview)
- `--base-url`: API endpoint URL
- `--api-key`: API key for authentication (or set OPENAI_API_KEY environment variable)
- `--concurrency`: Maximum number of concurrent requests (default: 5)
- `--output`: Path to save detailed results (default: results.jsonl)
- `--summary`: Path to save aggregated summary (default: summary.json)
- `--timeout`: Per-request timeout in seconds (default: 600)
- `--retries`: Number of retries on failure (default: 3)
- `--extra-body`: Extra JSON body as string to merge into each request payload (e.g., '{"temperature":0.6}')
- `--incremental`: Incremental mode to only rerun failed requests


For testing other providers via OpenRouter:

```bash
python tool_calls_eval.py samples.jsonl \
    --model moonshotai/kimi-k2-0905 \
    --base-url https://openrouter.ai/api/v1 \
    --api-key YOUR_OPENROUTER_API_KEY \
    --concurrency 5 \
    --extra-body '{"provider": {"only": ["YOUR_DESIGNATED_PROVIDER"]}}'
```

### Note for Kimi K2.5 Model Testing

For third-party APIs deployed with vLLM/SGLang/KTransformers, please note that:

- To use **Instant mode** (disable thinking), you need to pass `{"chat_template_kwargs": {"thinking": false}}` in `extra_body`:

```bash
python tool_calls_eval.py samples.jsonl \
    --model kimi-k2.5 \
    --base-url YOUR_API_BASE_URL \
    --api-key YOUR_API_KEY \
    --concurrency 5 \
    --extra-body '{"chat_template_kwargs": {"thinking": false}, "temperature": 0.6}'
```

## Contact Us
**We're preparing the next benchmark round and need your input.**

If there's any **metric or test case** you care about, please drop a note in [issue](https://github.com/MoonshotAI/K2-Vendor-Verifier/issues/9)

And welcome to drop the name of any vendor you’d like to see in in [issue](https://github.com/MoonshotAI/K2-Vendor-Verifier/issues/10)

---
If you have any questions or concerns, please reach out to us at contact-kvv@kimi.com.
