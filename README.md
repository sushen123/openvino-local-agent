# OpenVINO Local Agent

A chat agent that runs fully on your own computer, using [OpenVINO GenAI](https://github.com/openvinotoolkit/openvino.genai) and a small open model ([Qwen3-1.7B, int4](https://huggingface.co/OpenVINO/Qwen3-1.7B-int4-ov)). No cloud, no API keys.

This is the first version of a bigger project: a local agent that grows from a chat CLI into RAG over your own files, tool use with a sandbox and approvals, and benchmarks of different model sizes and precisions.

## Features

- Chat in the terminal with memory of the conversation
- Answers stream live, word by word
- Shows speed after every answer: tokens per second and time to first token
- Tool calling: the model can ask for the current time, the code runs the tool and the model answers with the result
- Commands:
  - `/system <text>` changes the system prompt
  - `/reset` clears the conversation
  - `/save` saves the conversation to `chat.json`
  - `/quit` exits

## Example

```
you> what time is it?
bot> {"tool": "check_time"}
Tool result: 10:05:37
bot> The current time is 10:05:37.

you> what is 2+2?
bot> 2 + 2 = 4.
(0.8s), Token Per Second: 20, TTFT: 236
```

## Setup

You need [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
git clone https://github.com/sushen123/openvino-local-agent.git
cd openvino-local-agent
uv sync
```

Download the model (about 1.1 GB). It is not stored in this repo:

```bash
uv run hf download OpenVINO/Qwen3-1.7B-int4-ov --local-dir models/qwen3-1.7b-int4
```

## Run

```bash
uv run python chat_cli.py
```

Tested on a MacBook Air (Apple M5, 16 GB RAM, CPU) with OpenVINO GenAI 2026.4.1, at about 20 tokens per second.

## Notes

- On Apple silicon, the model gives garbled text with the default KV cache precision, so the code sets `KV_CACHE_PRECISION="f16"` ([openvinotoolkit/openvino#39228](https://github.com/openvinotoolkit/openvino/issues/39228), fixed upstream after 2026.4.1).
- Qwen3 thinking mode is turned off with `/no_think` to keep answers short.
- On the stateful (SDPA) pipeline, a chat history that ends with a `tool` message is rejected, so the tool result is sent back as a `user` message.

## Roadmap

- [x] v1: chat CLI with streaming, commands and one tool
- [ ] v2: RAG over your own files (embeddings with OpenVINO)
- [ ] v3: agent loop with several tools
- [ ] v4: LangGraph agent with a Docker sandbox and approval before risky actions
- [ ] v5: benchmark mode comparing fp16, int8 and int4
