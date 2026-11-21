
import time
import openvino_genai as ov_genai
import json
from datetime import datetime
from pydantic import BaseModel
from pydantic import ValidationError
from typing import Literal
from pydantic import TypeAdapter

class ToolCall(BaseModel):
    tool: Literal["check_time"]

class Answer(BaseModel):
    answer: str

Reply = TypeAdapter(ToolCall | Answer)
REPLY_REGEX = r'\{"tool": "check_time"\}|\{"answer": "([^"\\\n]|\\.)*"\}'
MODEL = "models/qwen3-1.7b-int4"
SYSTEM = """You are a helpful, concise assistant. Sometimes you need to do the calculation or get any information from somewhere.
For that you can use tool call. Some of the tools are:
Check Time: [name: check_time, description: It is the tool which get current time] use check_time only when the user asks about the current time
For tool calling you have to answer it in JSON and just give answer in that Schema
SCHEMA: "{"tool": "check_time"} if you want to use get the exact time.
Only use tools when required based on the user input.
If tools calling is not required answer in JSON but use {"answer": "<Your reply>"}

"""


def stream(piece: str):
    # Called by GenAI each time a new piece of text is ready
    
    print(piece, end="", flush=True)
    return ov_genai.StreamingStatus.RUNNING   # keep going    
    


def main():
    print(f"Loading {MODEL}...")
    pipe = ov_genai.LLMPipeline(MODEL, "CPU", KV_CACHE_PRECISION="f16")

    config = ov_genai.GenerationConfig()
    config.max_new_tokens = 512
    config.do_sample = False
    config.temperature = 0.8
    config.structured_output_config = ov_genai.StructuredOutputConfig(regex=REPLY_REGEX)
    tools = {"check_time": check_time}

    # The conversation so far. The model sees all of it every turn.
    history = ov_genai.ChatHistory()
    system_prompt = SYSTEM
    history.append({"role": "system", "content": system_prompt})

    print("Ready. /reset to clear, /quit to exit.\n")
    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user:
            continue
        if user == "/quit":
            break
        if user == "/reset":
            history.clear()
            history.append({"role": "system", "content": system_prompt})
            print("(memory cleared)\n")
            continue
        if user == "/save":
            conversation = history.get_messages()
            with open("chat.json", "w") as f:
                json.dump(conversation, f, indent=2)
                print("saved to chat.json")
                continue
        if user.startswith("/system"):
            history.clear()
            history.append({"role": "system", "content": user[8:]})
            system_prompt = user[8:]
            print("Updated System Prompt")
            continue




        # /no_think turns off Qwen3's thinking mode, so it answers directly
        history.append({"role": "user", "content": user + " /no_think"})

        print("bot> ", end="", flush=True)
        start = time.perf_counter()
        
        result = pipe.generate(history, config)
        m = result.perf_metrics
        tokenPerSecond = m.get_throughput().mean
        ttft = m.get_ttft().mean
        
        seconds = time.perf_counter() - start


        reply = result.texts[0]
        # Qwen3 still writes an empty "<think>\n\n</think>" first. Keep only the text after it.

        if "</think>" in reply:
            reply = reply.split("</think>", 1)[1].strip()
        history.append({"role": "assistant", "content": reply})
        try:
            parsed = Reply.validate_json(reply)
            if isinstance(parsed, ToolCall):
                tool_result = tools[parsed.tool]()
                print(f"Tool check_time -> {tool_result} ")
                history.append({"role": "user", "content": f"Tool result: {tool_result}"})
                print("bot> ", end="", flush=True)
                result = pipe.generate(history, config)
                reply = result.texts[0]
                if "</think>" in reply:
                    reply = reply.split("</think>", 1)[1].strip()
                history.append({"role": "assistant", "content": reply})
                parsed = Reply.validate_json(reply)

            if isinstance(parsed, Answer):
                print(parsed.answer)
            else:
                print("(the model asked for a tool again, stopping here)")

        except ValidationError:
            print("(could not read the model's reply)")
            print(repr(reply))
        print(f"\n({seconds:.1f}s), Token Per Second: {tokenPerSecond:.0f}, TTFT: {ttft:.0f}\n")
        

    print("bye")

def check_time():
    now = datetime.now()
    formatted = now.strftime("%H:%M:%S")
    return formatted


if __name__ == "__main__":
    main()
