"""Translate between OpenAI's chat-completions shapes and our DeepSeek client.

DeepSeek's protocol has no system/role channel and no native tool calling —
just a single `prompt` string. This module makes the bridge *look* like a real
OpenAI API from the outside:

* `tools` in the request are rendered into the prompt as a strict protocol.
* The model's text output is parsed for emulated tool calls and re-emitted as
  native OpenAI `tool_calls` (with `id`, `type`, `function.name`, JSON-encoded
  `arguments`), including in streaming mode where deltas are split per index
  exactly like the real API.
* Tool results (`role: tool`) and prior assistant `tool_calls` are flattened
  back into the prompt so multi-turn agent loops work.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Iterable, List, Optional

from .schemas import ChatMessage

_ROLE_LABELS = {"system": "System", "user": "User", "assistant": "Assistant", "tool": "Tool Result"}

# Marker the model is told to emit when it wants to call a tool.
_TOOL_CALL_OPEN = "<tool_call>"
_TOOL_CALL_CLOSE = "</tool_call>"

_CHUNK_SIZE = 24  # chars per streaming arguments delta

# System-prompt fragment defining the tool-calling protocol. Kept strict and
# example-driven because DeepSeek's web model is a chat model, not a
# function-calling model.
_TOOL_PROTOCOL = """\
You are an AI assistant that can call tools to complete tasks.

## Available tools
%TOOLS%

## How to call a tool
When you need to use a tool, reply with EXACTLY this format and nothing else
(no markdown fences, no explanations, no surrounding text):

<tool_call>
{"name": "tool_name", "arguments": {"param1": "value1", "param2": "value2"}}
</tool_call>

Rules:
- `arguments` must be a JSON object matching the tool's parameters.
- To call multiple tools at once, emit multiple <tool_call> blocks back-to-back.
- You will receive the tool result in the next message. Then continue the task.
- Do NOT call a tool when you already have everything you need. When the task
  is complete, reply normally with a final text answer.
"""


def _tool_schema_to_text(tool: dict) -> str:
    """Render one OpenAI tool definition as compact text for the model."""
    fn = tool.get("function", {})
    name = fn.get("name", "unknown")
    desc = fn.get("description", "")
    params = fn.get("parameters", {})
    props = params.get("properties", {})
    required = params.get("required", [])
    lines = [f"- {name}: {desc}".strip()]
    if props:
        lines.append("  Parameters:")
        for pname, pmeta in props.items():
            req = " (required)" if pname in required else ""
            pdesc = pmeta.get("description", "")
            ptype = pmeta.get("type", "any")
            lines.append(f"    - {pname} ({ptype}){req}: {pdesc}".strip())
    return "\n".join(lines)


def _choice_instruction(tool_choice) -> str:
    """Return an extra instruction line for a given tool_choice value."""
    if tool_choice == "none":
        return (
            "\nIMPORTANT: Do NOT call any tools this turn. Answer directly.\n"
        )
    if tool_choice == "required":
        return (
            "\nIMPORTANT: You MUST call one of the available tools this turn. "
            "Do not answer directly.\n"
        )
    if isinstance(tool_choice, dict):
        fn = (tool_choice.get("function") or {}).get("name")
        if fn:
            return (
                f"\nIMPORTANT: You MUST call the tool '{fn}' this turn and "
                f"only that tool.\n"
            )
    return ""


def build_tool_prompt(tools: List[dict], messages: List[ChatMessage],
                      tool_choice=None) -> str:
    """Build a single DeepSeek prompt from messages + tool definitions.

    Injects the tool protocol into the system content, flattens the message
    history (including prior tool results) into the conversation, and appends
    an 'Assistant:' cue so the model continues appropriately.
    """
    tool_text = "\n".join(_tool_schema_to_text(t) for t in tools)
    protocol = _TOOL_PROTOCOL.replace("%TOOLS%", tool_text)
    protocol += _choice_instruction(tool_choice)

    lines: List[str] = []
    system_seen = False
    for m in messages:
        if m.role == "system":
            system_seen = True
            content = _text_of(m.content) or ""
            lines.append(f"System: {content}\n\n{protocol}")
        elif m.role == "tool":
            content = _text_of(m.content) or ""
            call_id = getattr(m, "tool_call_id", None) or ""
            label = f"Tool Result ({call_id}):" if call_id else "Tool Result:"
            lines.append(f"{label}\n{content}")
        else:
            label = _ROLE_LABELS.get(m.role, m.role.capitalize())
            content = _text_of(m.content)
            # Reconstruct prior assistant tool calls so the model sees what it did.
            if m.role == "assistant" and m.tool_calls and not content:
                blocks = []
                for tc in m.tool_calls:
                    fn = tc.get("function", {})
                    name = fn.get("name", "unknown")
                    try:
                        args = json.loads(fn.get("arguments", "{}"))
                    except json.JSONDecodeError:
                        args = {}
                    blocks.append(
                        f"{_TOOL_CALL_OPEN}\n"
                        f'{{"name": "{name}", "arguments": '
                        f'{json.dumps(args, ensure_ascii=False)}}}\n'
                        f"{_TOOL_CALL_CLOSE}"
                    )
                content = "\n".join(blocks)
            lines.append(f"{label}: {content}")
    if not system_seen:
        lines.insert(0, f"System: {protocol}")
    lines.append("Assistant:")
    return "\n\n".join(lines)


# --- tool-call parsing -------------------------------------------------------


def _strip_fences(raw: str) -> str:
    """Strip markdown code fences (```json ... ``` or ``` ... ```) if present."""
    s = raw.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    return s


def _salvage_json(raw: str) -> Optional[dict]:
    """Best-effort: find the first balanced {...} in a string."""
    depth = 0
    start = -1
    for i, ch in enumerate(raw):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start != -1:
                try:
                    return json.loads(raw[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _parse_block(raw: str) -> Optional[dict]:
    """Parse one tool-call block into {"name": ..., "arguments": {...}}."""
    raw = _strip_fences(raw)
    obj = None
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        obj = _salvage_json(raw)
    if not isinstance(obj, dict) or "name" not in obj:
        return None
    args = obj.get("arguments", {})
    if isinstance(args, str):
        # Some models emit arguments as an escaped JSON string.
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            pass
    if not isinstance(args, dict):
        args = {}
    return {"name": str(obj["name"]), "arguments": args}


def parse_tool_calls(text: str) -> List[dict]:
    """Extract emulated tool calls from the model's text output.

    Accepts tool calls in several formats (the web model is not always
    consistent about which one it uses):

    1. Protocol format (one or more):
         <tool_call>
         {"name": "...", "arguments": {...}}
         </tool_call>
    2. Markdown "Calling:" style (taught by some clients' system prompts):
         **Calling:** `bash`
         ```json
         {"command": "..."}
         ```
    3. Bare JSON objects with a "name" key anywhere in the text.

    Returns OpenAI-shaped tool call dicts:
        [{"id": "call_...", "type": "function",
          "function": {"name": ..., "arguments": "<json string>"}}]
    """
    calls: List[dict] = []
    idx = 0
    while True:
        start = text.find(_TOOL_CALL_OPEN, idx)
        if start == -1:
            break
        inner = text.find(">", start) + 1  # skip past the opening tag
        end = text.find(_TOOL_CALL_CLOSE, inner)
        if end == -1:
            break
        raw = text[inner:end].strip()
        idx = end + len(_TOOL_CALL_CLOSE)
        if not raw:
            continue
        parsed = _parse_block(raw)
        if parsed is None:
            continue
        calls.append(_to_openai_tool_call(parsed["name"], parsed["arguments"]))
    if calls:
        return calls

    # Fallback 1: markdown "Calling: `tool`" style followed by a JSON block.
    calls = _parse_calling_style(text)
    if calls:
        return calls

    # Fallback 2: any bare JSON object with a "name" key anywhere in the text.
    for candidate in _iter_json_objects(text):
        if isinstance(candidate, dict) and "name" in candidate:
            args = candidate.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            if not isinstance(args, dict):
                args = {}
            calls.append(_to_openai_tool_call(str(candidate["name"]), args))
    return calls


def _parse_calling_style(text: str) -> List[dict]:
    """Parse '**Calling:** `tool`' style blocks used by some clients.

    Looks for a line like `**Calling:** `bash`` or `Calling: bash` followed by
    a JSON object (fenced or bare) on subsequent lines.
    """
    import re

    calls: List[dict] = []
    # Match: optional markdown emphasis, "Calling:", optional closing
    # emphasis (e.g. "**Calling:** `bash`"), optional backticks, name.
    pattern = re.compile(
        r"(?im)^\s*(?:\*\*)?Calling:\s*(?:\*\*)?\s*`?([a-zA-Z_][a-zA-Z0-9_]*)`?\s*$"
    )
    for m in pattern.finditer(text):
        name = m.group(1)
        rest = text[m.end():]
        # Find the next JSON object after the heading — in this style it IS
        # the arguments object (e.g. {"command": "..."}).
        obj = _salvage_json(rest)
        if obj is None:
            continue
        args = obj
        if not isinstance(args, dict):
            args = {}
        calls.append(_to_openai_tool_call(name, args))
    return calls


def _iter_json_objects(text: str):
    """Yield JSON values found in a string (naive but effective)."""
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start != -1:
                try:
                    yield json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    pass
                start = -1


def _to_openai_tool_call(name: str, args: dict) -> dict:
    return {
        "id": "call_" + uuid.uuid4().hex[:24],
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(args, ensure_ascii=False),
        },
    }


def _text_of(content) -> str:
    """Extract plain text from a message's content (string or list-of-parts)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for p in content:
        if isinstance(p, dict) and p.get("type") == "text":
            parts.append(p.get("text", ""))
    return "\n".join(parts)


def messages_to_prompt(messages: List[ChatMessage]) -> str:
    """Flatten a chat history into a single prompt DeepSeek can answer.

    A lone user message is sent verbatim. Multi-turn / system-prompted
    conversations are serialised with role labels and a trailing 'Assistant:'
    cue so the model continues in the right voice.
    """
    if len(messages) == 1 and messages[0].role == "user":
        return _text_of(messages[0].content)

    lines = []
    for m in messages:
        label = _ROLE_LABELS.get(m.role, m.role.capitalize())
        lines.append(f"{label}: {_text_of(m.content)}")
    lines.append("Assistant:")
    return "\n\n".join(lines)


# --- response shaping --------------------------------------------------------


def _now() -> int:
    return int(time.time())


def _id() -> str:
    return "chatcmpl-" + uuid.uuid4().hex


def _est_tokens(text: str) -> int:
    """Token count for a string.

    Uses a real BPE tokenizer (tiktoken, cl100k_base — the counting scheme the
    OpenAI SDK and DeepSeek's own API use) for close-to-exact numbers, falling
    back to a ~4-char/token estimate when tiktoken isn't installed. DeepSeek's
    web API doesn't expose true counts, so this is an approximation — but a
    much better one than a naive char/4 guess.
    """
    if not text:
        return 0
    enc = _get_tokenizer()
    if enc is not None:
        try:
            return len(enc.encode(text))
        except Exception:
            pass
    return max(1, len(text) // 4)


_tokenizer_cache = None


def _get_tokenizer():
    """Lazily load a tiktoken BPE encoder (cl100k_base)."""
    global _tokenizer_cache
    if _tokenizer_cache is None:
        try:
            import tiktoken
            _tokenizer_cache = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _tokenizer_cache = False
    return _tokenizer_cache if _tokenizer_cache is not None else None


def completion_response(model: str, content: Optional[str], prompt: str,
                        conversation_id: Optional[str] = None,
                        tool_calls: Optional[List[dict]] = None,
                        finish_reason: Optional[str] = None,
                        reasoning: str = "") -> dict:
    """A full (non-streaming) OpenAI chat.completion object.

    `conversation_id` is an extra top-level field (outside OpenAI's schema) you
    send back to resume the conversation. When the model requested tools,
    `tool_calls` is populated, `content` is None and finish_reason is
    `tool_calls` — exactly like the real API. `reasoning` (DeepThink chain of
    thought) is exposed as `reasoning_content`, like the official API.
    """
    message: dict = {"role": "assistant", "content": content}
    if reasoning:
        message["reasoning_content"] = reasoning
    pt = _est_tokens(prompt)
    if tool_calls:
        message["content"] = None
        message["tool_calls"] = tool_calls
        # Tool-call arguments count as completion tokens too.
        args_text = "".join(
            tc.get("function", {}).get("arguments", "") for tc in tool_calls
        )
        ct = _est_tokens(args_text)
    else:
        ct = _est_tokens(content or "")
    return {
        "id": _id(),
        "object": "chat.completion",
        "created": _now(),
        "model": model,
        "conversation_id": conversation_id,
        "system_fingerprint": _fingerprint(),
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason or ("tool_calls" if tool_calls else "stop"),
            }
        ],
        "usage": _usage(prompt, content or "", reasoning),
    }


def _fingerprint() -> str:
    """A stable-ish system fingerprint like the real API's `fp_...`."""
    return "fp_" + uuid.uuid4().hex[:10]


def _usage(prompt: str, completion: str, reasoning: str = "") -> dict:
    """OpenAI-shaped usage object with BPE-based token estimates."""
    pt, ct = _est_tokens(prompt), _est_tokens(completion)
    usage: dict = {
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "total_tokens": pt + ct,
    }
    if reasoning:
        rt = _est_tokens(reasoning)
        usage["completion_tokens"] = ct + rt
        usage["total_tokens"] = pt + ct + rt
        usage["completion_tokens_details"] = {"reasoning_tokens": rt}
    return usage


def stream_chunks(model: str, stream: Iterable[str],
                  tools: bool = False,
                  prompt: str = "") -> Iterable[str]:
    """Yield OpenAI SSE lines (`data: {...}\n\n`) for a streamed completion.

    When `tools` is True, the stream is buffered, parsed for emulated tool
    calls, and re-emitted with *native* streaming tool-call deltas:
      - a role chunk,
      - a per-index chunk announcing id + name,
      - an arguments delta,
      - a final chunk with finish_reason "tool_calls".
    Plain text (no tool call) streams as usual with finish_reason "stop".
    The final chunk always carries `usage` and `system_fingerprint`, like the
    real API. DeepThink chains of thought are streamed as `reasoning_content`
    deltas and counted in `usage.completion_tokens_details.reasoning_tokens`.
    """
    cid, created = _id(), _now()

    def frame(delta: dict, finish=None, extra: dict = None) -> str:
        obj = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        if extra:
            obj.update(extra)
        return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

    # First frame announces the assistant role.
    yield frame({"role": "assistant", "content": ""})

    if tools:
        buf: List[str] = []
        streamed_len = 0
        marker_hit = False
        keep = len(_TOOL_CALL_OPEN) - 1  # lookahead: chars held back to detect the marker
        for d in stream:
            if not d:
                continue
            buf.append(d)
            text_so_far = "".join(buf)
            # 1) Marker visible past the already-streamed region?
            pos = text_so_far.find(_TOOL_CALL_OPEN, streamed_len)
            if pos != -1:
                # Stream text before the marker, then stop streaming content.
                if streamed_len < pos:
                    yield frame({"content": text_so_far[streamed_len:pos]})
                streamed_len = pos
                marker_hit = True
                continue
            # 2) No marker yet: safely flush everything except the last
            #    len(marker)-1 chars (they might be a marker prefix).
            flush_end = len(text_so_far) - keep
            if flush_end > streamed_len:
                yield frame({"content": text_so_far[streamed_len:flush_end]})
                streamed_len = flush_end

        text = "".join(buf)
        reasoning = getattr(stream, "reasoning", "") or ""  # set after iteration
        # With thinking enabled the model may write <tool_call> inside the
        # THINK fragment; look in both text and reasoning.
        tool_calls = parse_tool_calls(text) or parse_tool_calls(reasoning)
        conversation_id = getattr(stream, "conversation_id", None)
        if reasoning:
            yield frame({"reasoning_content": reasoning})
        if tool_calls:
            # Emit each tool call as a compact series of native deltas: one
            # announcement chunk (id + name) followed by arguments deltas.
            for i, tc in enumerate(tool_calls):
                fn = tc.get("function", {})
                # Announce the call: id + name, empty arguments.
                yield frame({
                    "tool_calls": [{
                        "index": i,
                        "id": tc.get("id"),
                        "type": "function",
                        "function": {"name": fn.get("name"), "arguments": ""},
                    }]
                })
                # Arguments — single delta keeps maximum client compatibility
                # (OpenCode, Hermes, openai SDK all accept this).
                args = fn.get("arguments", "")
                yield frame({
                    "tool_calls": [{
                        "index": i,
                        "function": {"arguments": args},
                    }]
                })
            yield frame(
                {},
                finish="tool_calls",
                extra={
                    "conversation_id": conversation_id,
                    "system_fingerprint": _fingerprint(),
                    "usage": _usage(prompt, text, reasoning),
                },
            )
        else:
            # No tool call: flush whatever was held back (marker in prose or
            # plain text answer that never matched).
            if streamed_len < len(text):
                yield frame({"content": text[streamed_len:]})
            yield frame(
                {},
                finish="stop",
                extra={
                    "conversation_id": conversation_id,
                    "system_fingerprint": _fingerprint(),
                    "usage": _usage(prompt, text, reasoning),
                },
            )
        yield "data: [DONE]\n\n"
        return

    buf: List[str] = []
    for d in stream:
        if d:
            buf.append(d)
            yield frame({"content": d})
    text = "".join(buf)
    conversation_id = getattr(stream, "conversation_id", None)
    reasoning = getattr(stream, "reasoning", "") or ""  # set after iteration
    if reasoning:
        yield frame({"reasoning_content": reasoning})
    yield frame(
        {},
        finish="stop",
        extra={
            "conversation_id": conversation_id,
            "system_fingerprint": _fingerprint(),
            "usage": _usage(prompt, text, reasoning),
        },
    )
    yield "data: [DONE]\n\n"