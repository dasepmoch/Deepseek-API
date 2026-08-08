# DeepSeek API Bridge

**Turn your free DeepSeek account into an OpenAI-compatible API.** No API key, no billing, no credit card — just your existing chat.deepseek.com login, exposed as a standard `/v1` endpoint you can use from code, agents, or any OpenAI-compatible tool.

![Version](https://img.shields.io/badge/version-0.1.0-blue)
![License](https://img.shields.io/badge/license-MIT-green)

```
┌──────────────┐   OpenAI format   ┌─────────────────┐   DeepSeek internal   ┌────────────────┐
│ OpenAI SDK   │ ◄───────────────► │  bridge (:8090) │ ────────────────────► │ chat.deepseek  │
│ OpenCode     │                   │  /v1/*          │ ◄──────────────────── │ .com (web)     │
│ Hermes       │                   └─────────────────┘                       └────────────────┘
└──────────────┘
```

Built around one idea: **make a free DeepSeek account behave like a real API** — including tool calling, streaming, DeepThink reasoning, and usage metadata.

---

## ✨ Features

- **OpenAI-compatible** — speaks the standard `/v1/chat/completions` protocol; drop-in for the OpenAI SDK.
- **Tool calling (agent-ready)** — `tools` in, native `tool_calls` out. Works with OpenCode, Hermes, and any OpenAI-compatible agent.
- **Streaming** — proper SSE deltas, including tool-call chunks and final `usage`.
- **DeepThink (reasoning)** — `thinking: true` returns `reasoning_content` + `reasoning_tokens`, exactly like the official API.
- **Web search** — `search: true` toggle per request.
- **Multi-turn** — resume threads with `conversation_id`.
- **Self-hosted & private** — session, cookies, and history stay on your machine.
- **Auto-refresh session** — token re-captured headlessly from your saved browser profile.

---

## Requirements

- Python 3.9+
- A DeepSeek account (free) — [chat.deepseek.com](https://chat.deepseek.com)
- Linux / macOS / Windows

---

## Setup (2 minutes)

```bash
git clone https://github.com/dasepmoch/Deepseek-API.git
cd Deepseek-API

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Install the browser Playwright needs (one-time)
playwright install chromium

# Sign in once: a browser opens, log in with your DeepSeek account
python -m deepseek.auth
```

Your session (token + cookies) is saved under `session/` and reused automatically. Token refresh is also automatic — you won't need to sign in repeatedly.

---

## Run the server

```bash
python app.py
# → OpenAI-compatible API at http://127.0.0.1:8090
```

Change the address with env vars: `HOST=0.0.0.0 PORT=8080 python app.py`

### Endpoints

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/v1/chat/completions` | Chat — supports `stream`, `tools`, `thinking`, `search`, `conversation_id` |
| `GET`  | `/v1/models` | List models |
| `GET`  | `/healthz` | Health check |

### Models

| Model | Mode | Notes |
| --- | --- | --- |
| `deepseek-v4-flash` | Instant | Fast model, default |
| `deepseek-v4-pro` | Expert | Stronger, slower |

---

## Usage examples

### Chat

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8090/v1", api_key="unused")

resp = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(resp.choices[0].message.content)
```

### Tool calling (agents)

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8090/v1", api_key="unused")

resp = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[{"role": "user", "content": "List the files in /tmp"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    }],
)
print(resp.choices[0].message.tool_calls)  # native tool_calls
```

Tool calling is emulated server-side: tool definitions are rendered into the prompt, the model's output is parsed, then re-wrapped as native OpenAI `tool_calls`. Because the protocol is standard, this works with **OpenCode**, **Hermes**, and other OpenAI-compatible agents.

### DeepThink (reasoning)

```python
resp = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[{"role": "user", "content": "Solve step by step: 17*23"}],
    extra_body={"thinking": True},
)
print(resp.choices[0].message.content)            # final answer
print(resp.choices[0].message.reasoning_content)  # chain-of-thought
print(resp.usage.completion_tokens_details.reasoning_tokens)  # reasoning tokens
```

### Streaming

```python
stream = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[{"role": "user", "content": "Write a short poem"}],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="")
```

---

## Auto-start (systemd)

Run the server as a service that restarts automatically on boot:

```ini
# ~/.config/systemd/user/deepseek-bridge.service
[Unit]
Description=DeepSeek API Bridge
After=network-online.target

[Service]
Type=simple
WorkingDirectory=/path/to/Deepseek-API
Environment=PORT=8090
Environment=DEEPSEEK_PROFILE_DIR=/path/to/Deepseek-API/session/profile
ExecStart=/path/to/Deepseek-API/venv/bin/python app.py
Restart=always

[Install]
WantedBy=default.target
```

```bash
systemctl --user enable --now deepseek-bridge.service
```

For periodic session refresh (keeps the token fresh), a systemd timer can run `scripts/refresh_session.py`.

---

## Project layout

| Path | Purpose |
| --- | --- |
| `deepseek/` | Core library: browser auth/login, HTTP client, PoW solver |
| `server/` | FastAPI OpenAI-compatible server |
| `scripts/` | Utilities (session refresh) |
| `examples/` | Runnable examples |
| `app.py` | Server entry point |

---

## Honest limitations

- **Not the official DeepSeek API.** This project bridges DeepSeek's web chat for personal, reasonable use. Please respect DeepSeek's terms of service.
- **Tool calling is emulated** — the web model is a chat model, so tool-call quality is best-effort. The parser tolerates several output formats to keep agent loops reliable.
- **Token usage is an estimate** — re-counted with a BPE tokenizer (`tiktoken`), not official backend numbers. Accurate to within ~1 token.
- **Rate limit** 30 req/min per IP by default (change with `RATE_LIMIT_PER_MINUTE`).
- **Serialized requests** — the PoW solver is not reentrant, so requests are processed one at a time.
- Vision is not supported yet.

---

## License

[MIT](LICENSE) © 2026 dasepmoch