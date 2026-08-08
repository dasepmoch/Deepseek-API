# DeepSeek API Bridge

**Ubah akun DeepSeek gratismu jadi OpenAI-compatible API.** Tanpa API key, tanpa biaya, tanpa kartu kredit — cukup pakai sesi login chat.deepseek.com yang sudah kamu punya, dan dapatkan endpoint API yang bisa dipakai dari kode, agent, atau tools apa pun.

```
┌──────────────┐   OpenAI format   ┌─────────────────┐   DeepSeek internal   ┌────────────────┐
│ OpenAI SDK   │ ◄───────────────► │  bridge (:8090) │ ────────────────────► │ chat.deepseek  │
│ OpenCode     │                   │  /v1/*          │ ◄──────────────────── │ .com (web)     │
│ Hermes       │                   └─────────────────┘                       └────────────────┘
└──────────────┘
```

Built with a single goal: **make a free DeepSeek account behave like a real API** — including tool calling, streaming, DeepThink reasoning, and usage metadata.

---

## ✨ Fitur

- **OpenAI-compatible** — speak the standard `/v1/chat/completions` protocol, drop-in for the OpenAI SDK.
- **Tool calling (agent-ready)** — `tools` in, native `tool_calls` out. Works with OpenCode, Hermes, and any OpenAI-compatible agent.
- **Streaming** — proper SSE deltas incl. tool-call chunks and final `usage`.
- **DeepThink (reasoning)** — `thinking: true` returns `reasoning_content` + `reasoning_tokens`, exactly like the official API.
- **Web search** — `search: true` toggle per request.
- **Multi-turn** — resume threads with `conversation_id`.
- **Self-hosted & private** — session, cookies, and history stay on your machine.
- **Auto-refresh session** — token re-captured headlessly from your saved browser profile.

---

## Requirements

- Python 3.9+
- Akun DeepSeek (gratis) — [chat.deepseek.com](https://chat.deepseek.com)
- Linux / macOS / Windows

---

## Setup (2 menit)

```bash
git clone https://github.com/dasepmoch/Deepseek-API.git
cd Deepseek-API

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Install browser untuk login (sekali saja)
playwright install chromium

# Login sekali: browser terbuka, masuk dengan akun DeepSeek-mu
python -m deepseek.auth
```

Sesi login (token + cookies) tersimpan di `session/` dan dipakai ulang otomatis. Refresh token juga otomatis — kamu tidak perlu login berulang.

---

## Jalankan server

```bash
python app.py
# → OpenAI-compatible API di http://127.0.0.1:8090
```

Ubah port/host dengan env: `HOST=0.0.0.0 PORT=8080 python app.py`

### Endpoints

| Method | Path | Deskripsi |
| --- | --- | --- |
| `POST` | `/v1/chat/completions` | Chat — dukung `stream`, `tools`, `thinking`, `search`, `conversation_id` |
| `GET`  | `/v1/models` | Daftar model |
| `GET`  | `/healthz` | Health check |

### Model

| Model | Mode | Catatan |
| --- | --- | --- |
| `deepseek-chat` | Instant | Model cepat, default |
| `deepseek-expert` | Expert | Lebih kuat, lebih lambat |

---

## Contoh pakai

### Chat biasa

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8090/v1", api_key="unused")

resp = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "Halo!"}],
)
print(resp.choices[0].message.content)
```

### Tool calling (agent)

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8090/v1", api_key="unused")

resp = client.chat.completions.create(
    model="deepseek-chat",
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

Tool calling di-emulasi di sisi server: deskripsi tool di-render ke prompt, output model di-parse, lalu dibungkus ulang jadi `tool_calls` native OpenAI. Karena protokolnya standar, ini bekerja dengan **OpenCode**, **Hermes**, dan agent OpenAI-compatible lain.

### DeepThink (reasoning)

```python
resp = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "Solve step by step: 17*23"}],
    extra_body={"thinking": True},
)
print(resp.choices[0].message.content)           # jawaban akhir
print(resp.choices[0].message.reasoning_content)  # chain-of-thought
print(resp.usage.completion_tokens_details.reasoning_tokens)  # token reasoning
```

### Streaming

```python
stream = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "Tulis puisi pendek"}],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="")
```

---

## Auto-start (systemd)

Server bisa jalan terus sebagai service dan restart otomatis saat boot:

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

Refresh session berkala (biar token selalu fresh) bisa lewat timer systemd yang menjalankan `scripts/refresh_session.py`.

---

## Struktur proyek

| Path | Fungsi |
| --- | --- |
| `deepseek/` | Library inti: auth/login browser, HTTP client, PoW solver |
| `server/` | FastAPI OpenAI-compatible server |
| `scripts/` | Util (refresh session) |
| `examples/` | Contoh pemakaian |
| `app.py` | Entry point server |

---

## Keterbatasan (jujur)

- **Ini bukan API resmi DeepSeek.** Proyek ini menjembatani chat web DeepSeek untuk penggunaan pribadi/wajar. Pakai sesuai ketentuan DeepSeek.
- **Tool calling di-emulasi** — model web adalah model chat, jadi kualitas tool call best-effort. Parser toleran ke beberapa format output supaya agent loop tetap jalan.
- **Token usage adalah estimasi** — dihitung ulang dengan tokenizer BPE (`tiktoken`), bukan angka resmi dari backend. Akurat sampai ±1 token.
- **Rate limit** 30 req/menit per IP bawaan (ubah dengan `RATE_LIMIT_PER_MINUTE`).
- **Request serial** — solver PoW tidak reentrant, jadi request diproses satu-satu.
- Vision belum didukung.

---

## Lisensi

[MIT](LICENSE) © 2026 Dasep M Luay