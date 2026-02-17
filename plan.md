# Telegram Bot Integration Plan

## Goal
Wrap the existing Harel Insurance chatbot in a Telegram bot interface, reusing the existing agent pipeline and conversation management.

## Architecture

```
Telegram User
  ↓ (message)
Telegram Bot API
  ↓
python-telegram-bot library (async handler)
  ↓
Reuse: AppResources (agent, vector store, embeddings, LLM)
  ↓
agent.invoke(agent_input)  ← same pipeline as FastAPI /chat
  ↓
Format response for Telegram (markdown + citations)
  ↓
Bot sends reply back to user
```

The Telegram bot runs as a **separate entry point** (`scripts/telegram_bot.py`) using the same shared `AppResources` initialization, so no code duplication of the agent logic.

---

## Implementation Steps

### Step 1: Add `python-telegram-bot` dependency
- Add `python-telegram-bot[job-queue]>=21.0` to `requirements.txt`

### Step 2: Add Telegram config to `config/settings.py`
- Add `telegram_bot_token: str = ""` field to the Settings model
- Load from env var `TELEGRAM_BOT_TOKEN`

### Step 3: Create `telegram_bot/` module (2 files)

**`telegram_bot/__init__.py`** — empty

**`telegram_bot/bot.py`** — core bot logic:
- `start` command handler → sends welcome message (Hebrew + English)
- `help` command handler → usage instructions
- `message_handler` → processes incoming text:
  1. Map `chat_id` → `conversation_id` (use Telegram's chat ID directly)
  2. Maintain per-chat conversation history (same in-memory dict pattern as the API)
  3. Send "typing" action while processing
  4. Call `resources.agent.invoke(agent_input)` (identical to `/chat` endpoint)
  5. Format response: answer text + inline citations + domain badge
  6. Handle Telegram's 4096-char message limit (split if needed)
- `format_response()` → convert ChatResponse-style output to Telegram MarkdownV2
  - Citations rendered as a compact list with links
  - Domain shown as a label at the top
- `run_bot()` → initialize AppResources, build Application, register handlers, start polling

### Step 4: Create entry point `scripts/telegram_bot.py`
- Thin script that calls `telegram_bot.bot.run_bot()`
- Loads env vars (dotenv)

### Step 5: Add Makefile target
- `make telegram` → runs the Telegram bot

### Step 6: Update `.env.example` with `TELEGRAM_BOT_TOKEN` placeholder

---

## Key Design Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Library | `python-telegram-bot` v21+ | Mature, async-native, well-documented |
| Running mode | Long-polling (`Application.run_polling()`) | Simple, no webhook/SSL setup needed |
| Entry point | Separate script, not embedded in FastAPI | Decoupled lifecycle, can run independently |
| Conversation ID | Use Telegram `chat_id` as string | Natural 1:1 mapping, persistent across sessions |
| Message formatting | Telegram HTML mode | More reliable than MarkdownV2 for Hebrew text |
| Citation display | Compact inline list below answer | Telegram doesn't support collapsible sections |

---

## Message Format Example (Telegram)

```
🏥 ביטוח בריאות

התשובה כאן בעברית...

📎 מקורות:
1. שם המסמך - סעיף X (קישור)
2. שם המסמך - סעיף Y (קישור)
```

---

## Files to Create/Modify

| File | Action |
|------|--------|
| `requirements.txt` | Modify — add python-telegram-bot |
| `config/settings.py` | Modify — add telegram_bot_token |
| `telegram_bot/__init__.py` | Create |
| `telegram_bot/bot.py` | Create |
| `scripts/telegram_bot.py` | Create |
| `Makefile` | Modify — add `telegram` target |
