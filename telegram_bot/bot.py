"""Telegram bot interface for the Harel Insurance chatbot."""

import asyncio

from langchain_core.messages import AIMessage, HumanMessage
from loguru import logger
from telegram import Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from api.dependencies import AppResources
from config.domains import DOMAINS
from config.settings import settings

# Domain emoji mapping
DOMAIN_ICONS: dict[str, str] = {
    "car": "\U0001f697",
    "life": "\U0001f49a",
    "travel": "\u2708\ufe0f",
    "health": "\U0001f3e5",
    "dental": "\U0001f9b7",
    "mortgage": "\U0001f3e0",
    "business": "\U0001f4bc",
    "apartment": "\U0001f3e2",
}

# In-memory conversation history (mirrors api/routes/chat.py)
conversations: dict[str, list] = {}

# Shared resources
resources = AppResources()


def _build_agent_input(query: str, history: list, language: str = "he") -> dict:
    """Build the agent input dict matching the expected AgentState."""
    return {
        "query": query,
        "messages": history,
        "rewritten_query": "",
        "detected_domains": [],
        "detected_language": language,
        "retrieved_documents": [],
        "graded_documents": [],
        "generation": "",
        "citations": [],
        "is_grounded": False,
        "retry_count": 0,
        "should_fallback": False,
        "quality_action": "",
        "quality_reasoning": "",
        "reasoning_trace": [],
    }


def _format_domain_label(domain: str | None) -> str:
    """Format domain name with emoji for Telegram display."""
    if not domain or domain not in DOMAINS:
        return ""
    icon = DOMAIN_ICONS.get(domain, "")
    name_he = DOMAINS[domain].name_he
    return f"{icon} {name_he}\n\n"


def _format_citations(citations: list[dict]) -> str:
    """Format citations as a compact numbered list."""
    if not citations:
        return ""

    lines = ["\n\n\U0001f4ce <b>Sources:</b>"]
    for i, c in enumerate(citations, 1):
        title = c.get("document_title", "")
        section = c.get("section", "")
        url = c.get("source_url", "")

        label = title
        if section:
            label += f" - {section}"

        if url:
            lines.append(f"{i}. <a href=\"{url}\">{_escape_html(label)}</a>")
        else:
            lines.append(f"{i}. {_escape_html(label)}")

    return "\n".join(lines)


def _escape_html(text: str) -> str:
    """Escape HTML special characters for Telegram HTML parse mode."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _split_message(text: str, max_length: int = 4096) -> list[str]:
    """Split a long message into chunks that fit Telegram's limit."""
    if len(text) <= max_length:
        return [text]

    chunks = []
    while text:
        if len(text) <= max_length:
            chunks.append(text)
            break

        # Try to split at a newline
        split_at = text.rfind("\n", 0, max_length)
        if split_at == -1:
            # Fall back to splitting at space
            split_at = text.rfind(" ", 0, max_length)
        if split_at == -1:
            # Hard split
            split_at = max_length

        chunks.append(text[:split_at])
        text = text[split_at:].lstrip("\n")

    return chunks


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command."""
    welcome = (
        "\U0001f44b <b>Harel Insurance Chatbot</b>\n\n"
        "I can help you with questions about Harel Insurance products:\n\n"
        "\U0001f697 Car  \u2022  \U0001f49a Life  \u2022  \u2708\ufe0f Travel  \u2022  \U0001f3e5 Health\n"
        "\U0001f9b7 Dental  \u2022  \U0001f3e0 Mortgage  \u2022  \U0001f4bc Business  \u2022  \U0001f3e2 Apartment\n\n"
        "Just type your question in Hebrew or English.\n\n"
        "\U0001f1ee\U0001f1f1 <b>\u05e6\u05d0\u05d8\u05d1\u05d5\u05d8 \u05d1\u05d9\u05d8\u05d5\u05d7 \u05d4\u05e8\u05d0\u05dc</b>\n\n"
        "\u05d0\u05e0\u05d9 \u05d9\u05db\u05d5\u05dc \u05dc\u05e2\u05d6\u05d5\u05e8 \u05dc\u05da \u05d1\u05e9\u05d0\u05dc\u05d5\u05ea \u05e2\u05dc \u05de\u05d5\u05e6\u05e8\u05d9 \u05d1\u05d9\u05d8\u05d5\u05d7 \u05d4\u05e8\u05d0\u05dc.\n"
        "\u05e4\u05e9\u05d5\u05d8 \u05db\u05ea\u05d1\u05d5 \u05d0\u05ea \u05d4\u05e9\u05d0\u05dc\u05d4 \u05e9\u05dc\u05db\u05dd."
    )
    await update.message.reply_text(welcome, parse_mode=ParseMode.HTML)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help command."""
    help_text = (
        "<b>How to use this bot:</b>\n\n"
        "\u2022 Send any question about Harel Insurance\n"
        "\u2022 I'll search the knowledge base and provide an answer with sources\n"
        "\u2022 Supports Hebrew and English\n\n"
        "<b>Commands:</b>\n"
        "/start - Welcome message\n"
        "/help - This help message"
    )
    await update.message.reply_text(help_text, parse_mode=ParseMode.HTML)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Process an incoming user message through the agent pipeline."""
    message_text = update.message.text
    if not message_text:
        return

    chat_id = str(update.effective_chat.id)
    logger.info(f"Telegram message from chat {chat_id}: {message_text[:80]}...")

    # Show typing indicator
    await update.message.chat.send_action(ChatAction.TYPING)

    # Get conversation history
    history = conversations.get(chat_id, [])

    # Build agent input and run the pipeline in a thread to not block the event loop
    agent_input = _build_agent_input(query=message_text, history=history)

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, resources.agent.invoke, agent_input)
    except Exception:
        logger.exception("Agent invocation failed")
        await update.message.reply_text(
            "\u274c An error occurred while processing your request. Please try again.",
        )
        return

    # Extract results
    answer = result.get("generation", "")
    citations_raw = result.get("citations", [])
    domains = result.get("detected_domains", [])
    domain = domains[0] if domains else None

    # Update conversation history
    history.append(HumanMessage(content=message_text))
    history.append(AIMessage(content=answer))
    conversations[chat_id] = history[-10:]  # Keep last 5 turns

    # Format response
    domain_label = _format_domain_label(domain)
    citations_text = _format_citations(citations_raw)
    full_response = f"{domain_label}{_escape_html(answer)}{citations_text}"

    # Send (split if too long)
    for chunk in _split_message(full_response):
        await update.message.reply_text(
            chunk,
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
        )


def run_bot() -> None:
    """Initialize resources and start the Telegram bot."""
    token = settings.telegram_bot_token
    if not token:
        logger.error(
            "TELEGRAM_BOT_TOKEN not set. "
            "Add it to your .env file or set the environment variable."
        )
        raise SystemExit(1)

    logger.info("Initializing chatbot resources...")
    resources.initialize()

    logger.info("Starting Telegram bot...")
    app = Application.builder().token(token).build()

    # Register handlers
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Bot is running. Press Ctrl+C to stop.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)
