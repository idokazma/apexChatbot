"""Fallback node: returns safe response when evidence is insufficient."""

from loguru import logger

from agent.state import AgentState


FALLBACK_HE = (
    "מצטער, לא הצלחתי למצוא תשובה מדויקת לשאלה הזו. "
    "אני ממליץ לפנות לשירות הלקוחות של הראל ביטוח בטלפון *6060 "
    "או באתר https://www.harel-group.co.il — הם ישמחו לעזור."
)

FALLBACK_EN = (
    "I'm sorry, I wasn't able to find a precise answer to your question. "
    "I'd recommend reaching out to Harel Insurance customer service at *6060 "
    "or visiting https://www.harel-group.co.il — they'll be happy to help."
)


def fallback(state: AgentState) -> dict:
    """Return a safe fallback response."""
    language = state.get("detected_language", "he")
    message = FALLBACK_HE if language == "he" else FALLBACK_EN

    # Include partial info if available
    graded = state.get("graded_documents", [])
    if graded:
        partial = "\n\n"
        if language == "he":
            partial += "עם זאת, הנה מידע שעשוי לעזור לך:\n"
        else:
            partial += "That said, here's some information that might help:\n"

        for doc in graded[:2]:
            title = doc.get("source_doc_title", "")
            section = doc.get("section_path", "")
            content = doc.get("content", "")[:300]
            partial += f"\n- {title} ({section}):\n  {content}...\n"

        message += partial

    logger.info("Returning fallback response")
    return {
        "generation": message,
        "citations": [],
        "should_fallback": True,
    }
