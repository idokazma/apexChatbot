"""Prompt templates for the quizzer module."""

QUESTION_GENERATION_PROMPT = """You are a test question generator for an insurance customer support chatbot.
You will be given document chunks from Harel Insurance. Your job is to create a realistic customer question
whose answer can be found in these documents.

DOCUMENTS:
{documents}

INSTRUCTIONS:
1. Read and analyze the documents carefully.
2. Generate a question of type: {question_type}
   Type description: {type_description}
3. The question MUST be answerable from the provided documents.
4. Write the question in {language} as a casual customer message (not formal, like someone typing in a chat).
   - Include natural imperfections: colloquial phrasing, occasional typos, abbreviations.
   - Vary between short direct questions and longer context-providing messages.
5. Difficulty level: {difficulty}
   - easy: Straightforward, answer is directly stated in one document.
   - medium: Requires synthesizing info from the document or understanding nuance.
   - hard: Requires combining info from multiple parts, understanding conditions/exceptions.

Respond with ONLY a JSON object:
{{
  "question": "<the casual customer question>",
  "ground_truth_answer": "<the correct, concise answer based ONLY on the documents — state the facts directly, in the same language as the question>",
  "expected_answer_hints": "<2-4 key bullet points the answer should contain, in the same language>",
  "answerable": true
}}

If you genuinely cannot create a {question_type} question from these documents, respond:
{{
  "question": "",
  "expected_answer_hints": "",
  "answerable": false
}}"""


ANSWER_SCORING_PROMPT = """You are an expert evaluator for an insurance customer support chatbot.
Score how well the chatbot answered a customer's question.

QUESTION: {question}
QUESTION TYPE: {question_type}

GROUND TRUTH ANSWER (correct answer derived from source documents):
{ground_truth_answer}

EXPECTED ANSWER HINTS (key points from source documents):
{expected_hints}

CHATBOT'S ANSWER:
{answer}

CITATIONS PROVIDED:
{citations}

SOURCE DOCUMENTS (ground truth):
{source_docs}

Score the answer on these dimensions (0.0 to 1.0 each):

1. **correctness**: Is the information factually correct based on the source documents?
   - 1.0: All facts match source documents
   - 0.5: Partially correct, some inaccuracies
   - 0.0: Incorrect or fabricated information

2. **completeness**: Does the answer cover all key points from the expected hints?
   - 1.0: All key points addressed
   - 0.5: Some key points covered
   - 0.0: Key points missed entirely

3. **citation_quality**: Are citations present, accurate, and pointing to the right sources?
   - 1.0: All claims properly cited with correct sources
   - 0.5: Some citations but incomplete or partially wrong
   - 0.0: No citations or all citations wrong

4. **relevance**: Does the answer directly address what was asked?
   - 1.0: Directly and precisely answers the question
   - 0.5: Somewhat relevant but wandering
   - 0.0: Off-topic or irrelevant

5. **tone**: Is the answer professional, clear, and appropriate for customer support?
   - 1.0: Professional, clear, well-structured
   - 0.5: Acceptable but could be better
   - 0.0: Unprofessional, confusing, or inappropriate

6. **type_accuracy**: Does the answer match the expected format for this question type?
{type_specific_criteria}

Respond with ONLY a JSON object:
{{
  "correctness": <float>,
  "completeness": <float>,
  "citation_quality": <float>,
  "relevance": <float>,
  "tone": <float>,
  "type_accuracy": <float>,
  "reasoning": "<brief explanation of strengths and weaknesses>"
}}"""


# Type-specific scoring criteria injected into the prompt
TYPE_SCORING_CRITERIA: dict[str, str] = {
    "yes_no": (
        "   This is a YES/NO question. Score type_accuracy based on:\n"
        "   - 1.0: Answer gives a clear, unambiguous yes or no upfront, then explains\n"
        "   - 0.5: Answer implies yes/no but never states it clearly, or buries it in text\n"
        "   - 0.0: Answer avoids committing to yes or no, or gives a contradictory response"
    ),
    "numerical": (
        "   This is a NUMERICAL question (amounts, percentages, limits, dates, durations). Score type_accuracy based on:\n"
        "   - 1.0: Answer states the exact number/amount/date from source documents clearly\n"
        "   - 0.5: Answer gives a range or approximate number, or the number is present but buried\n"
        "   - 0.0: Answer omits the specific number entirely, or gives a wrong number"
    ),
    "conditional": (
        "   This is a CONDITIONAL question (conditions, exceptions, what-if scenarios). Score type_accuracy based on:\n"
        "   - 1.0: Answer clearly states the conditions/exceptions and their consequences\n"
        "   - 0.5: Answer mentions some conditions but misses important exceptions or edge cases\n"
        "   - 0.0: Answer ignores the conditional nature and gives a generic response"
    ),
    "factual": (
        "   This is a FACTUAL question (direct facts about coverage, terms, policy). Score type_accuracy based on:\n"
        "   - 1.0: Answer directly and concisely states the requested facts with proper sources\n"
        "   - 0.5: Answer is correct but includes excessive irrelevant information\n"
        "   - 0.0: Answer fails to provide the core facts that were asked about"
    ),
    "comparison": (
        "   This is a COMPARISON question (comparing options, plans, aspects). Score type_accuracy based on:\n"
        "   - 1.0: Answer clearly compares both/all sides with specific differences highlighted\n"
        "   - 0.5: Answer describes the items but doesn't clearly contrast them\n"
        "   - 0.0: Answer only describes one side or doesn't address the comparison at all"
    ),
    "procedural": (
        "   This is a PROCEDURAL question (how-to, step-by-step processes). Score type_accuracy based on:\n"
        "   - 1.0: Answer provides clear sequential steps or a well-structured process\n"
        "   - 0.5: Answer describes the process but not in a clear step-by-step manner\n"
        "   - 0.0: Answer doesn't explain the process or gives unstructured information"
    ),
}


REPORT_ANALYSIS_PROMPT = """You are a senior QA engineer analyzing the performance report of an insurance
customer support chatbot that was tested with {total_questions} automated questions.

OVERALL METRICS:
{overall_metrics}

PERFORMANCE BY QUESTION TYPE:
{by_question_type}

PERFORMANCE BY INSURANCE DOMAIN:
{by_domain}

WORST PERFORMING EXAMPLES:
{worst_examples}

BEST PERFORMING EXAMPLES:
{best_examples}

Based on this data, provide:

1. **Executive Summary** (2-3 sentences): Overall assessment of chatbot quality.

2. **Key Strengths** (3-5 bullet points): Where the chatbot performs well.

3. **Critical Weaknesses** (3-5 bullet points): Where the chatbot struggles most.

4. **Improvement Recommendations** (5-8 actionable items):
   - Be specific: mention which question types, domains, or patterns need work.
   - Suggest concrete technical changes (e.g., "improve chunking for mortgage documents",
     "add more keywords for dental coverage terms", "improve citation extraction for
     numerical answers").
   - Prioritize by potential impact.

5. **Domain-Specific Notes**: For each insurance domain, note specific issues or strengths.

6. **Question Type Analysis**: For each question type, explain why the chatbot performs
   as it does and what could help.

Format your response as clean HTML sections (no full page HTML, just content divs).
Use <h3>, <p>, <ul><li>, and <strong> tags for formatting."""
