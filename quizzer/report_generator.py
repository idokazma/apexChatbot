"""Generate a visual HTML report from quiz run results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from llm.gemini_client import GeminiClient
from quizzer.prompts import REPORT_ANALYSIS_PROMPT

if TYPE_CHECKING:
    from quizzer.runner import QuizRunResult


def generate_report(
    result: QuizRunResult,
    output_path: Path,
    client: GeminiClient | None = None,
) -> None:
    """Generate a comprehensive visual HTML report.

    Includes:
        - Executive dashboard with KPI cards
        - Score distribution charts (Chart.js)
        - Breakdown by question type, domain, difficulty, language
        - Cross-tabulation heatmap (domain x question type)
        - Best and worst examples
        - LLM-generated improvement suggestions

    Args:
        result: The completed QuizRunResult.
        output_path: Path to write the HTML report file.
        client: Optional GeminiClient for generating improvement analysis.
    """
    data = result.to_dict()
    overall = data.get("overall_metrics", {})
    by_type = data.get("by_question_type", {})
    by_domain = data.get("by_domain", {})
    by_difficulty = data.get("by_difficulty", {})
    by_language = data.get("by_language", {})
    details = data.get("details", [])
    summary = data.get("summary", {})

    # Sort details by overall score for best/worst
    scored_details = [d for d in details if d.get("success", False)]
    scored_details.sort(key=lambda d: d.get("overall_score", 0))

    worst_10 = scored_details[:10]
    best_10 = scored_details[-10:][::-1]

    # Build cross-tabulation: domain x question_type
    cross_tab = _build_cross_tab(details)

    # Generate LLM analysis
    llm_analysis = _generate_llm_analysis(
        overall, by_type, by_domain, worst_10, best_10, client,
    )

    # Build HTML
    html = _build_html(
        summary=summary,
        overall=overall,
        by_type=by_type,
        by_domain=by_domain,
        by_difficulty=by_difficulty,
        by_language=by_language,
        cross_tab=cross_tab,
        best_10=best_10,
        worst_10=worst_10,
        llm_analysis=llm_analysis,
        details=details,
    )

    output_path.write_text(html, encoding="utf-8")
    logger.info(f"Report written to {output_path}")


def _build_cross_tab(details: list[dict]) -> dict[str, dict[str, dict]]:
    """Build a domain x question_type cross-tabulation of avg scores."""
    groups: dict[tuple[str, str], list[float]] = {}
    for d in details:
        key = (d.get("domain", "unknown"), d.get("question_type", "unknown"))
        groups.setdefault(key, []).append(d.get("overall_score", 0.0))

    domains = sorted({k[0] for k in groups})
    qtypes = sorted({k[1] for k in groups})

    table: dict[str, dict[str, dict]] = {}
    for dom in domains:
        table[dom] = {}
        for qt in qtypes:
            scores = groups.get((dom, qt), [])
            if scores:
                table[dom][qt] = {
                    "avg": round(sum(scores) / len(scores), 3),
                    "count": len(scores),
                }
            else:
                table[dom][qt] = {"avg": None, "count": 0}

    return table


def _generate_llm_analysis(
    overall: dict,
    by_type: dict,
    by_domain: dict,
    worst_10: list[dict],
    best_10: list[dict],
    client: GeminiClient | None,
) -> str:
    """Use Gemini to generate actionable analysis and improvement suggestions."""
    if not client:
        return "<p><em>LLM analysis skipped (no Gemini client provided).</em></p>"

    def _fmt_examples(examples: list[dict]) -> str:
        parts = []
        for e in examples[:5]:
            parts.append(
                f"- Q [{e.get('question_type', '?')}/{e.get('domain', '?')}]: "
                f"{e.get('question', '')[:100]}\n"
                f"  Score: {e.get('overall_score', 0):.2f} | "
                f"Correctness: {e.get('correctness', 0):.2f} | "
                f"Citations: {e.get('citation_quality', 0):.2f}\n"
                f"  Answer snippet: {e.get('answer', '')[:150]}"
            )
        return "\n".join(parts)

    prompt = REPORT_ANALYSIS_PROMPT.format(
        total_questions=overall.get("count", 0),
        overall_metrics=json.dumps(overall, indent=2),
        by_question_type=json.dumps(by_type, indent=2),
        by_domain=json.dumps(by_domain, indent=2),
        worst_examples=_fmt_examples(worst_10),
        best_examples=_fmt_examples(best_10),
    )

    try:
        response = client.generate(prompt, temperature=0.3, max_tokens=4096)
        return response
    except Exception as e:
        logger.error(f"LLM analysis generation failed: {e}")
        return f"<p><em>LLM analysis failed: {e}</em></p>"


def _score_color(value: float | None) -> str:
    """Return a CSS color for a score value (green=good, red=bad)."""
    if value is None:
        return "#6b7280"
    if value >= 0.8:
        return "#059669"
    if value >= 0.6:
        return "#d97706"
    if value >= 0.4:
        return "#ea580c"
    return "#dc2626"


def _score_bg(value: float | None) -> str:
    """Return a CSS background color for heatmap cells."""
    if value is None:
        return "#f3f4f6"
    if value >= 0.8:
        return "#d1fae5"
    if value >= 0.6:
        return "#fef3c7"
    if value >= 0.4:
        return "#ffedd5"
    return "#fee2e2"


def _build_html(
    summary: dict,
    overall: dict,
    by_type: dict,
    by_domain: dict,
    by_difficulty: dict,
    by_language: dict,
    cross_tab: dict,
    best_10: list[dict],
    worst_10: list[dict],
    llm_analysis: str,
    details: list[dict],
) -> str:
    """Build the complete HTML report."""
    # --- KPI Cards ---
    kpi_cards = _build_kpi_cards(summary, overall)

    # --- Charts data ---
    charts_js = _build_charts_js(overall, by_type, by_domain, by_difficulty, details)

    # --- Tables ---
    type_table = _build_breakdown_table(by_type, "Question Type")
    domain_table = _build_breakdown_table(by_domain, "Insurance Domain")
    difficulty_table = _build_breakdown_table(by_difficulty, "Difficulty")
    language_table = _build_breakdown_table(by_language, "Language")

    # --- Cross-tab heatmap ---
    heatmap = _build_heatmap(cross_tab)

    # --- Examples ---
    best_html = _build_examples_table(best_10, "Top Performing")
    worst_html = _build_examples_table(worst_10, "Worst Performing")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quizzer Report - Harel Insurance Chatbot</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f8fafc; color: #1e293b; line-height: 1.6;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; padding: 24px; }}
        h1 {{ font-size: 28px; margin-bottom: 8px; color: #0f172a; }}
        h2 {{
            font-size: 20px; margin: 32px 0 16px; color: #334155;
            border-bottom: 2px solid #e2e8f0; padding-bottom: 8px;
        }}
        h3 {{ font-size: 16px; margin: 16px 0 8px; color: #475569; }}
        .subtitle {{ color: #64748b; margin-bottom: 24px; }}
        .kpi-grid {{
            display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px; margin-bottom: 32px;
        }}
        .kpi-card {{
            background: white; border-radius: 12px; padding: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1); text-align: center;
        }}
        .kpi-card .value {{ font-size: 32px; font-weight: 700; }}
        .kpi-card .label {{ font-size: 13px; color: #64748b; margin-top: 4px; }}
        .chart-grid {{
            display: grid; grid-template-columns: repeat(auto-fit, minmax(450px, 1fr));
            gap: 24px; margin-bottom: 32px;
        }}
        .chart-box {{
            background: white; border-radius: 12px; padding: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .chart-box canvas {{ max-height: 350px; }}
        table {{
            width: 100%; border-collapse: collapse; background: white;
            border-radius: 12px; overflow: hidden;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 24px;
        }}
        th {{ background: #f1f5f9; padding: 12px 16px; text-align: left; font-size: 13px;
              color: #475569; font-weight: 600; }}
        td {{ padding: 10px 16px; border-top: 1px solid #e2e8f0; font-size: 13px; }}
        tr:hover {{ background: #f8fafc; }}
        .score-badge {{
            display: inline-block; padding: 2px 8px; border-radius: 6px;
            font-weight: 600; font-size: 12px; color: white;
        }}
        .analysis-box {{
            background: white; border-radius: 12px; padding: 24px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 24px;
        }}
        .analysis-box h3 {{ color: #1e40af; }}
        .analysis-box ul {{ margin-left: 20px; }}
        .analysis-box li {{ margin-bottom: 6px; }}
        .heatmap td {{ text-align: center; font-weight: 600; font-size: 12px; }}
        .example-q {{ color: #1e40af; font-weight: 500; }}
        .example-a {{ color: #475569; font-style: italic; font-size: 12px; }}
        .footer {{ text-align: center; color: #94a3b8; margin-top: 40px; font-size: 12px; }}
    </style>
</head>
<body>
<div class="container">
    <h1>Quizzer Performance Report</h1>
    <p class="subtitle">
        Harel Insurance Chatbot &mdash;
        {summary.get('questions_asked', 0)} questions tested |
        {summary.get('total_time_s', 0):.0f}s total runtime
    </p>

    {kpi_cards}

    <h2>Score Distributions</h2>
    <div class="chart-grid">
        <div class="chart-box"><canvas id="chartByType"></canvas></div>
        <div class="chart-box"><canvas id="chartByDomain"></canvas></div>
        <div class="chart-box"><canvas id="chartScoreDist"></canvas></div>
        <div class="chart-box"><canvas id="chartByDifficulty"></canvas></div>
        <div class="chart-box"><canvas id="chartMetrics"></canvas></div>
        <div class="chart-box"><canvas id="chartLatency"></canvas></div>
    </div>

    <h2>Breakdown by Question Type</h2>
    {type_table}

    <h2>Breakdown by Insurance Domain</h2>
    {domain_table}

    <h2>Breakdown by Difficulty</h2>
    {difficulty_table}

    <h2>Breakdown by Language</h2>
    {language_table}

    <h2>Domain x Question Type Heatmap</h2>
    {heatmap}

    <h2>Best Performing Examples</h2>
    {best_html}

    <h2>Worst Performing Examples</h2>
    {worst_html}

    <h2>LLM Analysis &amp; Improvement Suggestions</h2>
    <div class="analysis-box">
        {llm_analysis}
    </div>

    <p class="footer">
        Generated by Quizzer Module | Harel Insurance Chatbot Evaluation Suite
    </p>
</div>

<script>
{charts_js}
</script>
</body>
</html>"""


def _build_kpi_cards(summary: dict, overall: dict) -> str:
    """Build the KPI cards row."""
    cards = [
        ("Overall Score", f"{overall.get('avg_overall', 0):.1%}",
         _score_color(overall.get("avg_overall", 0))),
        ("Correctness", f"{overall.get('avg_correctness', 0):.1%}",
         _score_color(overall.get("avg_correctness", 0))),
        ("Completeness", f"{overall.get('avg_completeness', 0):.1%}",
         _score_color(overall.get("avg_completeness", 0))),
        ("Citation Quality", f"{overall.get('avg_citation_quality', 0):.1%}",
         _score_color(overall.get("avg_citation_quality", 0))),
        ("Relevance", f"{overall.get('avg_relevance', 0):.1%}",
         _score_color(overall.get("avg_relevance", 0))),
        ("Domain Routing", f"{overall.get('domain_match_rate', 0):.1%}",
         _score_color(overall.get("domain_match_rate", 0))),
        ("Avg Latency", f"{overall.get('avg_latency_s', 0):.1f}s", "#6366f1"),
        ("Failures", f"{summary.get('api_failures', 0)}",
         "#dc2626" if summary.get("api_failures", 0) > 0 else "#059669"),
    ]

    html_parts = ['<div class="kpi-grid">']
    for label, value, color in cards:
        html_parts.append(
            f'<div class="kpi-card">'
            f'<div class="value" style="color:{color}">{value}</div>'
            f'<div class="label">{label}</div>'
            f'</div>'
        )
    html_parts.append('</div>')
    return "\n".join(html_parts)


def _build_breakdown_table(data: dict, group_name: str) -> str:
    """Build a breakdown table for a dimension (type, domain, etc.)."""
    cols = [
        ("count", "Count"),
        ("avg_overall", "Overall"),
        ("avg_correctness", "Correctness"),
        ("avg_completeness", "Completeness"),
        ("avg_relevance", "Relevance"),
        ("avg_citation_quality", "Citation"),
        ("avg_tone", "Tone"),
        ("avg_efficiency", "Efficiency"),
        ("avg_latency_s", "Latency (s)"),
        ("domain_match_rate", "Domain Match"),
        ("failure_rate", "Failure Rate"),
    ]

    header = f"<th>{group_name}</th>" + "".join(f"<th>{c[1]}</th>" for c in cols)

    rows = []
    for name, metrics in sorted(data.items()):
        cells = [f"<td><strong>{name}</strong></td>"]
        for key, _ in cols:
            val = metrics.get(key, 0)
            if key == "count":
                cells.append(f"<td>{val}</td>")
            elif key == "avg_latency_s":
                cells.append(f"<td>{val:.1f}</td>")
            else:
                color = _score_color(val) if key != "failure_rate" else _score_color(1 - val)
                cells.append(
                    f'<td><span class="score-badge" '
                    f'style="background:{color}">{val:.1%}</span></td>'
                )
        rows.append("<tr>" + "".join(cells) + "</tr>")

    return f"""<table>
    <thead><tr>{header}</tr></thead>
    <tbody>{"".join(rows)}</tbody>
    </table>"""


def _build_heatmap(cross_tab: dict) -> str:
    """Build a domain x question_type heatmap table."""
    if not cross_tab:
        return "<p>No cross-tabulation data available.</p>"

    # Collect all question types
    all_qtypes: set[str] = set()
    for qt_map in cross_tab.values():
        all_qtypes.update(qt_map.keys())
    qtypes = sorted(all_qtypes)

    header = '<th>Domain \\ QType</th>' + "".join(f"<th>{qt}</th>" for qt in qtypes)

    rows = []
    for domain in sorted(cross_tab.keys()):
        cells = [f"<td><strong>{domain}</strong></td>"]
        for qt in qtypes:
            cell = cross_tab[domain].get(qt, {"avg": None, "count": 0})
            avg = cell["avg"]
            count = cell["count"]
            if avg is not None:
                bg = _score_bg(avg)
                cells.append(
                    f'<td style="background:{bg}">{avg:.0%}<br>'
                    f'<small style="color:#94a3b8">n={count}</small></td>'
                )
            else:
                cells.append('<td style="background:#f3f4f6;color:#94a3b8">-</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")

    return f"""<table class="heatmap">
    <thead><tr>{header}</tr></thead>
    <tbody>{"".join(rows)}</tbody>
    </table>"""


def _build_examples_table(examples: list[dict], title: str) -> str:
    """Build a table of example question-answer pairs with scores."""
    if not examples:
        return f"<p>No {title.lower()} examples available.</p>"

    rows = []
    for e in examples:
        score = e.get("overall_score", 0)
        color = _score_color(score)
        rows.append(f"""<tr>
            <td><span class="score-badge" style="background:{color}">{score:.0%}</span></td>
            <td>{e.get('question_type', '?')}</td>
            <td>{e.get('domain', '?')}</td>
            <td><span class="example-q">{_esc(e.get('question', '')[:120])}</span></td>
            <td><span class="example-a">{_esc(e.get('answer', '')[:200])}</span></td>
            <td>{e.get('correctness', 0):.0%}</td>
            <td>{e.get('citation_quality', 0):.0%}</td>
            <td>{e.get('latency_s', 0):.1f}s</td>
        </tr>""")

    return f"""<table>
    <thead><tr>
        <th>Score</th><th>Type</th><th>Domain</th><th>Question</th>
        <th>Answer (snippet)</th><th>Correct</th><th>Citation</th><th>Latency</th>
    </tr></thead>
    <tbody>{"".join(rows)}</tbody>
    </table>"""


def _build_charts_js(
    overall: dict,
    by_type: dict,
    by_domain: dict,
    by_difficulty: dict,
    details: list[dict],
) -> str:
    """Build Chart.js initialization JavaScript."""
    # Chart 1: By question type (bar)
    type_labels = json.dumps(list(by_type.keys()))
    type_scores = json.dumps([m.get("avg_overall", 0) for m in by_type.values()])
    type_counts = json.dumps([m.get("count", 0) for m in by_type.values()])

    # Chart 2: By domain (bar)
    domain_labels = json.dumps(list(by_domain.keys()))
    domain_scores = json.dumps([m.get("avg_overall", 0) for m in by_domain.values()])

    # Chart 3: Score distribution (histogram)
    bins = [0] * 10  # 0-10%, 10-20%, ..., 90-100%
    for d in details:
        s = d.get("overall_score", 0)
        idx = min(int(s * 10), 9)
        bins[idx] += 1
    bin_labels = json.dumps([f"{i*10}-{(i+1)*10}%" for i in range(10)])
    bin_values = json.dumps(bins)

    # Chart 4: By difficulty (bar)
    diff_labels = json.dumps(list(by_difficulty.keys()))
    diff_scores = json.dumps([m.get("avg_overall", 0) for m in by_difficulty.values()])

    # Chart 5: Metric radar
    metric_labels = json.dumps([
        "Correctness", "Completeness", "Relevance",
        "Citation Quality", "Tone", "Efficiency",
    ])
    metric_values = json.dumps([
        overall.get("avg_correctness", 0),
        overall.get("avg_completeness", 0),
        overall.get("avg_relevance", 0),
        overall.get("avg_citation_quality", 0),
        overall.get("avg_tone", 0),
        overall.get("avg_efficiency", 0),
    ])

    # Chart 6: Latency distribution
    lat_bins = [0] * 8  # 0-5, 5-10, ..., 30-35, 35+
    for d in details:
        lat = d.get("latency_s", 0)
        idx = min(int(lat / 5), 7)
        lat_bins[idx] += 1
    lat_labels = json.dumps(["0-5s", "5-10s", "10-15s", "15-20s", "20-25s", "25-30s", "30-35s", "35s+"])
    lat_values = json.dumps(lat_bins)

    return f"""
const COLORS = [
    '#3b82f6','#10b981','#f59e0b','#ef4444','#8b5cf6',
    '#ec4899','#06b6d4','#84cc16','#f97316','#6366f1'
];

// Chart 1: By Question Type
new Chart(document.getElementById('chartByType'), {{
    type: 'bar',
    data: {{
        labels: {type_labels},
        datasets: [{{
            label: 'Avg Overall Score',
            data: {type_scores},
            backgroundColor: COLORS.slice(0, {len(by_type)}),
            borderRadius: 6,
        }}]
    }},
    options: {{
        responsive: true,
        plugins: {{ title: {{ display: true, text: 'Performance by Question Type' }} }},
        scales: {{ y: {{ beginAtZero: true, max: 1.0, ticks: {{ callback: v => (v*100)+'%' }} }} }}
    }}
}});

// Chart 2: By Domain
new Chart(document.getElementById('chartByDomain'), {{
    type: 'bar',
    data: {{
        labels: {domain_labels},
        datasets: [{{
            label: 'Avg Overall Score',
            data: {domain_scores},
            backgroundColor: COLORS.slice(0, {len(by_domain)}),
            borderRadius: 6,
        }}]
    }},
    options: {{
        responsive: true,
        plugins: {{ title: {{ display: true, text: 'Performance by Insurance Domain' }} }},
        scales: {{ y: {{ beginAtZero: true, max: 1.0, ticks: {{ callback: v => (v*100)+'%' }} }} }}
    }}
}});

// Chart 3: Score Distribution
new Chart(document.getElementById('chartScoreDist'), {{
    type: 'bar',
    data: {{
        labels: {bin_labels},
        datasets: [{{
            label: 'Number of Questions',
            data: {bin_values},
            backgroundColor: '#6366f1',
            borderRadius: 6,
        }}]
    }},
    options: {{
        responsive: true,
        plugins: {{ title: {{ display: true, text: 'Overall Score Distribution' }} }},
        scales: {{ y: {{ beginAtZero: true }} }}
    }}
}});

// Chart 4: By Difficulty
new Chart(document.getElementById('chartByDifficulty'), {{
    type: 'bar',
    data: {{
        labels: {diff_labels},
        datasets: [{{
            label: 'Avg Overall Score',
            data: {diff_scores},
            backgroundColor: ['#10b981', '#f59e0b', '#ef4444'],
            borderRadius: 6,
        }}]
    }},
    options: {{
        responsive: true,
        plugins: {{ title: {{ display: true, text: 'Performance by Difficulty' }} }},
        scales: {{ y: {{ beginAtZero: true, max: 1.0, ticks: {{ callback: v => (v*100)+'%' }} }} }}
    }}
}});

// Chart 5: Metrics Radar
new Chart(document.getElementById('chartMetrics'), {{
    type: 'radar',
    data: {{
        labels: {metric_labels},
        datasets: [{{
            label: 'Average Score',
            data: {metric_values},
            backgroundColor: 'rgba(99,102,241,0.2)',
            borderColor: '#6366f1',
            pointBackgroundColor: '#6366f1',
        }}]
    }},
    options: {{
        responsive: true,
        plugins: {{ title: {{ display: true, text: 'Metric Radar Overview' }} }},
        scales: {{ r: {{ beginAtZero: true, max: 1.0, ticks: {{ callback: v => (v*100)+'%' }} }} }}
    }}
}});

// Chart 6: Latency Distribution
new Chart(document.getElementById('chartLatency'), {{
    type: 'bar',
    data: {{
        labels: {lat_labels},
        datasets: [{{
            label: 'Number of Questions',
            data: {lat_values},
            backgroundColor: '#f59e0b',
            borderRadius: 6,
        }}]
    }},
    options: {{
        responsive: true,
        plugins: {{ title: {{ display: true, text: 'Latency Distribution' }} }},
        scales: {{ y: {{ beginAtZero: true }} }}
    }}
}});
"""


def _esc(text: str) -> str:
    """Escape HTML special characters."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
