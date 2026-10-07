"""MkDocs hook: render `<!-- quiz: NAME -->` markers from quizzes/NAME.yml.

Quiz file format:

    title: Chapter 0 — check your understanding
    questions:
      - q: Markdown prompt (math allowed)
        type: single | multi
        options:
          - {text: "...", correct: true, why: "why this is right/wrong"}
        explain: overall explanation shown after answering

The HTML uses `markdown` attributes (md_in_html) so prompts, options and
explanations go through the normal Markdown + KaTeX pipeline.
"""

import re
from pathlib import Path

import yaml
from mkdocs.exceptions import PluginError

MARKER = re.compile(r"<!--\s*quiz:\s*([\w.-]+)\s*-->")
QUIZ_DIR = Path(__file__).resolve().parents[2] / "quizzes"


def _block(cls: str, body: str, attrs: str = "") -> str:
    # Blank lines around the body are required for md_in_html to parse it as Markdown.
    return f'<div class="{cls}"{attrs} markdown="1">\n\n{body.strip()}\n\n</div>'


def _validate(name: str, i: int, q: dict) -> None:
    where = f"quiz {name}, question {i + 1}"
    for key in ("q", "type", "options", "explain"):
        if key not in q:
            raise PluginError(f"{where}: missing '{key}'")
    if q["type"] not in ("single", "multi"):
        raise PluginError(f"{where}: type must be 'single' or 'multi'")
    n_correct = sum(bool(o.get("correct")) for o in q["options"])
    if n_correct == 0 or (q["type"] == "single" and n_correct != 1):
        raise PluginError(f"{where}: {n_correct} correct options for type {q['type']}")
    for o in q["options"]:
        if "text" not in o or "why" not in o:
            raise PluginError(f"{where}: every option needs 'text' and 'why'")


def render(name: str) -> str:
    path = QUIZ_DIR / f"{name}.yml"
    if not path.exists():
        raise PluginError(f"quiz file not found: {path}")
    data = yaml.safe_load(path.read_text())
    parts = [f'<div class="quiz" markdown="1">\n\n**{data.get("title", "Quiz")}**\n']
    for i, q in enumerate(data["questions"]):
        _validate(name, i, q)
        kind = "Select one" if q["type"] == "single" else "Select all that apply"
        opts = "\n".join(
            _block(
                "quiz-opt",
                o["text"] + "\n\n" + _block("quiz-why", o["why"]),
                f' data-correct="{int(bool(o.get("correct")))}"',
            )
            for o in q["options"]
        )
        parts.append(
            _block(
                "quiz-q",
                _block("quiz-prompt", f"**Q{i + 1}.** {q['q']}\n\n*{kind}.*")
                + "\n"
                + opts
                + "\n"
                + _block("quiz-explain", "**Explanation.** " + q["explain"]),
                f' data-type="{q["type"]}"',
            )
        )
    parts.append("</div>")
    return "\n\n".join(parts)


def on_page_markdown(markdown, page, config, files):
    return MARKER.sub(lambda m: render(m.group(1)), markdown)
