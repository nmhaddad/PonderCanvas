"""Loads LLM-facing prompt content from the markdown/Jinja files in
`templates/`. Static instructions are read as-is; dynamic prompts are
rendered through Jinja2."""

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

_TEMPLATES_DIR = Path(__file__).parent / "templates"

_env = Environment(
    loader=FileSystemLoader(_TEMPLATES_DIR),
    autoescape=select_autoescape(enabled_extensions=()),
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=False,
)


def _read(name: str) -> str:
    return (_TEMPLATES_DIR / name).read_text().strip()


GENERATION_INSTRUCTION = _read("generation_instruction.md")
EVALUATION_INSTRUCTION = _read("evaluation_instruction.md")
LOOP_CONTROL_INSTRUCTION = _read("loop_control_instruction.md")


def build_extraction_prompt(user_prompt: str) -> str:
    return _env.get_template("extraction_prompt.md.j2").render(user_prompt=user_prompt).strip()


def build_generation_prompt(brief: dict, grounding: dict | None, feedback: dict | None) -> str:
    return (
        _env.get_template("generation_prompt.md.j2")
        .render(brief=brief, grounding=grounding, feedback=feedback)
        .strip()
    )


def build_eval_prompt(brief: dict, threshold: float) -> str:
    constraints = ", ".join(brief.get("constraints", [])) or "none"
    return (
        _env.get_template("eval_prompt.md.j2")
        .render(brief=brief, threshold=threshold, constraints=constraints)
        .strip()
    )
