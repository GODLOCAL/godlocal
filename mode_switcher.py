"""
mode_switcher.py — Parse [MODE: X] blocks from soul file and apply behaviour overrides.
Called by _pre_evolve_plan() and during sleep_cycle() to activate the right agent profile.
"""
import re
from pathlib import Path


SOUL_PATHS = [
    "godlocal_data/souls/god_soul.md",
    "god_soul.md",
    "god_soul.example.md",
]

AGENT_MODE_MAP = {
    "CODING":   "coding",
    "TRADING":  "trading",
    "WRITING":  "writing",
    "MEDICAL":  "medical",
    "SLEEP":    "sleep",
    "ANALYSIS": "coding",
}


def load_modes(root: str = ".") -> dict[str, dict]:
    """
    Parse all [MODE: X] ... [/MODE] blocks from the soul file.
    Returns dict of mode_name → rules_text.
    """
    root_path = Path(root)
    soul_text = ""
    for path in SOUL_PATHS:
        candidate = root_path / path
        if candidate.exists():
            soul_text = candidate.read_text(encoding="utf-8")
            break

    modes: dict[str, dict] = {}
    pattern = re.compile(
        r"## \[MODE: (\w+)\]\n(.*?)## \[/MODE\]",
        re.DOTALL
    )
    for match in pattern.finditer(soul_text):
        name  = match.group(1).upper()
        rules = match.group(2).strip()
        # Extract key-value hints from "# key: value" lines
        hints = {}
        for line in rules.splitlines():
            line = line.lstrip("# ").strip()
            if line.startswith("AgentPool swaps to:"):
                agent_str = line.split(":", 1)[1].strip()
                # e.g. "coding agent (DeepSeek-Coder-V2)"
                agent_name = agent_str.split()[0].lower()
                hints["agent"] = AGENT_MODE_MAP.get(name, agent_name)
            elif ": " in line and not line.startswith("-"):
                k, v = line.split(": ", 1)
                hints[k.lower().replace(" ", "_")] = v
        hints["rules_text"] = rules
        hints["agent"] = hints.get("agent", AGENT_MODE_MAP.get(name, "default"))
        modes[name] = hints

    return modes


def detect_mode(task: str, modes: dict[str, dict] | None = None) -> str:
    """
    Detect the best mode for a task string.
    Falls back to keyword matching if soul modes not loaded.
    """
    task_lower = task.lower()

    if modes:
        # Check each mode's rules_text for keyword overlap
        scores = {}
        for mode_name, hints in modes.items():
            rules = hints.get("rules_text", "").lower()
            keywords = re.findall(r"\b\w{4,}\b", rules)
            score = sum(1 for kw in keywords if kw in task_lower)
            scores[mode_name] = score
        best = max(scores, key=scores.get)
        if scores[best] > 0:
            return best

    # Keyword fallback
    if any(k in task_lower for k in ["trad", "market", "fund", "position", "signal"]):
        return "TRADING"
    if any(k in task_lower for k in ["write", "blog", "post", "copy", "draft"]):
        return "WRITING"
    if any(k in task_lower for k in ["medical", "mri", "dicom", "hipaa"]):
        return "MEDICAL"
    if any(k in task_lower for k in ["sleep", "consolidat", "hippocampal"]):
        return "SLEEP"
    return "CODING"


def get_agent_for_mode(mode: str, modes: dict[str, dict] | None = None) -> str:
    """Return the AgentPool agent slug for a given mode."""
    if modes and mode in modes:
        return modes[mode].get("agent", "default")
    return AGENT_MODE_MAP.get(mode, "default")


if __name__ == "__main__":
    modes = load_modes()
    print(f"Loaded {len(modes)} modes: {list(modes.keys())}")
    for name, hints in modes.items():
        print(f"  {name} → agent={hints.get('agent')} ")
