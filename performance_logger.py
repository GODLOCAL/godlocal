"""
performance_logger.py — GodLocal ao-52 analog
Session signal persistence: records every interaction + correction signal.
sleep_cycle() Phase 3 analyzes this log and updates god_soul.md.

Inspired by @agent_wrapper "The Self-Improving AI System That Built Itself"
Key insight: most agents throw session signals away. GodLocal doesn't.
"""
import json
import time
import os
from datetime import datetime, date
from pathlib import Path
from typing import Optional

DATA_DIR = Path(os.getenv("GODLOCAL_DATA_DIR", "godlocal_data"))
PERF_LOG = DATA_DIR / "performance_log.jsonl"
SOUL_PATH = Path(os.getenv("GOD_SOUL_PATH", "god_soul.md"))
PATTERNS_PATH = DATA_DIR / "learned_patterns.md"


def _ensure_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def log_interaction(
    user_input: str,
    agent_response: str,
    was_corrected: bool = False,
    correction_note: Optional[str] = None,
    session_id: Optional[str] = None,
    tags: Optional[list] = None,
):
    """
    Record one interaction to the performance log.
    Called from chat handler after every response.

    Args:
        user_input:      Raw user message
        agent_response:  Agent's reply (truncated for storage efficiency)
        was_corrected:   True if user immediately corrected/re-asked
        correction_note: What the correction was (extracted from follow-up message)
        session_id:      Current session UUID
        tags:            Topic tags (e.g. ['solana', 'trading', 'sleep_cycle'])
    """
    _ensure_dir()
    entry = {
        "ts": datetime.utcnow().isoformat(),
        "date": date.today().isoformat(),
        "session_id": session_id,
        "user_input": user_input[:500],
        "response_preview": agent_response[:300],
        "was_corrected": was_corrected,
        "correction_note": correction_note,
        "tags": tags or [],
        "response_len": len(agent_response),
    }
    with PERF_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def get_recent_interactions(hours_back: int = 24) -> list[dict]:
    """Return interactions from the last N hours."""
    if not PERF_LOG.exists():
        return []
    cutoff = time.time() - hours_back * 3600
    results = []
    with PERF_LOG.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                ts = datetime.fromisoformat(entry["ts"]).timestamp()
                if ts >= cutoff:
                    results.append(entry)
            except Exception:
                continue
    return results


def get_correction_rate(hours_back: int = 24) -> float:
    """Returns fraction of interactions that triggered a correction."""
    interactions = get_recent_interactions(hours_back)
    if not interactions:
        return 0.0
    corrected = sum(1 for i in interactions if i.get("was_corrected"))
    return corrected / len(interactions)


def analyze_patterns(llm_fn, hours_back: int = 24) -> dict:
    """
    Analyze recent interactions to extract behavioral patterns.
    Returns dict with patterns and soul_updates list.
    Called by sleep_cycle() Phase 3.
    """
    interactions = get_recent_interactions(hours_back)
    if len(interactions) < 3:
        return {"status": "insufficient_data", "count": len(interactions)}

    # Build analysis prompt
    correction_rate = get_correction_rate(hours_back)
    sample = interactions[-30:]  # last 30 interactions

    lines = []
    for i in sample:
        prefix = "⚠️ CORRECTED" if i["was_corrected"] else "✅"
        lines.append(f"{prefix} Q: {i['user_input'][:120]}")
        if i["correction_note"]:
            lines.append(f"   Correction: {i['correction_note']}")

    prompt = f"""Analyze these GodLocal AI agent interactions (last {hours_back}h).
Correction rate: {correction_rate:.1%}

Interactions:
{chr(10).join(lines)}

Extract:
1. PATTERNS: What does the user frequently ask about? (max 5 bullet points)
2. PREFERENCES: What response style works best? (max 3 bullet points)  
3. SOUL_UPDATES: Concrete lines to add to god_soul.md under [LEARNED_PATTERNS] section (max 5 lines)
4. GAPS: Topics where agent struggled (max 3 bullet points)

Format as JSON with keys: patterns, preferences, soul_updates, gaps"""

    try:
        raw = llm_fn(prompt, max_tokens=600)
        # Try to parse JSON from response
        import re
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
        else:
            result = {"raw_analysis": raw}

        result["correction_rate"] = correction_rate
        result["interactions_analyzed"] = len(sample)
        result["status"] = "ok"
        return result
    except Exception as e:
        return {"status": "error", "error": str(e), "correction_rate": correction_rate}


def update_soul_with_patterns(patterns_result: dict, soul_path: Path = SOUL_PATH):
    """
    Append learned patterns to god_soul.md under [LEARNED_PATTERNS] section.
    Creates the section if it doesn't exist. Idempotent — checks for duplicates.
    """
    if patterns_result.get("status") != "ok":
        return False
    soul_updates = patterns_result.get("soul_updates", [])
    if not soul_updates:
        return False

    content = soul_path.read_text(encoding="utf-8") if soul_path.exists() else ""

    timestamp = datetime.utcnow().strftime("%Y-%m-%d")
    new_block = f"\n\n## [LEARNED_PATTERNS] — Auto-updated {timestamp}\n"
    for update in soul_updates:
        if update not in content:  # Deduplicate
            new_block += f"- {update}\n"

    if new_block.strip() == f"## [LEARNED_PATTERNS] — Auto-updated {timestamp}":
        return False  # Nothing new to add

    with soul_path.open("a", encoding="utf-8") as f:
        f.write(new_block)

    # Also save patterns snapshot to data dir
    _ensure_dir()
    snapshot = {
        "date": timestamp,
        "correction_rate": patterns_result.get("correction_rate"),
        "patterns": patterns_result.get("patterns"),
        "preferences": patterns_result.get("preferences"),
        "soul_updates": soul_updates,
        "gaps": patterns_result.get("gaps"),
    }
    snapshots_path = DATA_DIR / "patterns_history.jsonl"
    with snapshots_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(snapshot, ensure_ascii=False) + "\n")

    return True


def get_stats(days: int = 7) -> dict:
    """Summary stats for /status endpoint."""
    if not PERF_LOG.exists():
        return {"total": 0, "days": days}

    interactions = get_recent_interactions(hours_back=days * 24)
    corrections = [i for i in interactions if i.get("was_corrected")]

    # Most frequent tags
    from collections import Counter
    all_tags = []
    for i in interactions:
        all_tags.extend(i.get("tags", []))
    top_tags = Counter(all_tags).most_common(5)

    return {
        "total_interactions": len(interactions),
        "corrections": len(corrections),
        "correction_rate": f"{len(corrections)/max(len(interactions),1):.1%}",
        "top_topics": top_tags,
        "days_covered": days,
    }
