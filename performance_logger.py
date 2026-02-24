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


def _rotate_log_if_needed(max_mb: float = 5.0):
    """Rotate performance_log.jsonl when it exceeds max_mb. Keeps last 3 archives."""
    if not PERF_LOG.exists():
        return
    size_mb = PERF_LOG.stat().st_size / (1024 * 1024)
    if size_mb < max_mb:
        return
    archive_dir = DATA_DIR / "log_archives"
    archive_dir.mkdir(exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    archive_path = archive_dir / f"performance_log_{ts}.jsonl"
    import shutil
    shutil.copy2(PERF_LOG, archive_path)
    PERF_LOG.write_text("", encoding="utf-8")  # Truncate
    # Keep only last 3 archives
    archives = sorted(archive_dir.glob("performance_log_*.jsonl"))
    for old in archives[:-3]:
        old.unlink()
    import logging as _log
    _log.getLogger(__name__).info(f"[PerfLog] Rotated log → {archive_path.name} ({size_mb:.1f}MB)")


def log_interaction(
    user_input: str,
    agent_response: str,
    was_corrected: bool = False,
    correction_note: Optional[str] = None,
    session_id: Optional[str] = None,
    tags: Optional[list] = None,
    latency_ms: Optional[float] = None,
    tokens_used: Optional[int] = None,
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
        latency_ms:      Response generation time in milliseconds
        tokens_used:     Approximate token count for this response
    """
    _ensure_dir()
    _rotate_log_if_needed()  # Auto-rotate at 5MB
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
        "latency_ms": latency_ms,
        "tokens_used": tokens_used if tokens_used is not None else len(agent_response.split()),
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
    # Extended metrics: latency + token tracking
    latencies = [i["latency_ms"] for i in interactions if i.get("latency_ms") is not None]
    tokens    = [i["tokens_used"] for i in interactions if i.get("tokens_used") is not None]
    avg_latency_ms = round(sum(latencies) / len(latencies), 1) if latencies else None
    avg_tokens     = round(sum(tokens) / len(tokens), 1) if tokens else None
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


_LOCKED_MARKER = "[LOCKED]"
_ROLLBACK_THRESHOLD = 0.10  # Rollback if correction_rate worsens by >10pp


def _extract_locked_sections(soul_text: str) -> list[tuple[int, int]]:
    """Return (start, end) line ranges that are marked [LOCKED] — LLM must not modify."""
    locked_ranges = []
    lines = soul_text.splitlines(keepends=True)
    i = 0
    while i < len(lines):
        if _LOCKED_MARKER in lines[i]:
            start = i
            # Lock extends until next section header (##) or EOF
            j = i + 1
            while j < len(lines) and not (lines[j].startswith("##") or lines[j].startswith("# ")):
                j += 1
            locked_ranges.append((start, j))
            i = j
        else:
            i += 1
    return locked_ranges


def _strip_locked_content(soul_text: str) -> str:
    """Remove [LOCKED] markers from soul text for display/analysis (does not modify file)."""
    return "\n".join(
        l for l in soul_text.splitlines() if _LOCKED_MARKER not in l
    )


def update_soul_with_patterns(patterns_result: dict, soul_path: Path = SOUL_PATH):
    """
    Append learned patterns to god_soul.md under [LEARNED_PATTERNS] section.
    - Respects [LOCKED] sections — never overwrites them
    - Creates a pre-update backup (god_soul.md.bak) for rollback
    - Rolls back if correction_rate worsens by >10pp vs previous snapshot
    Creates the section if it doesn't exist. Idempotent — checks for duplicates.
    """
    import logging as _log
    import shutil as _shutil
    logger = _log.getLogger(__name__)

    if patterns_result.get("status") != "ok":
        return False
    soul_updates = patterns_result.get("soul_updates", [])
    if not soul_updates:
        return False

    _ensure_dir()
    content = soul_path.read_text(encoding="utf-8") if soul_path.exists() else ""

    # ── Check rollback condition ─────────────────────────────────────────────
    snapshots_path = DATA_DIR / "patterns_history.jsonl"
    if snapshots_path.exists():
        try:
            snap_lines = [l for l in snapshots_path.read_text().splitlines() if l.strip()]
            if snap_lines:
                prev = json.loads(snap_lines[-1])
                prev_rate = float(prev.get("correction_rate") or 0)
                curr_rate = float(patterns_result.get("correction_rate") or 0)
                if prev_rate > 0 and (curr_rate - prev_rate) > _ROLLBACK_THRESHOLD:
                    # Correction rate INCREASED → degradation → rollback
                    bak = soul_path.with_suffix(".md.bak")
                    if bak.exists():
                        _shutil.copy2(bak, soul_path)
                        logger.warning(
                            f"[SoulRollback] correction_rate degraded "
                            f"{prev_rate:.1%} → {curr_rate:.1%} (>{_ROLLBACK_THRESHOLD:.0%}pp). "
                            f"Rolled back god_soul.md from .bak"
                        )
                        return False
        except Exception as _re:
            logger.warning(f"[SoulRollback] check failed: {_re}")

    # ── Snapshot before update (for rollback next cycle) ────────────────────
    if soul_path.exists():
        _shutil.copy2(soul_path, soul_path.with_suffix(".md.bak"))

    # ── Build new block — skip anything that touches [LOCKED] lines ─────────
    timestamp = datetime.utcnow().strftime("%Y-%m-%d")
    new_block = f"\n\n## [LEARNED_PATTERNS] — Auto-updated {timestamp}\n"
    locked_ranges = _extract_locked_sections(content)

    for update in soul_updates:
        if update not in content:  # Deduplicate
            new_block += f"- {update}\n"

    if new_block.strip() == f"## [LEARNED_PATTERNS] — Auto-updated {timestamp}":
        return False  # Nothing new to add

    # Verify we're not about to overwrite a [LOCKED] section
    # (appending is safe; only flag if soul_path has [LOCKED] in [LEARNED_PATTERNS] area)
    if locked_ranges:
        soul_lines = content.splitlines()
        learned_start = next(
            (i for i, l in enumerate(soul_lines) if "[LEARNED_PATTERNS]" in l), None
        )
        for (ls, le) in locked_ranges:
            if learned_start is not None and ls >= learned_start:
                logger.warning("[SoulUpdate] Skipping update — [LOCKED] marker inside [LEARNED_PATTERNS]")
                return False

    with soul_path.open("a", encoding="utf-8") as f:
        f.write(new_block)

    # ── Save patterns snapshot ───────────────────────────────────────────────
    snapshot = {
        "date": timestamp,
        "correction_rate": patterns_result.get("correction_rate"),
        "patterns": patterns_result.get("patterns"),
        "preferences": patterns_result.get("preferences"),
        "soul_updates": soul_updates,
        "gaps": patterns_result.get("gaps"),
        "rollback_bak": str(soul_path.with_suffix(".md.bak")),
    }
    with snapshots_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(snapshot, ensure_ascii=False) + "\n")

    logger.info(f"[SoulUpdate] {len(soul_updates)} patterns written. Backup: god_soul.md.bak")
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
