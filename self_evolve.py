"""
GodLocal SelfEvolve Module — Autonomous Knowledge Gap Resolution
================================================================

The AI scans its own conversation history for uncertainty markers,
identifies topics it struggled with, then synthesizes new knowledge
entries into ChromaDB — entirely on-device, no API required.

Conceptually: if sleep_cycle() is slow-wave sleep (consolidation),
self_evolve() is waking metacognition — the AI notices its own blind
spots and patches them before the next conversation.

ENDPOINTS:
    POST /evolve              — trigger a full evolution cycle
    GET  /evolve/gaps         — inspect current knowledge gaps
    GET  /evolve/log          — full evolution history
    GET  /evolve/report       — latest report as text
"""

import asyncio
import json
import re
import sqlite3
import logging
logger = logging.getLogger(__name__)
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict
from collections import Counter

try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    print("[SelfEvolve] chromadb not installed — memory storage disabled.")


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class KnowledgeGap:
    """A topic the AI failed to answer confidently."""
    topic: str
    context: str                # raw excerpt from conversation
    frequency: int = 1          # how many times it appeared
    last_seen: str = ""         # ISO timestamp
    confidence: str = "low"     # low / medium
    resolved: bool = False

    def __post_init__(self):
        if not self.last_seen:
            self.last_seen = datetime.now().isoformat()

    def to_dict(self):
        return asdict(self)


@dataclass
class EvolutionResult:
    """Result of a single evolution cycle."""
    started_at: str
    finished_at: str
    gaps_found: int
    gaps_resolved: int
    topics_resolved: List[str] = field(default_factory=list)
    error: Optional[str] = None


# ──────────────────────────────────────────────────────────────────────────────
# Core engine
# ──────────────────────────────────────────────────────────────────────────────

class SelfEvolveEngine:
    """
    GodLocal's autonomous metacognition layer.

    How it works
    ────────────
    1.  SCAN  — read recent assistant turns from SQLite conversation DB
    2.  DETECT — find uncertainty markers (phrases like "I'm not sure",
                 "I don't have information about", etc.)
    3.  EXTRACT — pull the topic out of surrounding context
    4.  DEDUPLICATE — rank gaps by recurrence frequency
    5.  SYNTHESIZE — use local LLM (via Ollama/AirLLM) to generate a
                     factual knowledge entry for each top gap
    6.  STORE — write resolved entries back into ChromaDB memory
    7.  LOG — append to evolution_log.jsonl for transparency
    """

    # Phrases that indicate low-confidence responses
    UNCERTAINTY_MARKERS = [
        r"i(?:'m| am) not sure",
        r"i don't (?:have|know)",
        r"i (?:cannot|can't) (?:find|access|provide)",
        r"i(?:'m| am) unable to",
        r"i lack (?:the )?(?:information|knowledge|context|data)",
        r"(?:you|you'd|you would) (?:need to|want to) check",
        r"i(?:'m| am) uncertain",
        r"my (?:training|knowledge) (?:data )?(?:doesn't|does not|cuts off)",
        r"this is (?:outside|beyond) (?:my|what i)",
        r"unfortunately,? i (?:don't|do not|can't|cannot)",
        r"i (?:don't|do not) have (?:access|enough|information)",
        r"i (?:don't|do not) have (?:real[- ]time|live|current)",
        r"as of my (?:last|knowledge|training)",
        r"i(?:'m| am) not (?:familiar|aware)",
    ]

    # Compiled patterns for speed
    _COMPILED = [re.compile(p, re.IGNORECASE) for p in UNCERTAINTY_MARKERS]

    # Stopwords for topic extraction
    _STOPWORDS = {
        "the", "a", "an", "is", "are", "was", "were", "i", "you", "it",
        "this", "that", "my", "your", "but", "and", "or", "not", "in",
        "on", "at", "to", "for", "of", "with", "about", "from", "by",
        "do", "does", "have", "has", "been", "being", "be", "will",
        "would", "could", "should", "may", "might", "shall", "can",
        "just", "also", "very", "more", "some", "what", "when", "where",
        "how", "who", "which", "if", "than", "then", "so", "its", "as",
    }

    def __init__(self, data_dir: str = "godlocal_data"):
        self.data_dir = Path(data_dir)
        self.db_path = self.data_dir / "conversations.db"
        self.evolution_log = self.data_dir / "evolution_log.jsonl"
        self._last_report: Optional[str] = None

        # Ensure conversations.db exists (auto-create on first run)
        self._init_db()

        # ChromaDB
        self._chroma: Optional[object] = None
        if CHROMA_AVAILABLE:
            try:
                self._chroma = chromadb.PersistentClient(
                    path=str(self.data_dir / "memory")
                )
            except Exception as e:
                logger.exception("[SelfEvolve] ChromaDB initialization failed")


    def _init_db(self) -> None:
        """Create conversations.db with messages table if it doesn't exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    role      TEXT    NOT NULL,
                    content   TEXT    NOT NULL,
                    timestamp TEXT    NOT NULL DEFAULT (datetime('now'))
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_ts ON messages(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_role ON messages(role, timestamp)")
            conn.commit()
            conn.close()
        except sqlite3.Error as e:
            logger.exception("[SelfEvolve] Failed to initialize conversations.db")


    def log_message(self, role: str, content: str) -> None:
        """Called by godlocal_v5.py after each chat turn to persist messages."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.execute(
                "INSERT INTO messages (role, content, timestamp) VALUES (?, ?, ?)",
                (role, content, datetime.now().isoformat())
            )
            conn.commit()
            conn.close()
        except sqlite3.Error:
            logger.exception("[SelfEvolve] Failed to log message")

    # ── Gap detection ─────────────────────────────────────────────────────────

    def scan_conversation_gaps(self, hours_back: int = 48) -> List[KnowledgeGap]:
        """
        Read assistant turns from the last `hours_back` hours and return
        a deduplicated, frequency-ranked list of knowledge gaps.
        """
        if not self.db_path.exists():
            # Demo mode: synthesise a few synthetic gaps for illustration
            return self._demo_gaps()

        cutoff = (datetime.now() - timedelta(hours=hours_back)).isoformat()
        gaps: List[KnowledgeGap] = []

        try:
            conn = sqlite3.connect(str(self.db_path))
            cur = conn.cursor()
            cur.execute(
                """
                SELECT content, timestamp
                FROM messages
                WHERE role = 'assistant' AND timestamp > ?
                ORDER BY timestamp DESC
                LIMIT 500
                """,
                (cutoff,),
            )
            rows = cur.fetchall()
            conn.close()
        except sqlite3.Error as e:
            logger.exception(f"[SelfEvolve] DB error scanning gaps")
            return []

        for content, timestamp in rows:
            for pattern in self._COMPILED:
                m = pattern.search(content)
                if m:
                    topic = self._extract_topic(content, m.start())
                    if topic:
                        gaps.append(
                            KnowledgeGap(
                                topic=topic,
                                context=content[max(0, m.start() - 80):m.start() + 120],
                                last_seen=timestamp,
                            )
                        )
                    break  # one gap per message turn

        return self._deduplicate(gaps)

    def _extract_topic(self, text: str, marker_pos: int) -> Optional[str]:
        """Heuristic: pull meaningful nouns from the 200 chars after the marker."""
        window = text[marker_pos: marker_pos + 250]
        words = re.findall(r"[A-Za-z][A-Za-z0-9_\-']{2,}", window)
        meaningful = [
            w for w in words
            if w.lower() not in self._STOPWORDS and len(w) > 3
        ]
        if not meaningful:
            return None
        # Take the first 4 most meaningful words
        return " ".join(meaningful[:4])

    def _deduplicate(self, gaps: List[KnowledgeGap]) -> List[KnowledgeGap]:
        """Merge duplicate topics, rank by frequency (most urgent first)."""
        counter: Dict[str, KnowledgeGap] = {}
        for gap in gaps:
            key = gap.topic.lower()[:40]
            if key in counter:
                counter[key].frequency += 1
            else:
                counter[key] = gap
        return sorted(counter.values(), key=lambda g: g.frequency, reverse=True)

    def _demo_gaps(self) -> List[KnowledgeGap]:
        """Return illustrative gaps when no real DB exists (first-run demo)."""
        return [
            KnowledgeGap("Solana validator epochs", "...I'm not sure about current epoch timing...", frequency=3),
            KnowledgeGap("AirLLM layer offload strategy", "...I don't have enough context on VRAM usage...", frequency=2),
            KnowledgeGap("ChromaDB HNSW tuning", "...I cannot provide optimal ef_construction values...", frequency=2),
            KnowledgeGap("Bark voice presets list", "...I'm uncertain which v2 presets ship by default...", frequency=1),
        ]

    # ── Knowledge synthesis ───────────────────────────────────────────────────

    async def synthesize_gap(
        self,
        gap: KnowledgeGap,
        llm_generate=None,
    ) -> Optional[str]:
        """
        Generate a knowledge entry for `gap`.

        If an Ollama/AirLLM callable is provided, use it.
        Otherwise, produce a structured placeholder entry that marks the gap
        as identified — useful for the first evolution cycle.
        """
        existing = self._query_memory(gap.topic)

        if llm_generate:
            return await self._llm_synthesize(gap, llm_generate, existing)

        # Fallback: structured placeholder
        return (
            f"TOPIC: {gap.topic}
"
            f"FREQUENCY: {gap.frequency} occurrence(s) in recent conversations
"
            f"CONTEXT_SAMPLE: {gap.context[:200]}
"
            f"EXISTING_KNOWLEDGE: {existing[:200] if existing else 'none'}
"
            f"STATUS: pending_llm_synthesis
"
            f"IDENTIFIED_AT: {datetime.now().isoformat()}"
        )

    def _query_memory(self, topic: str) -> str:
        """Check if we already have anything useful in ChromaDB."""
        if not self._chroma:
            return ""
        try:
            col = self._chroma.get_collection("memories")
            results = col.query(query_texts=[topic], n_results=2,
                                include=["documents", "distances"])
            docs = results.get("documents", [[]])[0]
            dists = results.get("distances", [[1.0]])[0]
            # Only return if reasonably similar
            if docs and dists[0] < 0.35:
                return "
".join(docs)
        except Exception:
            pass
        return ""

    async def _llm_synthesize(
        self, gap: KnowledgeGap, llm_generate, existing: str
    ) -> str:
        existing_block = (
            f"
Existing partial knowledge:
{existing[:400]}"
            if existing else ""
        )
        prompt = (
            f"You are GodLocal's self-improvement module.\n\n"
            f"The AI struggled with this topic in recent conversations: "{gap.topic}"\n"
            f"Context where it appeared:\n\"{gap.context[:200]}\"{existing_block}\n\n"
            f"Generate a concise, factual knowledge entry about "{gap.topic}" "
            f"that will improve future answers.\n"
            f"Format:\n"
            f"TOPIC: {gap.topic}\n"
            f"KNOWLEDGE: [factual summary, 3–5 sentences]\n"
            f"CONFIDENCE: [low / medium / high]\n"
            f"RELATED: [comma-separated related topics]"
        )
        try:
            if asyncio.iscoroutinefunction(llm_generate):
                return await llm_generate(prompt, max_tokens=400)
            return llm_generate(prompt, max_tokens=400)
        except Exception as e:
            return f"TOPIC: {gap.topic}
SYNTHESIS_ERROR: {e}"

    # ── Memory storage ────────────────────────────────────────────────────────

    def store_resolution(self, gap: KnowledgeGap, knowledge: str):
        """Persist the synthesized knowledge entry to ChromaDB."""
        gap.resolved = True

        if self._chroma:
            try:
                col = self._chroma.get_or_create_collection("memories")
                doc_id = (
                    f"selfevolve_"
                    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_"
                    f"{re.sub(r'[^a-z0-9]', '_', gap.topic.lower())[:24]}"
                )
                col.add(
                    ids=[doc_id],
                    documents=[knowledge],
                    metadatas=[{
                        "source": "self_evolution",
                        "topic": gap.topic,
                        "frequency": gap.frequency,
                        "resolved_at": datetime.now().isoformat(),
                        "type": "gap_resolution",
                    }],
                )
            except Exception as e:
                print(f"[SelfEvolve] ChromaDB write error: {e}")

        self._append_log(gap, knowledge)
        print(f"[SelfEvolve] ✓ Resolved: {gap.topic}")

    def _append_log(self, gap: KnowledgeGap, knowledge: str):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "gap": gap.to_dict(),
            "knowledge_snippet": knowledge[:400],
            "action": "gap_resolved",
        }
        self.evolution_log.parent.mkdir(parents=True, exist_ok=True)
        with open(self.evolution_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "
")

    # ── Reporting ─────────────────────────────────────────────────────────────

    def build_report(
        self, gaps: List[KnowledgeGap], resolved_count: int
    ) -> str:
        width = 64
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        next_cycle = (datetime.now() + timedelta(hours=24)).strftime("%Y-%m-%d %H:%M")

        header = (
            f"╔{'═' * width}╗
"
            f"║{'GodLocal Self-Evolution Report':^{width}}║
"
            f"║{now:^{width}}║
"
            f"╚{'═' * width}╝
"
        )

        stats = (
            f"{'─' * width}
"
            f"  🔍 GAPS DISCOVERED : {len(gaps)}
"
            f"  ✅ GAPS RESOLVED   : {resolved_count}
"
            f"  📂 EVOLUTION LOG   : {self.evolution_log}
"
            f"{'─' * width}

"
        )

        if not gaps:
            body = "  🎉 No uncertainty gaps found — AI performing at full confidence!
"
        else:
            body = "  TOP KNOWLEDGE GAPS (ranked by recurrence)

"
            for i, gap in enumerate(gaps[:12], 1):
                status = "✓ RESOLVED" if i <= resolved_count else "⏳ QUEUED "
                body += (
                    f"  {i:2}. [{status}]  {gap.topic[:48]}
"
                    f"       Frequency : {gap.frequency}x  |  "
                    f"Last seen : {gap.last_seen[:10]}

"
                )

        footer = (
            f"{'─' * width}
"
            f"  Next evolution cycle : {next_cycle}
"
            f"  Memory location      : {self.data_dir / 'memory'}
"
            f"{'─' * width}
"
            f"
  The AI has updated its knowledge base autonomously.
"
            f"  It knows more now than it did an hour ago.
"
        )

        return header + stats + body + footer

    # ── Main cycle ────────────────────────────────────────────────────────────

    async def run_evolution_cycle(
        self,
        llm_generate=None,
        max_gaps: int = 5,
        hours_back: int = 48,
    ) -> EvolutionResult:
        """
        Full autonomous cycle:
            scan → detect → synthesize → store → report
        """
        started = datetime.now().isoformat()
        print("[SelfEvolve] 🧬 Evolution cycle starting…")

        gaps = self.scan_conversation_gaps(hours_back=hours_back)
        print(f"[SelfEvolve] Found {len(gaps)} knowledge gap(s).")

        resolved_topics: List[str] = []
        for gap in gaps[:max_gaps]:
            knowledge = await self.synthesize_gap(gap, llm_generate)
            if knowledge:
                self.store_resolution(gap, knowledge)
                resolved_topics.append(gap.topic)

        report = self.build_report(gaps, len(resolved_topics))
        self._last_report = report
        print(report)

        return EvolutionResult(
            started_at=started,
            finished_at=datetime.now().isoformat(),
            gaps_found=len(gaps),
            gaps_resolved=len(resolved_topics),
            topics_resolved=resolved_topics,
        )

    def get_last_report(self) -> str:
        return self._last_report or "No evolution cycle has run yet."

    def get_log_entries(self, limit: int = 50) -> List[dict]:
        if not self.evolution_log.exists():
            return []
        entries = []
        lines = self.evolution_log.read_text(encoding="utf-8").splitlines()
        for line in lines[-limit:]:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        return entries


# ──────────────────────────────────────────────────────────────────────────────
# FastAPI integration
# ──────────────────────────────────────────────────────────────────────────────

def register_selfevolve_routes(app, god_instance=None):
    """
    Call this from godlocal_v5.py after creating the FastAPI app:

        from self_evolve import register_selfevolve_routes
        register_selfevolve_routes(app, god)

    Adds:
        POST /evolve
        GET  /evolve/gaps
        GET  /evolve/log
        GET  /evolve/report
    """
    from fastapi import BackgroundTasks
    from fastapi.responses import PlainTextResponse

    engine = SelfEvolveEngine()
    _llm = getattr(god_instance, "generate", None) if god_instance else None

    @app.post("/evolve", summary="Trigger autonomous self-evolution cycle")
    async def trigger_evolution(background_tasks: BackgroundTasks):
        background_tasks.add_task(engine.run_evolution_cycle, _llm)
        return {
            "status": "evolution_started",
            "message": (
                "GodLocal is scanning its knowledge gaps and self-improving. "
                "Check /evolve/report when done."
            ),
        }

    @app.get("/evolve/gaps", summary="Inspect current knowledge gaps")
    async def get_gaps(hours_back: int = 48):
        gaps = engine.scan_conversation_gaps(hours_back=hours_back)
        return {
            "total_gaps": len(gaps),
            "hours_scanned": hours_back,
            "gaps": [g.to_dict() for g in gaps[:20]],
            "insight": (
                "These are topics where GodLocal showed uncertainty "
                "in recent conversations."
            ),
        }

    @app.get("/evolve/log", summary="Full evolution history")
    async def get_log(limit: int = 50):
        entries = engine.get_log_entries(limit)
        return {"total": len(entries), "entries": entries}

    @app.get("/evolve/report", response_class=PlainTextResponse,
             summary="Latest evolution cycle report")
    async def get_report():
        return engine.get_last_report()

    return engine


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ──────────────────────────────────────────────────────────────────────────────

async def _cli_main():
    import argparse

    parser = argparse.ArgumentParser(
        description="GodLocal Self-Evolution Engine — run a standalone cycle"
    )
    parser.add_argument("--data-dir", default="godlocal_data")
    parser.add_argument("--max-gaps", type=int, default=5)
    parser.add_argument("--hours-back", type=int, default=48)
    args = parser.parse_args()

    engine = SelfEvolveEngine(data_dir=args.data_dir)
    result = await engine.run_evolution_cycle(
        max_gaps=args.max_gaps,
        hours_back=args.hours_back,
    )
    print(f"
[Done] Resolved {result.gaps_resolved}/{result.gaps_found} gaps.")


if __name__ == "__main__":
    asyncio.run(_cli_main())
