"""
ConsciousnessLoop -> ReasoningBank Pipe + Inter-Component Event Bus
====================================================================
Closes the loop: ConsciousnessLoop thoughts become ReasoningBank memories.
Also provides EventBus so GlintSignalBus/xzero can inject topics into
ConsciousnessLoop's queue without direct imports (avoids circular deps).

Quick start:

    # Push a signal into ConsciousnessLoop from GlintSignalBus:
    from core.consciousness_pipe import event_bus
    event_bus.push("whale moved $5M USDC fresh wallet", source="glint", urgency=0.9)

    # After ConsciousnessLoop generates thought, pipe it to ReasoningBank:
    from core.consciousness_pipe import pipe_thought_to_reasoning_bank
    await pipe_thought_to_reasoning_bank(thought_text, topic="whale_pattern")
"""
import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Awaitable

log = logging.getLogger(__name__)


@dataclass
class SignalEvent:
    content: str
    source: str = "unknown"
    urgency: float = 0.5
    timestamp: float = field(default_factory=time.time)


class EventBus:
    """
    Lightweight pub/sub for inter-component communication.
    Push from any component; ConsciousnessLoop subscribes.
    """

    def __init__(self, maxsize: int = 100):
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self._subscribers: List[Callable[[SignalEvent], Awaitable[None]]] = []

    def push(self, content: str, source: str = "unknown", urgency: float = 0.5) -> None:
        """Non-blocking push from any component. Drops silently if queue full."""
        event = SignalEvent(content=content, source=source, urgency=urgency)
        try:
            self._queue.put_nowait(event)
            log.debug(f"[EventBus] {source} -> queue ({urgency:.1f}): {content[:80]}")
        except asyncio.QueueFull:
            log.warning("[EventBus] Queue full — signal dropped.")

    async def pop(self, timeout: float = 5.0) -> Optional[SignalEvent]:
        """Pop next event with timeout. Returns None on timeout."""
        try:
            return await asyncio.wait_for(self._queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    def subscribe(self, handler: Callable[[SignalEvent], Awaitable[None]]) -> None:
        """Register async handler for every new event."""
        self._subscribers.append(handler)

    async def dispatch_loop(self) -> None:
        """Run as background asyncio task."""
        while True:
            event = await self.pop(timeout=30.0)
            if event is None:
                continue
            for handler in self._subscribers:
                try:
                    await handler(event)
                except Exception as e:
                    log.warning(f"[EventBus] handler error: {e}")


# Global event bus singleton — import this anywhere
event_bus = EventBus()


# In-memory fallback buffer when ReasoningBank is unavailable (Vercel slim deploy)
_thought_buffer: List[dict] = []


async def pipe_thought_to_reasoning_bank(
    thought: str,
    topic: str,
    confidence: float = 0.7,
    source: str = "consciousness",
) -> bool:
    """
    Push a ConsciousnessLoop thought into ReasoningBank as a memory entry.
    Falls back to in-memory buffer if ReasoningBank is unavailable.
    Returns True if stored in ReasoningBank, False if buffered.
    """
    try:
        from core.reasoning_bank import get_reasoning_bank  # type: ignore
        bank = get_reasoning_bank()
        entry = {
            "content": thought,
            "topic": topic,
            "source": source,
            "confidence": confidence,
            "timestamp": time.time(),
            "tags": ["consciousness", "autonomous"],
        }
        await bank.store(entry)
        log.info(f"[ConscPipe] Thought -> ReasoningBank [{topic}]: {thought[:60]}...")
        return True
    except ImportError:
        _thought_buffer.append({"thought": thought, "topic": topic, "ts": time.time()})
        if len(_thought_buffer) > 200:
            _thought_buffer.pop(0)
        log.debug("[ConscPipe] ReasoningBank unavailable — buffered in memory.")
        return False
    except Exception as e:
        log.warning(f"[ConscPipe] pipe_thought failed: {e}")
        return False


def get_thought_buffer() -> List[dict]:
    """Returns in-memory thought buffer (when ReasoningBank unavailable)."""
    return list(_thought_buffer)


def thought_buffer_stats() -> dict:
    """Quick stats for /status endpoint."""
    return {
        "buffered_thoughts": len(_thought_buffer),
        "oldest_ts": _thought_buffer[0]["ts"] if _thought_buffer else None,
        "newest_ts": _thought_buffer[-1]["ts"] if _thought_buffer else None,
    }
