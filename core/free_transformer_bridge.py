"""
core/free_transformer_bridge.py
FreeTransformerBridge — VAE-style latent conditioning for GodLocal LLMBridge.

Based on: "The Free Transformer" — François Fleuret, Meta, Oct 2025
arXiv:2510.17558 | HuggingFace Paper: 2510.17558

Core idea (from abstract):
  "...extends the decoder Transformer by conditioning its generative process on
   random latent variables learned without supervision via a variational procedure."

What breaks since 2017:
  Standard decoder-only (GPT-style): P(x_t | x_1..t-1) — pure autoregressive.
  Free Transformer: P(x_t | x_1..t-1, z) where z ~ q(z|x) learned via ELBO.
  The latent z is a "planning variable" injected mid-decoder, enabling the model
  to condition generation on a compressed representation of *what it intends to say*
  before it says it. Like having a thought before speaking.

GodLocal integration:
  - Wraps any LLMBridge backend (Ollama, SonicGateway, MLX, Nexa)
  - Adds VAE-style latent sampling at inference time via prompt scaffolding
  - "Planning pass": generate z (latent summary) → condition full generation on z
  - Compatible with SkillOrchestraRouter: skill-conditioned z sampling
  - 1.5B model beats larger models benchmark claim → use PARO 4B instead of LFM2

Usage:
    from core.free_transformer_bridge import FreeTransformerBridge
    from core.brain import LLMBridge

    base = LLMBridge(model="paro:4b")
    ft = FreeTransformerBridge(base, latent_temp=0.7, n_samples=3)

    # Single best-of-n sample
    response = await ft.generate(prompt, skill_hint="analyze_onchain")

    # Multiple diverse samples (creativity mode)
    samples = await ft.sample_diverse(prompt, n=3)
    best = ft.select_best(samples, criterion="coherence")
"""
from __future__ import annotations

import asyncio
import json
import random
from dataclasses import dataclass
from typing import Optional

# ── Latent plan dataclass ─────────────────────────────────────────────────────

@dataclass
class LatentPlan:
    """
    z — compressed representation of generation intent.
    Equivalent to the VAE latent variable, but expressed as natural language
    (since we're prompting an existing LLM, not training a custom model).
    """
    summary: str          # 1-sentence intent summary
    key_points: list[str] # 2-4 key points the generation should cover
    tone: str             # e.g. "analytical", "concise", "creative"
    skill_hint: str = ""  # from SkillOrchestraRouter


# ── FreeTransformerBridge ────────────────────────────────────────────────────

class FreeTransformerBridge:
    """
    Wraps any GodLocal LLMBridge and adds Free Transformer-style
    latent conditioning at inference time.

    Two-pass generation:
      Pass 1 (Planning pass):  LLM generates latent plan z from prompt.
                               This is the "conditional VAE encoder" analog.
      Pass 2 (Generation pass): LLM generates final response conditioned on
                               both the original prompt AND latent plan z.
                               This is the "free decoder" analog.

    Inspiration: "The Free Transformer" (Fleuret, Meta, arXiv:2510.17558)
    Practical note: We approximate the VAE-style procedure via prompt engineering
    rather than weight-level conditioning, making it backend-agnostic.
    """

    # System prompt for the planning pass (generates latent z)
    PLANNER_SYSTEM = """You are a latent planner. Given a user request, produce a
compact generation plan as JSON — NOT the final answer.
Output ONLY valid JSON, no prose.
Schema:
{
  "summary": "one sentence: what the response will accomplish",
  "key_points": ["point 1", "point 2", "point 3"],
  "tone": "analytical|concise|creative|technical|instructive"
}"""

    # System prompt for the generation pass (conditioned on z)
    GENERATOR_SYSTEM_TEMPLATE = """You are a precise AI assistant.
Before generating, internalize this execution plan (your latent guidance):

PLAN:
  Summary   : {summary}
  Key Points: {key_points}
  Tone      : {tone}
{skill_section}
Now generate the final response. Follow the plan precisely."""

    def __init__(
        self,
        llm_bridge,                   # any LLMBridge-compatible object with .generate()
        latent_temp: float = 0.6,     # temperature for planning pass (lower = more focused)
        gen_temp: float = 0.8,        # temperature for generation pass
        n_samples: int = 1,           # number of diverse samples (best-of-n)
        max_plan_tokens: int = 256,   # matches SYSTEM_PROMPT [PLAN] block budget
        fallback_on_error: bool = True,
    ):
        self.llm = llm_bridge
        self.latent_temp = latent_temp
        self.gen_temp = gen_temp
        self.n_samples = n_samples
        self.max_plan_tokens = max_plan_tokens
        self.fallback_on_error = fallback_on_error

    # ── Core: planning pass ──────────────────────────────────────────────────

    async def _plan(self, prompt: str, skill_hint: str = "") -> LatentPlan:
        """
        Pass 1: Generate latent plan z.
        Runs at lower temperature for stable, focused planning.
        """
        plan_prompt = f"USER REQUEST:\n{prompt}"
        if skill_hint:
            plan_prompt += f"\nSKILL CONTEXT: {skill_hint}"

        try:
            raw = await self.llm.generate(
                prompt=plan_prompt,
                system=self.PLANNER_SYSTEM,
                temperature=self.latent_temp,
                max_tokens=self.max_plan_tokens,
            )
            # Extract JSON block robustly
            text = raw.strip()
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            data = json.loads(text)
            return LatentPlan(
                summary=data.get("summary", ""),
                key_points=data.get("key_points", []),
                tone=data.get("tone", "concise"),
                skill_hint=skill_hint,
            )
        except Exception as e:
            # Fallback: minimal latent plan from prompt keywords
            return LatentPlan(
                summary=f"Respond helpfully to: {prompt[:100]}",
                key_points=["Be accurate", "Be concise"],
                tone="concise",
                skill_hint=skill_hint,
            )

    # ── Core: generation pass ────────────────────────────────────────────────

    async def _generate_with_plan(
        self, prompt: str, plan: LatentPlan, temperature: Optional[float] = None
    ) -> str:
        """
        Pass 2: Generate conditioned on latent plan z.
        This is the "free decoder" — generation is no longer purely autoregressive;
        it's conditioned on the compressed latent representation.
        """
        skill_section = (
            f"  Skill     : {plan.skill_hint}\n" if plan.skill_hint else ""
        )
        system = self.GENERATOR_SYSTEM_TEMPLATE.format(
            summary=plan.summary,
            key_points=", ".join(plan.key_points),
            tone=plan.tone,
            skill_section=skill_section,
        )
        return await self.llm.generate(
            prompt=prompt,
            system=system,
            temperature=temperature or self.gen_temp,
        )

    # ── Public API ───────────────────────────────────────────────────────────

    async def generate(
        self,
        prompt: str,
        skill_hint: str = "",
        temperature: Optional[float] = None,
    ) -> str:
        """
        Full Free Transformer inference: plan → generate.
        Drop-in replacement for LLMBridge.generate().
        """
        try:
            plan = await self._plan(prompt, skill_hint=skill_hint)
            return await self._generate_with_plan(prompt, plan, temperature=temperature)
        except Exception as e:
            if self.fallback_on_error:
                # Graceful degradation: direct generation without latent conditioning
                return await self.llm.generate(prompt=prompt, temperature=temperature or self.gen_temp)
            raise

    async def sample_diverse(
        self,
        prompt: str,
        n: int = 3,
        skill_hint: str = "",
    ) -> list[tuple[LatentPlan, str]]:
        """
        Generate n diverse samples with different latent plans (different z draws).
        Approximates VAE sampling from posterior q(z|x).
        Higher n = more creativity (matches "2× more creativity" claim from paper).

        Returns list of (plan, response) tuples for downstream selection.
        """
        # Generate one plan, then vary temperature for diverse z draws
        temps = [self.latent_temp * (0.8 + 0.2 * i) for i in range(n)]

        async def _single(temp: float):
            plan = await self._plan(prompt, skill_hint=skill_hint)
            # Jitter generation temperature for diversity
            gen_temp = self.gen_temp + random.uniform(-0.1, 0.2)
            response = await self._generate_with_plan(prompt, plan, temperature=gen_temp)
            return (plan, response)

        results = await asyncio.gather(*[_single(t) for t in temps])
        return list(results)

    def select_best(
        self,
        samples: list[tuple[LatentPlan, str]],
        criterion: str = "coherence",
    ) -> str:
        """
        Simple heuristic selection from diverse samples.
        For production: replace with a reward model / LLM-as-judge.
        criterion options: "coherence" (longest coherent), "shortest", "random"
        """
        responses = [r for _, r in samples]
        if not responses:
            return ""
        if criterion == "shortest":
            return min(responses, key=len)
        if criterion == "random":
            return random.choice(responses)
        # "coherence" heuristic: longest response with reasonable length cap
        scored = [(r, min(len(r), 2000)) for r in responses]
        return max(scored, key=lambda x: x[1])[0]

    # ── Convenience: X-ZERO integration ─────────────────────────────────────

    async def analyze_onchain(self, data: dict) -> str:
        """Pre-wired skill: analyze on-chain data with latent planning."""
        prompt = f"Analyze this Solana on-chain data and give a concise summary:\n{json.dumps(data, indent=2)}"
        return await self.generate(prompt, skill_hint="analyze_onchain")

    async def generate_alert(self, event: dict) -> str:
        """Pre-wired skill: generate Telegram alert message from X-ZERO event."""
        prompt = (
            f"Write a concise Telegram alert (≤280 chars) for this X-ZERO event:\n"
            f"{json.dumps(event, indent=2)}\nInclude emoji. Be specific about price/% change."
        )
        return await self.generate(prompt, skill_hint="alert_telegram")

    async def trade_rationale(self, signal: dict) -> str:
        """Pre-wired skill: explain trading decision before execution."""
        prompt = f"Explain in 2 sentences why this trade signal warrants action:\n{json.dumps(signal, indent=2)}"
        return await self.generate(prompt, skill_hint="trade_swap")
