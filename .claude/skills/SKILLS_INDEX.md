# GodLocal Agent Skills Index
# Source v2: VoltAgent/awesome-agent-skills + sickn33/antigravity-awesome-skills (930+)
# Install path: .claude/skills/<slug>/
#
# Usage in Claude Code / any agent:
#   @<slug>  — loads the skill for current session
#
# Skills are fetched from upstream at install time.
# Run: bash .claude/skills/install.sh to install all

# ─────────────────────────────────────────────────────────
# BLOCK A: VoltAgent/awesome-agent-skills (curated originals)
# ─────────────────────────────────────────────────────────

## Security
- trailofbits/building-secure-contracts   # smart contract audit: 6 chains
- trailofbits/insecure-defaults           # detect hardcoded secrets, weak crypto
- trailofbits/static-analysis             # CodeQL + Semgrep + SARIF
- trailofbits/modern-python               # uv, ruff, ty, pytest best practices
- openai/security-best-practices          # language-specific vuln review
- openai/security-threat-model            # repo threat model with trust boundaries

## Cloud / Edge
- cloudflare/agents-sdk                   # stateful agents: scheduling, RPC, MCP
- cloudflare/durable-objects              # SQLite + WebSockets on Cloudflare Workers
- cloudflare/building-mcp-server-on-cloudflare  # remote MCP + OAuth

## ML / Models
- huggingface/hugging-face-model-trainer  # SFT, DPO, GRPO, GGUF conversion
- huggingface/hugging-face-evaluation     # vLLM + lighteval eval tables
- huggingface/hugging-face-jobs           # compute jobs on HF infra

## Mobile / Swift
- AvdLee/swiftui-expert-skill             # SwiftUI best practices + iOS 26 Liquid Glass
- efremidze/swift-patterns-skill          # Modern Swift/SwiftUI patterns
- conorluddy/ios-simulator-skill          # Control iOS Simulator

## DevOps / Infra
- hashicorp/terraform-code-generation     # generate + validate Terraform HCL
- microsoft/pydantic-models-py            # Pydantic models for API schemas
- microsoft/fastapi-router-py             # FastAPI routers with CRUD + auth

## Vercel / Next.js
- vercel-labs/react-best-practices        # React patterns
- vercel-labs/next-best-practices         # Next.js recommended patterns
- vercel-labs/vercel-deploy-claimable     # deploy to Vercel from agent

## Productivity / Code Quality
- getsentry/code-review                   # structured code review
- getsentry/find-bugs                     # bug detection
- getsentry/create-pr                     # PR creation best practices
- openai/yeet                             # stage + commit + push + open PR
- openai/gh-fix-ci                        # debug failing GitHub Actions

## AI Models / Generative
- replicate/replicate                     # 100K+ models: image, LLM, audio, video

# ─────────────────────────────────────────────────────────
# BLOCK B: antigravity-awesome-skills (930+ skills — GodLocal curated subset)
# Source: github.com/sickn33/antigravity-awesome-skills
# Install: npx antigravity-awesome-skills --claude
# ─────────────────────────────────────────────────────────

## AI Agents / LLM
- antigravity/ai-agents-architect                      # Design + build autonomous multi-tool AI agents
- antigravity/ai-ml                                    # LLM app dev, RAG pipelines, model evaluation
- antigravity/llm-app-patterns                         # Prod patterns: RAG, function calling, streaming, eval
- antigravity/ai-agent-development                     # Multi-agent systems, orchestration, HITL
- antigravity/agent-memory-mcp                         # Hybrid memory: vector + KV + episodic via MCP
- antigravity/agent-memory-systems                     # Memory architecture for persistent intelligent agents
- antigravity/agent-evaluation                         # Behavioral testing + benchmarking for LLM agents
- antigravity/prompt-engineer                          # Structured prompt engineering, chain-of-thought

## Telegram / Bot
- antigravity/telegram-bot-builder                     # Prod Telegram bots: commands, menus, webhooks, payments
- antigravity/discord-bot-architect                    # Production Discord bots (patterns shared with Telegram)

## Web3 / Solana
- antigravity/web3-smart-contracts                     # Smart contract patterns, auditing, deployment
- antigravity/solana-programs                          # Solana program development with Anchor framework

## Full-Stack / API
- antigravity/senior-fullstack                         # Full-stack web dev: React + Node + DB + auth
- antigravity/react-nextjs-development                 # React + Next.js 14+ App Router, Server Components
- antigravity/senior-architect                         # Scalable system design, ADRs, C4 diagrams
- antigravity/python-performance-optimization          # Profile + optimize Python: cProfile, memory
- antigravity/brainstorming                            # Structured ideation before architecture/features

## Security / Audit
- antigravity/security-audit                           # Full-stack sec audit: OWASP Top10, API, infra
- antigravity/api-security-best-practices              # Auth, authz, rate limiting, injection prevention
- antigravity/web-security-testing                     # OWASP Top10 vuln testing workflow
- antigravity/find-bugs                                # Bug + security vuln detection in local codebase
- antigravity/codebase-cleanup-deps-audit              # Dependency vuln scanning + CVE remediation
- antigravity/sast-configuration                       # Configure Semgrep/CodeQL/Bandit for CI SAST

## Testing / QA
- antigravity/tdd-workflows-tdd-red                    # TDD red-phase: generate failing tests first
- antigravity/code-reviewer                            # Elite AI-powered code review with actionable feedback

## DevOps / CI-CD
- antigravity/git-pr-workflows-git-workflow            # Git: branch strategy, PR, merge, release
- antigravity/antigravity-workflows                    # Multi-skill orchestration: SaaS MVP, sec audit, agent build

## UI/UX
- antigravity/ui-ux-pro-max                            # 50 styles, 21 palettes, 50 fonts — production-grade UI
- antigravity/tailwind-patterns                        # Tailwind CSS v4, container queries, dark mode

## Voice AI
- antigravity/voice-ai-engine-development              # Real-time voice AI: async workers, STT/TTS pipeline
