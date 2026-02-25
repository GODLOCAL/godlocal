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
- trailofbits/building-secure-contracts    # smart contract audit: 6 chains
- trailofbits/insecure-defaults             # detect hardcoded secrets, weak crypto
- trailofbits/static-analysis               # CodeQL + Semgrep + SARIF
- trailofbits/modern-python                 # uv, ruff, ty, pytest best practices
- openai/security-best-practices            # language-specific vuln review
- openai/security-threat-model              # repo threat model with trust boundaries

## Cloud / Edge
- cloudflare/agents-sdk                     # stateful agents: scheduling, RPC, MCP
- cloudflare/durable-objects                # SQLite + WebSockets on Cloudflare Workers
- cloudflare/building-mcp-server-on-cloudflare  # remote MCP + OAuth

## ML / Models
- huggingface/hugging-face-model-trainer   # SFT, DPO, GRPO, GGUF conversion
- huggingface/hugging-face-evaluation       # vLLM + lighteval eval tables
- huggingface/hugging-face-jobs             # compute jobs on HF infra

## Mobile / Swift
- AvdLee/swiftui-expert-skill               # SwiftUI best practices + iOS 26 Liquid Glass
- efremidze/swift-patterns-skill            # Modern Swift/SwiftUI patterns
- conorluddy/ios-simulator-skill            # Control iOS Simulator

## DevOps / Infra
- hashicorp/terraform-code-generation       # generate + validate Terraform HCL
- microsoft/pydantic-models-py              # Pydantic models for API schemas
- microsoft/fastapi-router-py               # FastAPI m ters with CRUD + auth

## Vercel / Next.js
- vercel-labs/react-best-practices          # React patterns
- vercel-labs/next-best-practices           # Next.js recommended patterns
- vercel-labs/vercel-deploy-claimable      # deploy to Vercel from agent

## Productivity / Code Quality
- getsentry/code-review                     # structured code review
- getsentry/find-bugs                       # bug detection
- getsentry/create-pr                       # PR creation best practices
- openai/yeet                                # stage + commit + push + open PR
- openai/gh-fix-ci                           # debug failing GitHub Actions

## AI Models / Generative
- replicate/replicate                       # 100K+ models: image, LLM, audio, video


## Content / Writing
- blader/humanizer                                        # 24 AI patterns: significance inflation, em dash, vocab, hedging → natural text
