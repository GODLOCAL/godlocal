"""
GodLocal Telegram Bot — v5 adapter
====================================
Bridges Telegram ↔ GodLocal v5 FastAPI server running locally.

Commands:
  /start    — greeting + quick help
  /status   — show GodLocal v5 capabilities + device info
  /sleep    — trigger sleep_cycle() manually
  /evolve   — trigger self_evolve() cycle
  /gaps     — show current knowledge gaps
  /soul     — show active soul file
  /souls    — list available soul files
  /image    — generate image (usage: /image a cyberpunk city)
  /audio    — generate speech (usage: /audio Hello world)
  /app      — generate mini app (usage: /app todo list)
  /clear    — clear conversation history

Plain text → /chat endpoint (streaming response)

Setup:
  pip install python-telegram-bot httpx
  TELEGRAM_BOT_TOKEN=xxx GODLOCAL_URL=http://localhost:8000 python godlocal_telegram.py
"""

import asyncio
import logging
import os
import httpx

from telegram import Update, BotCommand
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from telegram.constants import ParseMode, ChatAction

# ── Config ────────────────────────────────────────────────────────────────────

BOT_TOKEN    = os.getenv("TELEGRAM_BOT_TOKEN", "")
GODLOCAL_URL = os.getenv("GODLOCAL_URL", "http://localhost:8000").rstrip("/")
LOG_LEVEL    = os.getenv("LOG_LEVEL", "INFO")

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    level=getattr(logging, LOG_LEVEL),
)
logger = logging.getLogger("godlocal_tg")

if not BOT_TOKEN:
    raise RuntimeError(
        "TELEGRAM_BOT_TOKEN is not set.\n"
        "Get one from @BotFather and run:\n"
        "  export TELEGRAM_BOT_TOKEN=your_token_here"
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

async def gl_get(path: str, timeout: int = 30) -> dict:
    """GET request to GodLocal v5 API."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.get(f"{GODLOCAL_URL}{path}")
        r.raise_for_status()
        return r.json()


async def gl_post(path: str, body: dict | None = None, timeout: int = 120) -> dict:
    """POST request to GodLocal v5 API."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(f"{GODLOCAL_URL}{path}", json=body or {})
        r.raise_for_status()
        return r.json()


def fmt_status(data: dict) -> str:
    """Format /status response into readable text."""
    caps = data.get("capabilities", {})
    soul = data.get("soul", "default")
    device = data.get("device", "unknown")
    msgs  = data.get("session_messages", 0)

    on  = "✅"
    off = "❌"

    def c(key): return on if caps.get(key) else off

    return (
        f"*GodLocal v5 — Status*

"
        f"🧠 Soul: `{soul}`
"
        f"💻 Device: `{device}`
"
        f"💬 Session messages: {msgs}

"
        f"*Capabilities:*
"
        f"{c('llm_ollama')} Ollama LLM
"
        f"{c('llm_airllm')} AirLLM (4-bit)
"
        f"{c('memory_chroma')} ChromaDB memory
"
        f"{c('sleep_cycle')} sleep\_cycle()
"
        f"{c('self_evolve')} self\_evolve()
"
        f"{c('image_generation')} Image generation
"
        f"{c('video_generation')} Video generation
"
        f"{c('tts_bark')} Bark TTS
"
        f"{c('music_musicgen')} MusicGen
"
        f"{c('app_generation')} AppGen
"
        f"{c('medical_mri')} MRI Analyzer
"
        f"{c('safe_executor')} SafeExecutor
"
    )


# ── Command handlers ──────────────────────────────────────────────────────────

async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🧠 *GodLocal v5* — your sovereign AI, running on your machine.

"
        "Commands:
"
        "/status — capabilities
"
        "/sleep — run sleep\_cycle()
"
        "/evolve — run self\_evolve()
"
        "/gaps — knowledge gaps
"
        "/soul — active soul
"
        "/souls — list souls
"
        "/image \<prompt\> — generate image
"
        "/audio \<text\> — generate speech
"
        "/app \<description\> — generate mini-app
"
        "/clear — reset conversation

"
        "Or just type a message to chat.",
        parse_mode=ParseMode.MARKDOWN_V2,
    )


async def cmd_status(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.chat.send_action(ChatAction.TYPING)
    try:
        data = await gl_get("/status")
        await update.message.reply_text(fmt_status(data), parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        await update.message.reply_text(f"❌ GodLocal unreachable: {e}")


async def cmd_sleep(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🌙 Running sleep\_cycle\(\)… this takes a moment.", parse_mode=ParseMode.MARKDOWN_V2)
    await update.message.chat.send_action(ChatAction.TYPING)
    try:
        data = await gl_post("/sleep", timeout=120)
        promoted = data.get("promoted_count", 0)
        pruned   = data.get("pruned_count", 0)
        dur      = data.get("duration_s", "?")
        insights = data.get("insights", "")
        evolve   = data.get("self_evolve", {})

        text = (
            f"✅ *sleep\_cycle() done* in {dur}s

"
            f"📈 Promoted: {promoted} memories
"
            f"🗑 Pruned: {pruned} memories

"
            f"*Insights:*
{insights[:800]}"
        )
        if evolve and not evolve.get("error"):
            text += (
                f"

🧬 *self\_evolve():* "
                f"{evolve.get('gaps_resolved', 0)}/{evolve.get('gaps_found', 0)} gaps resolved"
            )
        await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        await update.message.reply_text(f"❌ sleep\_cycle failed: {e}", parse_mode=ParseMode.MARKDOWN)


async def cmd_evolve(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🧬 Launching self\_evolve\(\)…", parse_mode=ParseMode.MARKDOWN_V2)
    await update.message.chat.send_action(ChatAction.TYPING)
    try:
        data = await gl_post("/evolve", timeout=5)
        await update.message.reply_text(
            f"✅ {data.get('message', 'Evolution started')}

Check /gaps in a minute.",
        )
    except Exception as e:
        await update.message.reply_text(f"❌ evolve failed: {e}")


async def cmd_gaps(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.chat.send_action(ChatAction.TYPING)
    try:
        data = await gl_get("/evolve/gaps")
        gaps = data.get("gaps", [])
        total = data.get("total_gaps", 0)
        if not gaps:
            await update.message.reply_text("🎉 No knowledge gaps found — AI at full confidence.")
            return
        lines = [f"*Knowledge gaps ({total} total):*
"]
        for i, g in enumerate(gaps[:10], 1):
            freq = g.get("frequency", 1)
            topic = g.get("topic", "unknown")
            lines.append(f"{i}. `{topic}` — {freq}x")
        await update.message.reply_text("
".join(lines), parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        await update.message.reply_text(f"❌ {e}")


async def cmd_soul(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.chat.send_action(ChatAction.TYPING)
    try:
        data = await gl_get("/status")
        soul = data.get("soul", "default")
        await update.message.reply_text(f"🧿 Active soul: *{soul}*", parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        await update.message.reply_text(f"❌ {e}")


async def cmd_souls(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.chat.send_action(ChatAction.TYPING)
    try:
        data = await gl_get("/souls")
        souls   = data.get("souls", [])
        current = data.get("current", "")
        lines = ["*Available souls:*
"]
        for s in souls:
            marker = " ← active" if s == current else ""
            lines.append(f"• `{s}`{marker}")
        await update.message.reply_text("
".join(lines), parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        await update.message.reply_text(f"❌ {e}")


async def cmd_image(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    prompt = " ".join(ctx.args) if ctx.args else ""
    if not prompt:
        await update.message.reply_text("Usage: /image <prompt>  e.g. /image cyberpunk city at night")
        return
    await update.message.reply_text(f"🎨 Generating: _{prompt}_…", parse_mode=ParseMode.MARKDOWN)
    await update.message.chat.send_action(ChatAction.UPLOAD_PHOTO)
    try:
        data = await gl_post("/generate/image", {"prompt": prompt}, timeout=180)
        url  = data.get("url") or data.get("image_url")
        path = data.get("path") or data.get("file_path")
        if url:
            await update.message.reply_photo(url, caption=f"_{prompt}_", parse_mode=ParseMode.MARKDOWN)
        elif path:
            with open(path, "rb") as f:
                await update.message.reply_photo(f, caption=f"_{prompt}_", parse_mode=ParseMode.MARKDOWN)
        else:
            await update.message.reply_text(f"✅ Image generated: {data}")
    except Exception as e:
        await update.message.reply_text(f"❌ Image gen failed: {e}")


async def cmd_audio(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    text = " ".join(ctx.args) if ctx.args else ""
    if not text:
        await update.message.reply_text("Usage: /audio <text>  e.g. /audio Hello from GodLocal")
        return
    await update.message.reply_text(f"🔊 Generating speech…")
    await update.message.chat.send_action(ChatAction.RECORD_VOICE)
    try:
        data = await gl_post("/generate/audio", {"text": text}, timeout=120)
        path = data.get("path") or data.get("audio_path")
        if path:
            with open(path, "rb") as f:
                await update.message.reply_voice(f)
        else:
            await update.message.reply_text(f"✅ Audio: {data}")
    except Exception as e:
        await update.message.reply_text(f"❌ Audio gen failed: {e}")


async def cmd_app(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    desc = " ".join(ctx.args) if ctx.args else ""
    if not desc:
        await update.message.reply_text("Usage: /app <description>  e.g. /app todo list with dark theme")
        return
    await update.message.reply_text(f"⚙️ Generating app: _{desc}_…", parse_mode=ParseMode.MARKDOWN)
    await update.message.chat.send_action(ChatAction.TYPING)
    try:
        data = await gl_post("/generate/app", {"description": desc}, timeout=300)
        code = data.get("code", "")
        html = data.get("html", "")
        output = html or code
        if output:
            # Send as file if long
            if len(output) > 3000:
                fname = "generated_app.html" if html else "generated_app.py"
                await update.message.reply_document(
                    document=output.encode(),
                    filename=fname,
                    caption=f"✅ App for: _{desc}_",
                    parse_mode=ParseMode.MARKDOWN,
                )
            else:
                lang = "html" if html else "python"
                await update.message.reply_text(
                    f"```{lang}
{output[:4000]}
```",
                    parse_mode=ParseMode.MARKDOWN,
                )
        else:
            await update.message.reply_text(f"✅ {data}")
    except Exception as e:
        await update.message.reply_text(f"❌ App gen failed: {e}")


async def cmd_clear(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    try:
        await gl_post("/clear")
        await update.message.reply_text("🧹 Conversation history cleared.")
    except Exception as e:
        await update.message.reply_text(f"❌ {e}")


# ── Chat handler ──────────────────────────────────────────────────────────────

async def handle_message(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Forward plain messages to GodLocal /chat endpoint."""
    user_msg = update.message.text
    if not user_msg:
        return

    await update.message.chat.send_action(ChatAction.TYPING)
    try:
        data = await gl_post(
            "/chat",
            {"message": user_msg, "history": ctx.user_data.get("history", [])},
            timeout=120,
        )
        reply = data.get("reply") or data.get("response") or str(data)

        # Update local history buffer (last 20 turns)
        history = ctx.user_data.get("history", [])
        history.append({"role": "user",      "content": user_msg})
        history.append({"role": "assistant",  "content": reply})
        ctx.user_data["history"] = history[-20:]

        # Telegram max 4096 chars per message
        if len(reply) > 4000:
            for i in range(0, len(reply), 4000):
                await update.message.reply_text(reply[i:i+4000])
        else:
            await update.message.reply_text(reply)

    except httpx.ConnectError:
        await update.message.reply_text(
            "❌ Can't reach GodLocal. Is the server running?
"
            f"Expected: `{GODLOCAL_URL}`
"
            "Start with: `python godlocal_v5.py`",
            parse_mode=ParseMode.MARKDOWN,
        )
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────

async def post_init(app):
    """Register bot commands visible in Telegram menu."""
    await app.bot.set_my_commands([
        BotCommand("start",  "Welcome + help"),
        BotCommand("status", "GodLocal capabilities"),
        BotCommand("sleep",  "Run sleep_cycle()"),
        BotCommand("evolve", "Run self_evolve()"),
        BotCommand("gaps",   "Knowledge gaps"),
        BotCommand("soul",   "Active soul"),
        BotCommand("souls",  "List all souls"),
        BotCommand("image",  "Generate image"),
        BotCommand("audio",  "Generate speech"),
        BotCommand("app",    "Generate mini-app"),
        BotCommand("clear",  "Clear conversation"),
    ])
    logger.info("✅ Bot commands registered")


def main():
    logger.info(f"Starting GodLocal Telegram Bot → {GODLOCAL_URL}")

    app = (
        Application.builder()
        .token(BOT_TOKEN)
        .post_init(post_init)
        .build()
    )

    app.add_handler(CommandHandler("start",  cmd_start))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("sleep",  cmd_sleep))
    app.add_handler(CommandHandler("evolve", cmd_evolve))
    app.add_handler(CommandHandler("gaps",   cmd_gaps))
    app.add_handler(CommandHandler("soul",   cmd_soul))
    app.add_handler(CommandHandler("souls",  cmd_souls))
    app.add_handler(CommandHandler("image",  cmd_image))
    app.add_handler(CommandHandler("audio",  cmd_audio))
    app.add_handler(CommandHandler("app",    cmd_app))
    app.add_handler(CommandHandler("clear",  cmd_clear))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("🚀 Bot polling started. Press Ctrl+C to stop.")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
