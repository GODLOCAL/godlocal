"""
sleep_scheduler.py — Standalone cron for sleep_cycle() + AutoGenesis
Fires sleep_cycle() every night at SLEEP_CYCLE_HOUR (default: 01:00).
Phase 4 (AutoGenesis) runs automatically inside sleep_cycle().

Optionally start AutoGenesis Shortcuts server alongside:
  AUTOGENESIS_SERVE=1 python sleep_scheduler.py
"""

import os
import time
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [sleep_cycle] %(message)s",
    handlers=[
        logging.StreamHandler(),
    ]
)

SLEEP_CYCLE_HOUR = int(os.getenv("SLEEP_CYCLE_HOUR", 1))
RUN_TIME = f"{SLEEP_CYCLE_HOUR:02d}:00"




AUTOGENESIS_SERVE = os.getenv("AUTOGENESIS_SERVE", "0") == "1"
AUTOGENESIS_PORT  = int(os.getenv("AUTOGENESIS_PORT", "7100"))


def _start_autogenesis_server():
    """Start AutoGenesis iPhone Shortcuts server in background thread."""
    try:
        from oasis_autogenesis import AutoGenesis, make_server
        import uvicorn
        genesis = AutoGenesis(root=".")
        app = make_server(genesis)
        logging.info(f"🚀 AutoGenesis server → http://localhost:{AUTOGENESIS_PORT}")
        uvicorn.run(app, host="0.0.0.0", port=AUTOGENESIS_PORT, log_level="warning")
    except Exception as e:
        logging.error(f"AutoGenesis server failed: {e}")

def run_sleep_cycle():
    logging.info(f"🌙 sleep_cycle() starting — {datetime.now().isoformat()}")
    try:
        from godlocal_v5 import GodLocalAgent
        agent = GodLocalAgent()
        result = agent.run_sleep_cycle()
        logging.info(f"✅ sleep_cycle() complete: {result}")
    except Exception as e:
        logging.error(f"❌ sleep_cycle() failed: {e}")


logging.info(f"🕐 sleep_cycle scheduler active — fires daily at {RUN_TIME}")

if AUTOGENESIS_SERVE:
    import threading
    _t = threading.Thread(target=_start_autogenesis_server, daemon=True)
    _t.start()
    logging.info(f"  + AutoGenesis Shortcuts server on :{AUTOGENESIS_PORT}")


while True:
    now = datetime.now()
    if now.hour == SLEEP_CYCLE_HOUR and now.minute == 0:
        run_sleep_cycle()
        time.sleep(61)  # avoid double-fire
    time.sleep(30)
