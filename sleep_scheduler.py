"""
sleep_scheduler.py — Standalone cron for sleep_cycle()
Fires sleep_cycle() every night at SLEEP_CYCLE_HOUR (default: 01:00).
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

while True:
    now = datetime.now()
    if now.hour == SLEEP_CYCLE_HOUR and now.minute == 0:
        run_sleep_cycle()
        time.sleep(61)  # avoid double-fire
    time.sleep(30)
