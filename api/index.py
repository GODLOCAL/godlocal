from fastapi import FastAPI
import os, time

app = FastAPI()

@app.get("/health")
async def health():
    return {"status": "ok", "ts": int(time.time())}

@app.get("/status")
async def status():
    return {"kill_switch_active": False, "circuit_breaker": {"is_tripped": False, "consecutive_losses": 0, "daily_loss_sol": 0.0}, "sparks": [], "thoughts": [], "ts": int(time.time())}

@app.get("/mobile/status")
async def mobile_status():
    return {"kill_switch_active": False, "circuit_breaker": {"is_tripped": False, "consecutive_losses": 0, "daily_loss_sol": 0.0}, "sparks": [], "thoughts": [], "ts": int(time.time())}

handler = app
