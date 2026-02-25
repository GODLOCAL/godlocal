"""
api/routes/status.py — add to existing /status/warrior endpoint:

@router.get("/status/hardware")
async def hardware_status():
    from core.hardware_probe import get_hardware_probe
    report = get_hardware_probe().scan()
    return {
        "hardware": str(report.hardware),
        "tier_ceiling": report.tier_ceiling.value,
        "runnable_models": [s.model.id for s in report.runnable],
        "installed_models": [s.model.id for s in report.installed_runnable],
        "recommendations": [
            {
                "tier": s.model.tier.value,
                "model": s.model.name,
                "overall_score": round(s.overall, 3),
                "install_hint": s.model.install_hint,
            }
            for s in report.recommendations()
        ],
    }
"""
