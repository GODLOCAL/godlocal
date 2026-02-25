
# Add to TieredRouter.__init__ (after self.stats = TierStats()):

    def __init__(self) -> None:
        self._brain: Any = None
        self._fast_brain: Any = None
        self.wasm = WASMHandlers()
        self.stats = TierStats()
        # Auto-configure based on hardware (llmfit-inspired)
        try:
            from core.hardware_probe import probe_and_configure
            self._hw_report = probe_and_configure()
        except Exception as e:
            self._hw_report = None
            import logging
            logging.getLogger(__name__).debug("[TieredRouter] HardwareProbe unavailable: %s", e)
