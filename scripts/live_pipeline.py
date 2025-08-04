#!/usr/bin/env python
"""
Entry point for the daily live‐prediction pipeline.
Fetches data, denoises, computes indicators, makes a prediction,
and appends it to artifacts/live_results.csv.
"""

import os
import sys

# Ensure project root is on PYTHONPATH so imports work
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "..")))

import logging
from config import CONFIG
from evaluation.live import run_live_predict

# Configure logging: INFO for stages, DEBUG for shapes if needed
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

if __name__ == "__main__":
    # Kick off the live prediction process for the default symbol
    run_live_predict(CONFIG["default_symbol"])
