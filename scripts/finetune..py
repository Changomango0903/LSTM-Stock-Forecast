#!/usr/bin/env python
"""
Daily incremental fine-tuning after market close.
"""
import logging
from finetune.finetuner import run_finetune
from config import CONFIG

logging.basicConfig(level=logging.INFO)
if __name__ == "__main__":
    run_finetune(CONFIG["default_symbol"])
