#!/usr/bin/env python3
"""
Simple keepalive pinger to prevent Render services from idling.

Usage examples:
  python scripts/keepalive.py --url https://url-purifier.onrender.com --interval 600
  KEEPALIVE_URL=https://url-purifier.onrender.com python scripts/keepalive.py --once

This script intentionally does not modify your application; run it from a separate
process (local machine, scheduler, or a background worker).
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from typing import Optional

import requests


logger = logging.getLogger("keepalive")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def ping(url: str, timeout: int = 10) -> tuple[int, str]:
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "keepalive/1.0"})
        return resp.status_code, resp.reason
    except requests.exceptions.RequestException as exc:
        return 0, str(exc)


def run_loop(url: str, interval: int, once: bool = False) -> None:
    logger.info("Keepalive started: url=%s interval=%ds once=%s", url, interval, once)

    stop = False

    def _signal_handler(signum, frame):
        nonlocal stop
        logger.info("Received signal %s, stopping keepalive...", signum)
        stop = True

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    while not stop:
        status, info = ping(url)
        if status == 200:
            logger.info("Ping OK: %s (200) - %s", url, info)
        elif status == 0:
            logger.warning("Ping failed: %s - %s", url, info)
        else:
            logger.warning("Ping returned %s - %s", status, info)

        if once:
            break

        sleep_for = interval
        # Sleep in smaller chunks so signals are responsive
        while sleep_for > 0 and not stop:
            to_sleep = min(5, sleep_for)
            time.sleep(to_sleep)
            sleep_for -= to_sleep


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Keepalive pinger for Render-hosted app")
    p.add_argument("--url", "-u", help="Full URL of the app to ping (overrides KEEPALIVE_URL env)")
    p.add_argument(
        "--interval",
        "-i",
        type=int,
        default=int(os.getenv("KEEPALIVE_INTERVAL", "600")),
        help="Interval between pings in seconds (default: 600)",
    )
    p.add_argument("--once", action="store_true", help="Ping once and exit (use for testing)")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    url = args.url or os.getenv("KEEPALIVE_URL")
    if not url:
        logger.error("No URL configured. Pass --url or set KEEPALIVE_URL environment variable.")
        return 2

    try:
        run_loop(url, args.interval, once=args.once)
    except Exception as exc:
        logger.exception("Unexpected error in keepalive: %s", exc)
        return 1

    logger.info("Keepalive finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
