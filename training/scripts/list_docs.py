#!/usr/bin/env python3
"""
List documents in eScriptorium with their IDs.

Usage:
    cd training && uv run python scripts/list_docs.py \\
        --url http://localhost:8080 \\
        --username admin --password admin
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.client import authenticate, EscriptoriumClient


def main():
    parser = argparse.ArgumentParser(description="List eScriptorium documents")
    parser.add_argument("--url", required=True)
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", required=True)
    args = parser.parse_args()

    token = authenticate(args.url, args.username, args.password)
    client = EscriptoriumClient(args.url, token)

    docs = client._get_paginated("/api/documents/")
    if not docs:
        print("No documents found.")
        return

    print(f"{'ID':>5}  NAME")
    print(f"{'─'*5}  ─{'─'*50}")
    for d in docs:
        name = d.get("name") or "(unnamed)"
        print(f"{d['pk']:>5}  {name}")


if __name__ == "__main__":
    main()
