import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import requests

ESCRIPTIONIUM_URL = "http://localhost:8080"
USERNAME = "admin"
PASSWORD = "admin"


def authenticate(base_url, username, password):
    resp = requests.post(
        f"{base_url}/api/token-auth/",
        json={"username": username, "password": password},
    )
    resp.raise_for_status()
    return resp.json()["token"]


def get_paginated(session, url, params=None):
    all_results = []
    while url:
        resp = session.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list):
            return data
        all_results.extend(data.get("results", []))
        url = data.get("next")
        params = {}
    return all_results


def get_transcriptions(session, base_url, doc_id):
    return get_paginated(
        session, f"{base_url}/api/documents/{doc_id}/transcriptions/"
    )


def get_parts(session, base_url, doc_id):
    return get_paginated(session, f"{base_url}/api/documents/{doc_id}/parts/")


def get_lines(session, base_url, doc_id, part_id):
    return get_paginated(
        session, f"{base_url}/api/documents/{doc_id}/parts/{part_id}/lines/"
    )


def get_line_transcriptions(session, base_url, doc_id, part_id, transcription_id):
    return get_paginated(
        session,
        f"{base_url}/api/documents/{doc_id}/parts/{part_id}/transcriptions/",
        params={"transcription": transcription_id},
    )


def main():
    parser = argparse.ArgumentParser(
        description="Count character frequencies in an eScriptorium transcription"
    )
    parser.add_argument("--url", default=ESCRIPTIONIUM_URL, help="eScriptorium URL")
    parser.add_argument("--username", default=USERNAME)
    parser.add_argument("--password", default=PASSWORD)
    parser.add_argument("--document-id", type=int, required=True)
    parser.add_argument("--transcription-name", default="manual")
    parser.add_argument("--output", default="char_frequencies.json")
    args = parser.parse_args()

    print("Authenticating...", file=sys.stderr)
    token = authenticate(args.url, args.username, args.password)
    session = requests.Session()
    session.headers.update({"Authorization": f"Token {token}"})

    print(f"Fetching document {args.document_id}...", file=sys.stderr)
    doc = session.get(f"{args.url}/api/documents/{args.document_id}/").json()
    print(f"Document: {doc.get('name')} (pk={doc.get('pk')})", file=sys.stderr)

    print("Fetching transcriptions...", file=sys.stderr)
    transcriptions = get_transcriptions(session, args.url, args.document_id)
    transcription = next(
        (t for t in transcriptions if t["name"] == args.transcription_name), None
    )
    if not transcription:
        available = [t["name"] for t in transcriptions]
        print(f"Transcription '{args.transcription_name}' not found. Available: {available}", file=sys.stderr)
        sys.exit(1)
    print(f"Using transcription: {transcription['name']} (pk={transcription['pk']})", file=sys.stderr)

    print("Fetching parts (pages)...", file=sys.stderr)
    parts = get_parts(session, args.url, args.document_id)
    print(f"Found {len(parts)} parts", file=sys.stderr)

    counter = Counter()
    total_lines = 0
    total_chars = 0
    page_freqs = {}

    for idx, part in enumerate(parts):
        part_id = part["pk"]
        lines = get_lines(session, args.url, args.document_id, part_id)
        lts = get_line_transcriptions(
            session, args.url, args.document_id, part_id, transcription["pk"]
        )
        content_by_line = {lt["line"]: lt["content"] for lt in lts if lt.get("content")}

        page_counter = Counter()
        for line in lines:
            content = content_by_line.get(line["pk"])
            if not content:
                continue
            counter.update(content)
            page_counter.update(content)
            total_lines += 1
            total_chars += len(content)

        page_freqs[str(part_id)] = {
            "order": part.get("order"),
            "total_chars": sum(page_counter.values()),
            "frequencies": dict(page_counter.most_common()),
        }

        if (idx + 1) % 10 == 0 or (idx + 1) == len(parts):
            print(f"  [{idx+1}/{len(parts)}] processed", file=sys.stderr)

    output = {
        "document_id": args.document_id,
        "document_name": doc.get("name"),
        "transcription_name": transcription["name"],
        "total_parts": len(parts),
        "total_lines": total_lines,
        "total_chars": total_chars,
        "unique_chars": len(counter),
        "global_frequencies": dict(counter.most_common()),
        "page_frequencies": page_freqs,
    }

    print(f"\nWriting to {args.output}...", file=sys.stderr)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nSummary:", file=sys.stderr)
    print(f"  Pages: {len(parts)}", file=sys.stderr)
    print(f"  Lines: {total_lines}", file=sys.stderr)
    print(f"  Total characters: {total_chars}", file=sys.stderr)
    print(f"  Unique characters: {len(counter)}", file=sys.stderr)
    print("\nTop 30 most frequent characters:", file=sys.stderr)
    for char, count in counter.most_common(30):
        display = repr(char)[1:-1] if char in "\n\r\t" else char
        print(f"  {display!r:>6}  ({char!r:>4})  {count:>6}  {count/total_chars*100:>5.2f}%", file=sys.stderr)


if __name__ == "__main__":
    main()
