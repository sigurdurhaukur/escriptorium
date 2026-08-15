import argparse
import json
import os
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import requests


ESCRIPTIONIUM_URL = "http://localhost:8080"
USERNAME = "admin"
PASSWORD = "admin"

ICELANDIC_ALPHA = set(
    "aábdðeéfghiíjklmnoóprstuúvxyýþæö"
    "AÁBDÐEÉFGHIÍJKLMNOÓPRSTUÚVXYÝÞÆÖ"
)

COMMON_PUNCTUATION = set(
    " .,:;!?()[]{}'\"„“«»-–—/\\@&%#*+=°•|~\n\r\t"
)

DIGITS = set("0123456789")

ALLOWED = ICELANDIC_ALPHA | COMMON_PUNCTUATION | DIGITS


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


def nfc_normalize(text):
    return unicodedata.normalize("NFC", text)


def classify_char(ch):
    if ch in ICELANDIC_ALPHA:
        return "icelandic"
    if ch in DIGITS:
        return "digit"
    if ch in COMMON_PUNCTUATION:
        return "punctuation"
    return "suspicious"


def main():
    parser = argparse.ArgumentParser(
        description="Export, clean, and save a benchmark dataset from eScriptorium"
    )
    parser.add_argument("--url", default=ESCRIPTIONIUM_URL)
    parser.add_argument("--username", default=USERNAME)
    parser.add_argument("--password", default=PASSWORD)
    parser.add_argument("--document-id", type=int, required=True)
    parser.add_argument("--transcription-name", default="manual")
    parser.add_argument("--output", default="clean_dataset.json")
    args = parser.parse_args()

    print("Authenticating...", file=sys.stderr)
    token = authenticate(args.url, args.username, args.password)
    session = requests.Session()
    session.headers.update({"Authorization": f"Token {token}"})

    doc = session.get(f"{args.url}/api/documents/{args.document_id}/").json()
    print(f"Document: {doc.get('name')} (pk={doc.get('pk')})", file=sys.stderr)

    transcriptions = get_transcriptions(session, args.url, args.document_id)
    transcription = next(
        (t for t in transcriptions if t["name"] == args.transcription_name), None
    )
    if not transcription:
        available = [t["name"] for t in transcriptions]
        print(f"Transcription '{args.transcription_name}' not found. Available: {available}", file=sys.stderr)
        sys.exit(1)
    print(f"Transcription: {transcription['name']} (pk={transcription['pk']})", file=sys.stderr)

    parts = get_parts(session, args.url, args.document_id)
    print(f"Pages: {len(parts)}", file=sys.stderr)

    # Stats
    total_lines = 0
    total_chars_before = 0
    total_chars_after = 0
    lines_with_diffs = 0
    changed_lines = []

    before_freq = Counter()
    after_freq = Counter()
    suspicious_before = Counter()
    suspicious_after = Counter()

    dataset = {
        "document_name": doc.get("name"),
        "document_id": args.document_id,
        "transcription_name": transcription["name"],
        "transcription_id": transcription["pk"],
        "total_parts": len(parts),
        "pages": [],
    }

    for idx, part in enumerate(parts):
        part_id = part["pk"]
        lines = get_lines(session, args.url, args.document_id, part_id)
        lts = get_line_transcriptions(
            session, args.url, args.document_id, part_id, transcription["pk"]
        )
        content_by_line = {lt["line"]: lt["content"] for lt in lts if lt.get("content")}

        page_data = {
            "part_id": part_id,
            "order": part.get("order"),
            "lines": [],
        }

        for line in lines:
            raw = content_by_line.get(line["pk"])
            if not raw:
                continue

            clean = nfc_normalize(raw)

            before_freq.update(raw)
            after_freq.update(clean)
            total_chars_before += len(raw)
            total_chars_after += len(clean)
            total_lines += 1

            for ch in raw:
                if classify_char(ch) == "suspicious":
                    suspicious_before[ch] += 1
            for ch in clean:
                if classify_char(ch) == "suspicious":
                    suspicious_after[ch] += 1

            if raw != clean:
                lines_with_diffs += 1
                changed_lines.append({
                    "part_id": part_id,
                    "line_pk": line["pk"],
                    "before": raw,
                    "after": clean,
                })

            page_data["lines"].append({
                "line_pk": line["pk"],
                "baseline": line.get("baseline"),
                "mask": line.get("mask"),
                "text": clean,
            })

        dataset["pages"].append(page_data)

        if (idx + 1) % 10 == 0 or (idx + 1) == len(parts):
            print(f"  [{idx+1}/{len(parts)}] processed", file=sys.stderr)

    print(f"\n{'='*60}", file=sys.stderr)
    print("CLEANING REPORT", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    print(f"Lines processed: {total_lines}", file=sys.stderr)
    print(f"Chars before: {total_chars_before}", file=sys.stderr)
    print(f"Chars after:  {total_chars_after}", file=sys.stderr)
    print(f"Lines changed (NFC diff): {lines_with_diffs}", file=sys.stderr)

    if changed_lines:
        print(f"\nSample changes (first 10):", file=sys.stderr)
        for c in changed_lines[:10]:
            print(f"  BEFORE: {c['before']!r}", file=sys.stderr)
            print(f"  AFTER:  {c['after']!r}", file=sys.stderr)

    def print_freq(label, freq, total, limit=30):
        print(f"\n{label} ({len(freq)} unique):", file=sys.stderr)
        for ch, count in freq.most_common(limit):
            cat = classify_char(ch)
            tag = f"[{cat}]" if cat == "suspicious" else ""
            display = ch if ch.isprintable() else repr(ch)[1:-1]
            print(f"  {display!r:>6}  U+{ord(ch):04X}  {count:>6}  {count/total*100:>5.2f}%  {tag}", file=sys.stderr)

    print_freq("BEFORE NFC (top 30)", before_freq, total_chars_before)
    print_freq("AFTER NFC (top 30)", after_freq, total_chars_after)

    print(f"\nSUSPICIOUS CHARACTERS BEFORE:", file=sys.stderr)
    if suspicious_before:
        for ch, count in suspicious_before.most_common():
            print(f"  U+{ord(ch):04X} {ch!r}  ×{count}", file=sys.stderr)
    else:
        print("  (none)", file=sys.stderr)

    print(f"\nSUSPICIOUS CHARACTERS AFTER:", file=sys.stderr)
    if suspicious_after:
        for ch, count in suspicious_after.most_common():
            print(f"  U+{ord(ch):04X} {ch!r}  ×{count}", file=sys.stderr)
    else:
        print("  (none)", file=sys.stderr)

    dataset["summary"] = {
        "total_lines": total_lines,
        "total_chars_before": total_chars_before,
        "total_chars_after": total_chars_after,
        "lines_normalized": lines_with_diffs,
        "suspicious_chars_after": dict(suspicious_after),
        "unique_chars_after": len(after_freq),
        "after_frequencies": dict(after_freq.most_common()),
    }

    print(f"\nWriting dataset to {args.output}...", file=sys.stderr)
    with open(args.output, "w") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
