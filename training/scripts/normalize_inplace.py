import argparse
import sys
import unicodedata

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
        description="NFC-normalize transcription content in-place via API"
    )
    parser.add_argument("--url", default=ESCRIPTIONIUM_URL)
    parser.add_argument("--username", default=USERNAME)
    parser.add_argument("--password", default=PASSWORD)
    parser.add_argument("--document-id", type=int, required=True)
    parser.add_argument("--transcription-name", default="manual")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would change without updating")
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

    total_lines = 0
    total_changed = 0
    updated_pages = 0
    dry_run = args.dry_run

    for idx, part in enumerate(parts):
        part_id = part["pk"]
        lines = get_lines(session, args.url, args.document_id, part_id)
        lts = get_line_transcriptions(
            session, args.url, args.document_id, part_id, transcription["pk"]
        )
        content_by_line = {lt["line"]: lt["content"] for lt in lts if lt.get("content")}
        lt_by_line = {lt["line"]: lt for lt in lts}

        updates = []
        for line in lines:
            raw = content_by_line.get(line["pk"])
            if not raw:
                continue
            clean = unicodedata.normalize("NFC", raw)
            if raw != clean:
                updates.append({
                    "pk": lt_by_line[line["pk"]]["pk"],
                    "content": clean,
                })

        total_lines += len(content_by_line)
        total_changed += len(updates)

        if updates:
            updated_pages += 1
            if not dry_run:
                resp = session.put(
                    f"{args.url}/api/documents/{args.document_id}/parts/{part_id}/transcriptions/bulk_update/",
                    json={"lines": updates},
                )
                if resp.status_code != 200:
                    print(f"  [ERROR] page {part_id} (order={part.get('order')}): HTTP {resp.status_code}", file=sys.stderr)
                    print(f"  Response: {resp.text[:500]}", file=sys.stderr)
                else:
                    result = resp.json()
                    print(f"  [{idx+1}/{len(parts)}] updated {len(updates)} lines on page {part.get('order')}", file=sys.stderr)
            else:
                print(f"  [{idx+1}/{len(parts)}] would update {len(updates)} lines on page {part.get('order')}", file=sys.stderr)

        if not updates and not dry_run:
            print(f"  [{idx+1}/{len(parts)}] no changes on page {part.get('order')}", file=sys.stderr)

    action = "Would update" if dry_run else "Updated"
    print(f"\n{'='*50}", file=sys.stderr)
    print(f"{action} {total_changed} out of {total_lines} lines across {updated_pages} pages", file=sys.stderr)
    if dry_run:
        print("(dry run — no changes written)", file=sys.stderr)


if __name__ == "__main__":
    main()
