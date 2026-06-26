from pathlib import Path

import requests


def authenticate(base_url: str, username: str, password: str) -> str:
    resp = requests.post(
        f"{base_url}/api/token-auth/",
        json={"username": username, "password": password},
    )
    resp.raise_for_status()
    return resp.json()["token"]


class EscriptoriumClient:
    def __init__(self, base_url: str, token: str):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Token {token}"})

    def _get_paginated(self, path: str, **params) -> list:
        all_results = []
        url = f"{self.base_url}{path}"
        while url:
            resp = self.session.get(url, params=params or None)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list):
                return data
            all_results.extend(data.get("results", []))
            url = data.get("next")
            params = {}
        return all_results

    def get_document(self, doc_id: int) -> dict:
        return self.session.get(f"{self.base_url}/api/documents/{doc_id}/").json()

    def get_transcriptions(self, doc_id: int) -> list:
        return self._get_paginated(f"/api/documents/{doc_id}/transcriptions/")

    def get_parts(self, doc_id: int) -> list:
        return self._get_paginated(f"/api/documents/{doc_id}/parts/")

    def get_lines(self, doc_id: int, part_id: int) -> list:
        return self._get_paginated(f"/api/documents/{doc_id}/parts/{part_id}/lines/")

    def get_line_transcriptions(self, doc_id: int, part_id: int, transcription_id: int) -> list:
        return self._get_paginated(
            f"/api/documents/{doc_id}/parts/{part_id}/transcriptions/",
            transcription=transcription_id,
        )

    def download_image(self, url: str, dest: Path):
        if url.startswith("/"):
            url = f"{self.base_url}{url}"
        resp = self.session.get(url, stream=True)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
