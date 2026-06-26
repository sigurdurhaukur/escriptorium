import responses

from training.client import EscriptoriumClient, authenticate


class TestAuthenticate:
    @responses.activate
    def test_returns_token_on_success(self):
        responses.post(
            "http://escr.io/api/token-auth/",
            json={"token": "abc123"},
            status=200,
        )

        token = authenticate("http://escr.io", "alice", "pass")

        assert token == "abc123"

    @responses.activate
    def test_raises_on_bad_credentials(self):
        responses.post(
            "http://escr.io/api/token-auth/",
            json={"non_field_errors": ["Invalid credentials"]},
            status=400,
        )

        import pytest
        with pytest.raises(Exception):
            authenticate("http://escr.io", "alice", "wrong")


class TestEscriptoriumClient:
    @responses.activate
    def test_get_document_detail(self):
        responses.get(
            "http://escr.io/api/documents/1/",
            json={"pk": 1, "name": "My Doc"},
        )

        client = EscriptoriumClient("http://escr.io", "tok")
        doc = client.get_document(1)

        assert doc["pk"] == 1
        assert doc["name"] == "My Doc"

    @responses.activate
    def test_get_transcriptions_paginated(self):
        responses.get(
            "http://escr.io/api/documents/1/transcriptions/",
            json={
                "count": 2,
                "next": None,
                "previous": None,
                "results": [
                    {"pk": 10, "name": "diplomatic", "archived": False},
                    {"pk": 11, "name": "normalized", "archived": True},
                ],
            },
        )

        client = EscriptoriumClient("http://escr.io", "tok")
        transcriptions = client.get_transcriptions(1)

        assert len(transcriptions) == 2
        assert transcriptions[0]["name"] == "diplomatic"

    @responses.activate
    def test_paginates_across_multiple_pages(self):
        responses.get(
            "http://escr.io/api/documents/1/parts/",
            json={
                "count": 3,
                "next": "http://escr.io/api/documents/1/parts/?page=2",
                "previous": None,
                "results": [{"pk": 1, "order": 0, "image": {"uri": "/img/1.jpg"}}],
            },
        )
        responses.get(
            "http://escr.io/api/documents/1/parts/?page=2",
            json={
                "count": 3,
                "next": "http://escr.io/api/documents/1/parts/?page=3",
                "previous": "http://escr.io/api/documents/1/parts/?page=1",
                "results": [{"pk": 2, "order": 1, "image": {"uri": "/img/2.jpg"}}],
            },
        )
        responses.get(
            "http://escr.io/api/documents/1/parts/?page=3",
            json={
                "count": 3,
                "next": None,
                "previous": "http://escr.io/api/documents/1/parts/?page=2",
                "results": [{"pk": 3, "order": 2, "image": {"uri": "/img/3.jpg"}}],
            },
        )

        client = EscriptoriumClient("http://escr.io", "tok")
        parts = client.get_parts(1)

        assert len(parts) == 3
        assert [p["pk"] for p in parts] == [1, 2, 3]

    @responses.activate
    def test_sends_auth_header(self):
        responses.get(
            "http://escr.io/api/documents/1/",
            json={"pk": 1},
        )

        client = EscriptoriumClient("http://escr.io", "my-token")
        client.get_document(1)

        assert responses.calls[0].request.headers["Authorization"] == "Token my-token"

    @responses.activate
    def test_download_image(self, tmp_path):
        responses.get(
            "http://escr.io/media/img.jpg",
            body=b"fake-image-bytes",
            content_type="image/jpeg",
        )

        client = EscriptoriumClient("http://escr.io", "tok")
        dest = tmp_path / "test.jpg"
        client.download_image("/media/img.jpg", dest)

        assert dest.read_bytes() == b"fake-image-bytes"

    @responses.activate
    def test_get_lines(self):
        doc_id, part_id = 1, 5
        responses.get(
            f"http://escr.io/api/documents/{doc_id}/parts/{part_id}/lines/",
            json={
                "count": 1,
                "next": None,
                "previous": None,
                "results": [
                    {
                        "pk": 100,
                        "baseline": [[0, 0], [10, 0]],
                        "mask": [[0, -5], [10, -5], [10, 5], [0, 5]],
                    }
                ],
            },
        )

        client = EscriptoriumClient("http://escr.io", "tok")
        lines = client.get_lines(doc_id, part_id)

        assert len(lines) == 1
        assert lines[0]["pk"] == 100

    @responses.activate
    def test_get_line_transcriptions(self):
        doc_id, part_id, tr_id = 1, 5, 20
        responses.get(
            f"http://escr.io/api/documents/{doc_id}/parts/{part_id}/transcriptions/",
            json={
                "count": 1,
                "next": None,
                "previous": None,
                "results": [
                    {"pk": 200, "line": 100, "content": "hello", "transcription": tr_id}
                ],
            },
        )

        client = EscriptoriumClient("http://escr.io", "tok")
        lts = client.get_line_transcriptions(doc_id, part_id, tr_id)

        assert len(lts) == 1
        assert lts[0]["content"] == "hello"
