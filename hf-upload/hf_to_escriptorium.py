#!/usr/bin/env python3
import logging
import os
import shutil
import tempfile
from pathlib import Path

import requests
from datasets import load_dataset
from dotenv import load_dotenv
from lxml import html
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_api_session():
    load_dotenv()
    url = os.environ["ESCRIPTORIUM_URL"]
    username = os.environ["ESCRIPTORIUM_USERNAME"]
    password = os.environ["ESCRIPTORIUM_PASSWORD"]

    s = requests.Session()
    s.headers.update({"Accept": "application/json"})

    login_url = f"{url}/login/"
    r = s.get(login_url)
    tree = html.fromstring(r.text)
    csrf = tree.xpath("//input[@name='csrfmiddlewaretoken']/@value")[0]

    s.post(login_url, data={"username": username, "password": password, "csrfmiddlewaretoken": csrf},
           headers={**s.headers, "referer": login_url})

    r = s.get(f"{url}/profile/apikey/")
    tree = html.fromstring(r.text)
    api_key = tree.xpath("//button[@id='api-key-clipboard']/@data-key")[0]
    s.headers.update({"Authorization": f"Token {api_key}"})
    s.headers.pop("Accept", None)

    return s, url


def find_project_id(session, api_url):
    r = session.get(f"{api_url}projects/")
    data = r.json()
    projects = data.get("results", [])
    if not projects:
        logger.error("No projects found")
        return None
    project = projects[0]
    logger.info(f"Using project: {project['name']} (id={project['id']}, slug={project['slug']})")
    return project["id"], project["slug"]


def create_document(session, api_url, project_slug, doc_name):
    payload = {
        "name": doc_name,
        "project": project_slug,
        "main_script": "Latin",
        "read_direction": "ltr",
        "line_offset": 0,
        "tags": [],
    }
    r = session.post(f"{api_url}documents/", json=payload)
    doc = r.json()
    doc_id = doc.get("pk") or doc.get("id")
    logger.info(f"Created document: {doc_name} (id={doc_id})")
    return doc_id


def upload_image(session, api_url, document_id, image_path):
    filename = Path(image_path).name
    with open(image_path, "rb") as f:
        r = session.post(
            f"{api_url}documents/{document_id}/parts/",
            data={"filename": filename},
            files={"image": (filename, f)},
        )
    r.raise_for_status()
    return r.json()


def main(dataset_name, split="train", project_slug="admins-project",
         doc_name=None, image_column="image", max_samples=None):
    session, base_url = get_api_session()
    api_url = f"{base_url}/api/"

    project_id, slug = find_project_id(session, api_url)
    if not project_id:
        return

    dataset = load_dataset(dataset_name, split=split, streaming=True)
    logger.info("Dataset loaded (streaming)")

    temp_dir = tempfile.mkdtemp(prefix="escriptorium_")
    image_paths = []
    try:
        for idx, sample in enumerate(dataset):
            if max_samples is not None and idx >= max_samples:
                break
            if image_column not in sample:
                continue
            img = sample[image_column]
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img) if hasattr(img, "shape") else img
            img_path = Path(temp_dir) / f"image_{idx:06d}.png"
            img.save(img_path)
            image_paths.append(str(img_path))
            if (idx + 1) % 100 == 0:
                logger.info(f"Extracted {idx + 1} images")

        logger.info(f"Extracted {len(image_paths)} images")

        if doc_name is None:
            doc_name = f"{dataset_name.replace('/', '_')}_{split}"

        document_id = create_document(session, api_url, project_slug, doc_name)

        for idx, img_path in enumerate(image_paths):
            upload_image(session, api_url, document_id, img_path)
            if (idx + 1) % 10 == 0:
                logger.info(f"Uploaded {idx + 1}/{len(image_paths)} images")

        logger.info(f"Done — uploaded {len(image_paths)} images to document {document_id}")

    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Upload HF dataset images to eScriptorium")
    parser.add_argument("dataset")
    parser.add_argument("--split", default="train")
    parser.add_argument("--project-slug", default="admins-project")
    parser.add_argument("--doc-name")
    parser.add_argument("--image-column", default="image")
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()
    main(dataset_name=args.dataset, split=args.split, project_slug=args.project_slug,
         doc_name=args.doc_name, image_column=args.image_column, max_samples=args.max_samples)
