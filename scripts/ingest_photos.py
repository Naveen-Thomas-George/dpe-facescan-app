# scripts/ingest_photos.py
import argparse, os, uuid, numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import select

from app.config import settings
from app.db import SessionLocal
from app.models import Event, Photo
from app.services.storage import upload_image_bytes
from app.services.face import extract_face_embeddings, embed_image_bytes  # multi + single
from app.services.index import load_or_create_index, add_embeddings, persist_index

# ---- NEW: hash util ----
import hashlib
def compute_file_hash(data: bytes) -> str:
    """Compute SHA256 hash of file bytes."""
    return hashlib.sha256(data).hexdigest()


def is_image(p): 
    return p.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))


def main(event_slug: str, folder: str):
    os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
    emb_root = os.path.join(settings.MEDIA_ROOT, "embeddings")
    os.makedirs(emb_root, exist_ok=True)

    db: Session = SessionLocal()

    # ensure event exists
    ev = db.execute(select(Event).where(Event.slug == event_slug)).scalar_one_or_none()
    if not ev:
        ev = Event(slug=event_slug, name=event_slug)
        db.add(ev); db.commit(); db.refresh(ev)

    index, ids = load_or_create_index(event_slug=event_slug)
    new_embs, new_ids = [], []

    files = [os.path.join(folder, f) for f in os.listdir(folder) if is_image(f)]
    print(f"Found {len(files)} images")

    for path in files:
        with open(path, "rb") as f:
            data = f.read()

        # ---- Duplicate detection ----
        file_hash = compute_file_hash(data)
        existing = db.execute(select(Photo).where(Photo.file_hash == file_hash)).scalar_one_or_none()
        if existing:
            print(f"[SKIP] Duplicate already ingested → {path}")
            continue

        # ---- Upload to storage ----
        public_id = f"{event_slug}/{uuid.uuid4().hex}"
        uri, thumb = upload_image_bytes(data, public_id)

        # ---- Representative embedding ----
        rep_emb = embed_image_bytes(data)
        rep_path = os.path.join(emb_root, f"{uuid.uuid4().hex}.npy")
        np.save(rep_path, rep_emb)

        # ---- Save Photo row ----
        photo = Photo(
            event_id=ev.id,
            uri=uri,
            thumb_uri=thumb,
            embedding_path=rep_path,
            file_hash=file_hash    # <---- IMPORTANT
        )
        db.add(photo); db.commit(); db.refresh(photo)

        # ---- Multi-face extraction & indexing ----
        faces = extract_face_embeddings(data)
        if not faces:
            if rep_emb.size > 0:
                new_embs.append(rep_emb.astype("float32"))
                new_ids.append(photo.id)
            print(f"[NoFaces] {os.path.basename(path)} → stored photo only.")
            continue

        count = 0
        for emb, bbox in faces:
            if emb is None or emb.size == 0:
                continue
            emb = emb.astype("float32")
            face_path = os.path.join(emb_root, f"{uuid.uuid4().hex}.npy")
            np.save(face_path, emb)

            new_embs.append(emb)
            new_ids.append(photo.id)
            count += 1

        print(f"[OK] {os.path.basename(path)} → {count} face(s) indexed → {uri}")

    # ---- Persist FAISS index ----
    if new_embs:
        embs = np.vstack(new_embs).astype("float32")
        new_ids_arr = np.array(new_ids, dtype="int64")
        index, ids = add_embeddings(index, ids, embs, new_ids_arr, metric=settings.FAISS_METRIC)
        persist_index(index, ids, event_slug=event_slug)
        print(f"Indexed {len(new_ids)} face embeddings.")
    else:
        print("No new embeddings to index.")

    db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--event", required=True, help="event slug, e.g., christ-sports-2025")
    parser.add_argument("folder", help="folder with images")
    args = parser.parse_args()
    settings.EVENT_SLUG = args.event  # set for this run
    main(args.event, args.folder)
