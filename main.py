import io
import zipfile
import requests
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from sqlalchemy.orm import Session
from sqlalchemy import select

from app.config import settings
from app.db import Base, engine, get_db
from app.models import Event, Photo
from app.services.face import embed_image_bytes
from app.services.index import load_or_create_index, search


app = FastAPI(title="CUBYCSPO")
Base.metadata.create_all(bind=engine)

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")
app.state.templates = templates
templates.env.globals["config"] = settings


@app.get("/", response_class=HTMLResponse)
async def landing(request: Request):
    return templates.TemplateResponse(
        "index.html", {"request": request, "event_slug": settings.EVENT_SLUG}
    )


@app.get("/results", response_class=HTMLResponse)
async def results_page(request: Request):
    return templates.TemplateResponse("results.html", {"request": request})


@app.post("/api/search")
async def api_search(
    selfie: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    if selfie.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(status_code=400, detail="Unsupported file type")

    img_bytes = await selfie.read()
    emb = embed_image_bytes(img_bytes)
    if emb.sum() == 0.0:
        return {"matches": [], "note": "No face detected"}

    index, ids = load_or_create_index(event_slug=settings.EVENT_SLUG)
    if index.ntotal == 0 or ids.size == 0:
        return {"matches": [], "note": "Index empty"}

    sims, idxs = search(index, emb, top_k=settings.TOP_K)
    matches = []
    for sim, i in zip(sims.tolist(), idxs.tolist()):
        if i < 0:
            continue
        if settings.FAISS_METRIC == "cosine" and sim < settings.MATCH_THRESHOLD:
            continue
        photo_id = ids[i]
        photo = db.execute(
            select(Photo).where(Photo.id == int(photo_id))
        ).scalar_one_or_none()
        if photo:
            matches.append(
                {
                    "photo_id": photo.id,
                    "url": photo.uri,
                    "thumb": photo.thumb_uri,
                    "score": float(sim),
                }
            )

    return {"matches": matches}


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.post("/download_zip")
async def download_zip(request: Request):
    """
    Expects JSON body: { "urls": ["https://...jpg", "https://...jpg"] }
    """
    data = await request.json()
    photo_urls = data.get("urls", [])

    if not photo_urls:
        return JSONResponse({"error": "No photos to download"}, status_code=400)

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        for i, url in enumerate(photo_urls):
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    zf.writestr(f"photo_{i+1}.jpg", response.content)
            except Exception as e:
                print(f"Error fetching {url}: {e}")

    zip_buffer.seek(0)

    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=matched_photos.zip"},
    )
