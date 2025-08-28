import os, numpy as np, faiss
from app.config import settings

def _idx(event: str): return os.path.join(settings.MEDIA_ROOT, "indices", f"{event}.faiss")
def _ids(event: str): return os.path.join(settings.MEDIA_ROOT, "indices", f"{event}.ids.npy")

def load_or_create_index(dim=512, metric=None, event_slug=None):
    metric = metric or settings.FAISS_METRIC
    event_slug = event_slug or settings.EVENT_SLUG
    p, pid = _idx(event_slug), _ids(event_slug)
    if os.path.exists(p) and os.path.exists(pid):
        return faiss.read_index(p), np.load(pid)
    index = faiss.IndexFlatIP(dim) if metric == "cosine" else faiss.IndexFlatL2(dim)
    return index, np.array([], dtype=np.int64)

def persist_index(index, ids, event_slug):
    os.makedirs(os.path.dirname(_idx(event_slug)), exist_ok=True)
    faiss.write_index(index, _idx(event_slug))
    np.save(_ids(event_slug), ids)

def add_embeddings(index, ids, embs, new_ids, metric=None):
    metric = metric or settings.FAISS_METRIC
    if metric == "cosine":
        embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)
    index.add(embs.astype(np.float32))
    ids = np.concatenate([ids, new_ids.astype(np.int64)]) if ids.size else new_ids.astype(np.int64)
    return index, ids

def search(index, q, top_k=None, metric=None):
    top_k = top_k or settings.TOP_K
    metric = metric or settings.FAISS_METRIC
    q = q.astype(np.float32).reshape(1,-1)
    if metric == "cosine":
        q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)
        sims, idx = index.search(q, top_k); return sims[0], idx[0]
    dists, idx = index.search(q, top_k); return (-dists[0]), idx[0]
