# ==================== similarity.py ====================
# Image-similarity helpers shared by 0a_prepare_dataset.py (grouping) and 0b_verify_split.py (verification).
# Heavy imports (torch, transformers, imagehash, ultralytics) are done lazily inside the functions.

import numpy as np


# ------------------------------------------------------------------
# Perceptual hash (catches near-identical images: re-encodes, resizes, small crops)
# ------------------------------------------------------------------
def phash_matrix(paths):
    import imagehash
    from PIL import Image
    hashes = []
    for p in paths:
        with Image.open(p) as im:
            hashes.append(imagehash.phash(im.convert("RGB")).hash.flatten())
    return np.array(hashes, dtype=bool)


def duplicate_pairs(H, thr):
    """All index pairs (i<j) whose 64-bit hashes differ in at most `thr` bits."""
    pairs = []
    for i in range(len(H)):
        d = (H[i + 1:] != H[i]).sum(axis=1)
        for j in np.where(d <= thr)[0]:
            pairs.append((i, i + 1 + int(j)))
    return pairs


# ------------------------------------------------------------------
# Embeddings (catches related images that a hash misses: other pose, other crop, same scene/animal)
# NOTE: embedding similarity is evidence of visual relatedness, not proof of "same animal".
# ------------------------------------------------------------------
def embed_images(paths, backend="dinov2", batch=32):
    """L2-normalised global embeddings, shape (n, d)."""
    paths = [str(p) for p in paths]
    if backend == "dinov2":
        import torch
        from PIL import Image
        from transformers import AutoImageProcessor, AutoModel
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        proc = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
        model = AutoModel.from_pretrained("facebook/dinov2-small").to(dev).eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(paths), batch):
                ims = []
                for p in paths[i:i + batch]:
                    with Image.open(p) as im:
                        ims.append(im.convert("RGB"))
                x = proc(images=ims, return_tensors="pt").to(dev)
                out.append(model(**x).last_hidden_state[:, 0].cpu().numpy())      # CLS token
        E = np.concatenate(out)
    elif backend == "ultralytics":
        from ultralytics import YOLO
        feats = YOLO("yolo11n-cls.pt").embed(paths, imgsz=224, verbose=False)
        E = np.stack([f.detach().cpu().numpy().ravel() for f in feats])
    else:
        raise ValueError(f"unknown embedding backend: {backend}")
    return E / np.linalg.norm(E, axis=1, keepdims=True)


def nearest_neighbors(Q, R, exclude_self=False):
    """For each row of Q, index and cosine similarity of its nearest row in R (rows are L2-normalised)."""
    S = Q @ R.T
    if exclude_self:
        np.fill_diagonal(S, -np.inf)
    j = S.argmax(axis=1)
    return j, S[np.arange(len(Q)), j]


def similarity_pairs(E, thr):
    """All index pairs (i<j) with cosine similarity >= thr."""
    S = E @ E.T
    iu = np.triu_indices(len(E), 1)
    keep = S[iu] >= thr
    return list(zip(iu[0][keep].tolist(), iu[1][keep].tolist()))
