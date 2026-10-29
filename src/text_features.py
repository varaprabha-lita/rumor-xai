"""Source-tweet text representations.

  tfidf : word+char TF-IDF -> TruncatedSVD (fit on the training fold only). Runs anywhere.
  sbert : sentence-transformers encoder (default all-mpnet-base-v2). Needs `pip install sentence-transformers`.
  llm   : mean-pooled last hidden state of a local HF causal LM (e.g. Qwen/Qwen2.5-7B-Instruct,
          meta-llama/Llama-3.1-8B-Instruct). Needs a GPU and `transformers`.

sbert/llm embeddings are label-free, so they are computed once for all events and cached in
data/cache/. tfidf is fitted per fold inside `fold_text_features` to avoid test leakage.
"""
from __future__ import annotations

import hashlib
import os
import re

import numpy as np

CACHE = os.path.join(os.path.dirname(__file__), "..", "data", "cache")


def clean(t: str) -> str:
    t = re.sub(r"http\S+|\bURL\b", " url ", t)
    t = re.sub(r"@\w+", " @user ", t)
    return t.lower().strip()


def tfidf_svd(train_txt, test_txt, dim=256, seed=0):
    from scipy.sparse import hstack
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    w = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)
    xtr = hstack([w.fit_transform(train_txt), c.fit_transform(train_txt)]).tocsr()
    xte = hstack([w.transform(test_txt), c.transform(test_txt)]).tocsr()
    dim = min(dim, xtr.shape[1] - 1, xtr.shape[0] - 1)
    svd = TruncatedSVD(dim, random_state=seed)
    ztr = svd.fit_transform(xtr)
    zte = svd.transform(xte)
    mu, sd = ztr.mean(0), ztr.std(0) + 1e-6
    return (ztr - mu) / sd, (zte - mu) / sd


def _cache_path(kind, model, texts):
    h = hashlib.md5(("\n".join(texts) + model).encode()).hexdigest()[:10]
    os.makedirs(CACHE, exist_ok=True)
    return os.path.join(CACHE, f"{kind}_{model.replace('/', '_')}_{h}.npy")


def sbert_embed(texts, model="sentence-transformers/all-mpnet-base-v2", batch=64):
    p = _cache_path("sbert", model, texts)
    if os.path.exists(p):
        return np.load(p)
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(model)
    z = m.encode([clean(t) for t in texts], batch_size=batch, show_progress_bar=True, normalize_embeddings=True)
    np.save(p, z)
    return z


def llm_embed(texts, model="Qwen/Qwen2.5-7B-Instruct", batch=16, max_len=96, load_4bit=False):
    p = _cache_path("llm", model, texts)
    if os.path.exists(p):
        return np.load(p)
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = dict(torch_dtype=torch.bfloat16, device_map="auto")
    if load_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    enc = AutoModel.from_pretrained(model, **kw).eval()
    out = []
    prompt = "Social media post: {}\nThis post is"
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            chunk = [prompt.format(clean(t)) for t in texts[i:i + batch]]
            b = tok(chunk, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(enc.device)
            h = enc(**b).last_hidden_state.float()
            m = b["attention_mask"].unsqueeze(-1).float()
            out.append(((h * m).sum(1) / m.sum(1)).cpu().numpy())
    z = np.concatenate(out)
    z = (z - z.mean(0)) / (z.std(0) + 1e-6)
    np.save(p, z)
    return z


def fold_text_features(kind, texts, tr_idx, te_idx, model=None):
    """Return (Z_train, Z_test) for one fold."""
    texts = list(texts)
    if kind == "tfidf":
        return tfidf_svd([clean(texts[i]) for i in tr_idx], [clean(texts[i]) for i in te_idx])
    if kind == "sbert":
        z = sbert_embed(texts, model or "sentence-transformers/all-mpnet-base-v2")
    elif kind == "llm":
        z = llm_embed(texts, model or "Qwen/Qwen2.5-7B-Instruct")
    else:
        raise ValueError(kind)
    return z[tr_idx], z[te_idx]
