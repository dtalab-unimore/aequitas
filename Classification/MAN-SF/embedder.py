import os
import queue
import threading

import numpy as np
import tensorflow_hub as hub

_global_embed = None
_global_sess = None


def worker_initializer():
    global _global_embed, _global_sess
    print(f"[Worker {os.getpid()}] Loading Universal Sentence Encoder...")
    _global_embed, _global_sess = init_embedder()


def init_embedder():
    os.environ.setdefault("TFHUB_CACHE_DIR", "./tfhub_modules")
    try:
        model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    except Exception as exc:
        raise RuntimeError(
            "Universal Sentence Encoder is not cached locally and could not be downloaded."
        ) from exc
    return model, None


def async_generator(iterable, max_prefetch=10):
    items = queue.Queue(max_prefetch)

    def producer():
        for item in iterable:
            items.put(item)
        items.put(None)

    threading.Thread(target=producer, daemon=True).start()
    while True:
        item = items.get()
        if item is None:
            break
        yield item


def batch(iterable, size=512):
    values = []
    for item in iterable:
        values.append(item)
        if len(values) == size:
            yield values
            values = []
    if values:
        yield values


def embed_news_async(news_texts, embed, sess=None, batch_size=512):
    del sess
    embeddings = []
    for texts in batch(async_generator(news_texts), batch_size):
        encoded = embed(texts)
        embeddings.append(np.asarray(encoded))
    if not embeddings:
        return np.empty((0, 512), dtype=np.float32)
    return np.vstack(embeddings)
