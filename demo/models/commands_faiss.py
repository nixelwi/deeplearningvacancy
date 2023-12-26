import pandas as pd
import numpy as np


def text_for_demo(dataset, text):
    vac_demo = pd.read_csv(dataset)[text]
    vac_demo = vac_demo[text].astype(str).agg(" ".join, axis=1)
    vac = vac_demo.to_list()
    return vac


def batching(model, sentences, batch_size):
    embeddings = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        batch_embeddings = model.encode(batch)
        embeddings.extend(batch_embeddings)
    embeddings = np.array(embeddings)
    return embeddings
