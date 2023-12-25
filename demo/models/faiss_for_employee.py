import os

import faiss
from commands_faiss import batching, text_for_demo
from sentence_transformers import SentenceTransformer

if __name__ == "__main__":
    
    db_path = "demo/data"
    vac_db_name = "vac_db.index"

    vac_path = ".data/processed/vac_prepared.csv"
    needed_columns = ["name", "description"]

    model_name = "cointegrated/rubert-tiny2"
    model = SentenceTransformer(model_name)

    vacs = text_for_demo(vac_path, needed_columns)
    vacs_vectors = batching(model, vacs, batch_size=256)
    dim = vacs_vectors.shape[1]

    vac_index = faiss.IndexFlatL2(dim)
    vac_index.add(vacs_vectors)

    vac_db = os.path.join(db_path, vac_db_name)
    faiss.write_index(vac_index, vac_db)