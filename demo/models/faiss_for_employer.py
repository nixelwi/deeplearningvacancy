import os

import faiss
from commands_faiss import batching, text_for_demo
from sentence_transformers import SentenceTransformer

if __name__ == "__main__":

    db_path = "demo/data"
    cv_db_name = "cv_db.index"

    cv_path = ".data/processed/res_prepared.csv"
    needed_columns = ["Ищет работу на должность:", "Опыт работы"]

    model_name = "cointegrated/rubert-tiny2"
    model = SentenceTransformer(model_name)

    cvs = text_for_demo(cv_path, needed_columns)
    cvs_vectors = batching(model, cvs, batch_size=256)
    dim = cvs_vectors.shape[1]

    cv_index = faiss.IndexFlatL2(dim)
    cv_index.add(cvs_vectors)

    cv_db = os.path.join(db_path, cv_db_name)
    faiss.write_index(cv_index, cv_db)
