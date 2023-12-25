import faiss
import pandas as pd
from commands_faiss import embeded_id

if __name__ == "__main__":
    cv_csv = "./data/processed/res_prepared.csv"
    vac_csv = "./data/processed/vac_prepared.csv"
    cv_index_demo = "demo/data/cv_db.index"
    vac_index_demo = "demo/data/vac_db.index"
    vac_id = 7
    topn = 10

    cv_index = faiss.read_index(cv_index_demo)
    vac_index = faiss.read_index(vac_index_demo)

    cv_data = pd.read_csv(cv_csv)
    vac_data = pd.read_csv(vac_csv)

    vac_embed = get_embedding_by_id(vac_index, vac_id)
    _, cv_ids = cv_index.search(vac_embed, topn)

    print("Для лучшего работодателя")
    print(vac_data.iloc[vac_id])

    print(f"Были подобраны лучшие кандидаты:")
    print(vac_data.iloc[cv_ids[0]])
