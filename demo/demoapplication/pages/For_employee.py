import faiss
import numpy as np
import pandas as pd
import streamlit as st

cv_csv = "./data/processed/res_prepared.csv"
vac_csv = "./data/processed/vac_prepared.csv"
cv_index_demo = "data/cv_db.index"
vac_index_demo = "data/vac_db.index"
vac_id = 7
topn = 10

cv_index = faiss.read_index(cv_index_demo)
vac_index = faiss.read_index(vac_index_demo)

cv_data = pd.read_csv(cv_csv)
vac_data = pd.read_csv(vac_csv)
cv_number = len(cv_data) - 1


def embeded_id(index, vec_id):
    vec = np.empty((1, index.d), dtype="float32")
    index.reconstruct(vec_id, vec[0])
    return vec


def main():
    
    st.title("Подбор наиболее подходящих вакансий по резюме")

    cv_id = st.number_input(
        "Введите id резюме",
        min_value=0,
        max_value=cv_number,
        value=7,
    )

    cv_button = st.button("Показать резюме")
    if cv_button:
        st.write("Выбранное резюме")
        st.dataframe(cv_data.iloc[cv_id])

    topn = st.number_input(
        "Введите кол-во подбираемых вакансий (оптимально 10)",
        min_value=1,
        max_value=20,
        value=10,
    )

    vac_button = st.button("Подобрать вакансии")
    if vac_button:
        cv_embed = embeded_id(cv_index, cv_id)
        _, vac_ids = vac_index.search(cv_embed, topn)
        
        st.write("Для лучшего соискателя")
        st.dataframe(cv_data.iloc[cv_id])
        
        st.write(f"Были подобраны лучшие работодатели:")
        st.dataframe(vac_data.iloc[vac_ids[0]])


if __name__ == "__main__":
    main()
