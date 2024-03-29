import os

import faiss
import numpy as np
import pandas as pd
import streamlit as st

cv_csv = "data/res_prepared.csv"
vac_csv = "data/vac_prepared.csv"
cv_index_demo = "data/cv_db.index"
vac_index_demo = "data/vac_db.index"

cv_index = faiss.read_index(cv_index_demo)
vac_index = faiss.read_index(vac_index_demo)

cv_data = pd.read_csv(cv_csv)
vac_data = pd.read_csv(vac_csv)
vac_number = len(cv_data) - 1


def embeded_id(index, vec_id):
    vec = np.empty((1, index.d), dtype="float32")
    index.reconstruct(vec_id, vec[0])
    return vec


def main():
    st.title("Подбор наиболее подходящих кандидатов")

    vac_id = st.number_input(
        "Введите номер вакансии",
        min_value=0,
        max_value=vac_number,
        value=7,
    )

    topn = st.number_input(
        "Введите кол-во подбираемых кандидатов (оптимально 10)",
        min_value=1,
        max_value=30,
        value=10,
    )

    cv_button = st.button("Подобрать кандидатов")
    if cv_button:
        vac_embed = embeded_id(vac_index, vac_id)
        _, cv_ids = cv_index.search(vac_embed, topn)

        st.write("Для лучшего работодателя")
        st.dataframe(vac_data.iloc[vac_id])

        st.write(f"Были подобраны лучшие кандидаты:")
        st.dataframe(cv_data.iloc[cv_ids[0]])


if __name__ == "__main__":
    main()
