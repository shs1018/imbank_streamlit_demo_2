# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import seaborn as sns

@st.cache_data
def load_data():
    df = sns.load_dataset('penguins')
    return df

def main():
    st.title('SelectBox 사용')

    pg = load_data()
    st.markdown('##Raw Data')
    st.dataframe(pg)

    st.markdown("<hr>", unsafe_allow_html = True)
    st.markdown("## MultiSelect")

    cols = st.multiselect("복수 칼럼 선택", pg.columns)
    st.write("선택된 칼럼: ", cols)
    st.dataframe(pg.loc[:, cols])

if __name__ == "__main__":
    main()