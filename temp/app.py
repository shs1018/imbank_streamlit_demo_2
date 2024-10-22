# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import time

@st.cache_data
def load_data():
    iris = load_iris()
    iris = pd.DataFrame(iris.data, columns=iris.feature_names)
    return iris

def main():
    st.title('Now let\'s get down to business!')
    
    iris = load_data()
    print(iris.head())
    st.dataframe(iris.head(), width = 1000)
    st.data_editor(iris)

    data_df = pd.DataFrame(
        {
            "sales": [
                [0, 4, 26, 80, 100, 40],
                [80, 20, 80, 35, 40, 100],
                [10, 20, 80, 80, 70, 0],
                [10, 100, 20, 100, 30, 100],
            ],
        }
    )

    st.dataframe(data_df)

    st.data_editor(
        data_df,
        column_config = {
            'sales': st.column_config.BarChartColumn(
                "sales (last 6 months)",
                help = "The sales colume in the last 6 months",
                y_min = 0,
                y_max = 100,
            ),
        },
        hide_index = True,
    )



if __name__ == "__main__":
    main()