# -*- coding: utf-8 -*-
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns

@st.cache_data
def load_data():
    df = sns.load_dataset('penguins')
    return df

def main():
    st.title("Let's get down to business!")

    pg = load_data()

    #성별 가공
    m_pg = pg.loc[pg['sex'] == 'Male',:]
    f_pg = pg.loc[pg['sex'] == 'Female',:]

    #시각화 차트
    fig = make_subplots(
        rows = 1,
        cols = 2,
        subplot_titles = ("Male", "Female"),
        shared_yaxes = True,
        shared_xaxes = True,
        x_title = 'body_mass'
    )

    fig.add_trace(go.Scatter(x = m_pg['body_mass_g'], y = m_pg['species'], mode = 'markers'), row = 1, col = 1)
    fig.add_trace(go.Scatter(x = f_pg['body_mass_g'], y = f_pg['species'], mode = 'markers'), row = 1, col = 2)
    fig.update_yaxes(title_text = "Species", row = 1, col = 1)
    fig.update_xaxes(range = [2500,6500])
    fig.update_layout(showlegend = False)


    # 중요포인트
    # fig_show()
    st.plotly_chart(fig, use_container_width = True)



if __name__ == "__main__":
    main()