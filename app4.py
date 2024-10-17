import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

@st.cache_data
def load_data():
    df = sns.load_dataset('planets')
    return df

def plot_matplotlib(df):
    st.title("Scatterplot with matplotlib")
    fig, ax = plt.subplots()
    ax.scatter(df['orbital_period'], df['distance'])
    st.pyplot(fig)

def plot_sns(df):
    st.title('Scatterplot with seaborn')
    fig, ax = plt.subplots()
    sns.scatterplot(df, x = 'orbital_period', y = 'distance')
    st.pyplot(fig)

def plot_plotly(df):
    st.title('Scatterplot with plotly')
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x = df['orbital_period'], 
                   y = df['distance'],
                   made = 'markers')
    )
    st.plotly_chart(fig)

def main():
    st.title("Choose Ur fighter")
    planets = load_data()
    st.data_editor(planets.head(30))

    plot_type = st.radio(
        "Who's your Fighter?",
        ('Matplotlib', 'Seaborn', 'Plotly')
    )

    st.write(plot_type)

    if plot_type == 'Matplotlib':
        plot_matplotlib(planets)
    elif plot_type == 'Seaborn':
        plot_sns(planets)
    elif plot_type == 'Plotly':
        plot_plotly(planets)

if __name__ == "__main__":
    main()