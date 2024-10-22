import streamlit as st 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

def cal_sales_revenue(price, total_sales):
    revenue = price * total_sales
    return revenue

def cal_original_cost(price, total_sales, cost):
    original_cost = price * total_sales - total_sales * cost
    return original_cost

def main():
    st.title("Why Me? Why Not")

    price = st.slider('단가:', 1, 100000, value =50000)
    total_sales = st.slider('판매 갯수:',1,1000, value = 500)
    cost = st.slider("원가: ", 1, 100000, value = 50000)
    
    print(price, total_sales, cost)
    st.write("단가: ", price, "판매 갯수:", total_sales, '원가: ', cost)

    if st.button("매출액 계산"):
        revenue = cal_sales_revenue(price, total_sales)
        st.write(revenue)

    if st.button("순이익 계산"):
        original_cost = cal_original_cost(price, total_sales, cost)
        st.write(original_cost)

    st.title('Check Box Control')
    x = np.linspace(0, price, total_sales)
    y = np.sin(x)
    z = np.cos(x)

    show_plot = st.checkbox('시각화 보여주기')
    if show_plot:
        fig, ax = plt.subplots()
        ax.plot(x,y)
        ax.plot(x,z)
        st.pyplot(fig)

if __name__ == "__main__":
    main()