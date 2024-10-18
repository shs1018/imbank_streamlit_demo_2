# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

@st.cache_data
# 데이터 불러오기
def load_data():  
    df = pd.read_csv('dataset/data.csv')
    return df

# Visualization1
def vis_barchart(df,col_hue ,cols):

    if len(cols) == 1:
        st.warning("You have selected only one column. Please select at least one more column.")
    elif len(cols) == 0:
        st.info("No columns have chosen yet.")
    else:
        pass

    plt.figure(figsize=(10, 8))
    sns.barplot(x=cols[0], y = cols[1], data = df, hue = col_hue)
    plt.title(f"Bar Chart of {cols[0]}and {cols[1]}")
    plt.xlabel(cols[0])
    plt.ylabel(cols[1])
    plt.xticks(rotation = 45)
    plt.tight_layout()
    st.pyplot(plt)

# Visualization2
def vis_lmplot(df, col_hue, cols):
    if len(cols) == 1:
        st.warning("You have selected only one column. Please select at least one more column.")
    elif len(cols) == 0:
        st.info("No columns have chosen yet.")
    else:
        pass


    plt.figure(figsize=(10, 6))
    sns.lmplot(x = cols[0], y = cols[1], hue = col_hue,data = df ,height = 6, aspect = 1.5)
    plt.xlabel(cols[0])
    plt.ylabel(cols[1])
    plt.xticks(rotation = 45)
    plt.tight_layout()
    st.pyplot(plt)


# Visualization3
def vis_piechart(df, label, cols):

    cnt1 = df[label].value_counts()
    cnt2 = df[cols[0]].value_counts()
    cnt3 = df[cols[1]].value_counts()
    fig,ax = plt.subplots(1,3, figsize = (13,6))

    ax[0].pie(cnt1,labels = cnt1.index, autopct = '%1.1f%%', startangle = 90)
    ax[0].set_title(f"Pie Chart of {label}")

    ax[1].pie(cnt2,labels = cnt2.index, autopct = '%1.1f%%', startangle = 90)
    ax[1].set_title(f"Pie Chart of {cols[0]}")

    ax[2].pie(cnt3,labels = cnt3.index, autopct = '%1.1f%%', startangle = 90)
    ax[2].set_title(f"Pie Chart of {cols[1]}")

    plt.tight_layout()
    st.pyplot(fig)
    

# run ANOVA, and Visualization
def ANOVA_res(df, cols):
    # handling data
    if len(cols) < 2:
        st.warning("Please choose more than 2 colmumns!")
        return
    
    tmp_data = [df[col] for col in cols]

    # working ANOVA
    f, p = stats.f_oneway(*tmp_data)

    # print results
    res_tb = [{
        'F-statistics': f,
        'P-value': p 
    }]
    st.dataframe(res_tb, use_container_width = True)

    if p < 0.05:
        st.markdown("**The diff between your Chosen cloumns' mean is SIgnificant!**")
    else:
        st.markdown("**The diff between your Chosen cloumns' mean is Not SIgnificant!**")

    # Visualization
    melted_df = df[cols].melt(var_name='Group', value_name='Value')
    st.header("Boxplot of chosen columns")

    plt.figure(figsize = (15,6))
    sns.boxplot(data = melted_df, x = 'Group', y = 'Value')
    plt.title("Boxplot of ANOVA")
    plt.xticks(rotation = 45)
    plt.tight_layout()
    st.pyplot(plt)

    #pro-test
    st.header("tukey HSD resulst: ")
    tukey_res = pairwise_tukeyhsd(endog = melted_df['Value'], groups = melted_df['Group'], alpha = 0.05)
    st.dataframe(tukey_res.summary(), use_container_width = True)

def Credit_Logit_Reg(df):
    # loan_grade를 예측할 수 있는 credit point를 만들어야 한다.
    credit_info_cols = ['person_income','person_home_ownership','cb_person_cred_hist_length','person_emp_length', 'cb_person_default_on_file']
    target = 'loan_grade'

    # 범주형 target이기에 연속형으로 Encoding 해 준다
    label_encoder = LabelEncoder()
    df['loan_grade_encoded'] = label_encoder.fit_transform(df[target])

    # feature에 범주형 변수가 포함되어 있기에 전체를 대상으로 더미변수 생성
    X = pd.get_dummies(df[credit_info_cols], drop_first = True)
    y = df['loan_grade_encoded']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 연속형 변수는 따로 스케일링 해 주어야 한다
    scaler = StandardScaler()
    X_train[['person_income', 'cb_person_cred_hist_length', 'person_emp_length']] = scaler.fit_transform(X_train[['person_income', 'cb_person_cred_hist_length', 'person_emp_length']])
    X_test[['person_income', 'cb_person_cred_hist_length', 'person_emp_length']] = scaler.transform(X_test[['person_income', 'cb_person_cred_hist_length', 'person_emp_length']])

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    return model




def main():
    st.title("Insight From Loan Data! :face_with_monocle:")
    data_loan = load_data()

    #Ssidebar
    st.sidebar.title("Data Info :roller_coaster:")

    st.sidebar.header("Data Preview")
    st.sidebar.dataframe(data_loan, use_container_width = True)

    st.sidebar.header("Columns type")
    type_lst = [(col, dtype) for col, dtype in zip(data_loan.columns, data_loan.dtypes)]
    st.sidebar.dataframe(type_lst, use_container_width = True)

    st.sidebar.header("Basic Statistics")
    st.sidebar.write(data_loan.describe())

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Loan Prediction", "Risk Prediction", "Credit Prediction", "ANOVA Tool", "Feature Visualizatin"])

    with tab1:
        st.title("Loan Prediction")

        st.header("Please Enteit Info!")
        person_income_val = st.number_input('Input youal income!')
        person_home_ownership_val = st.selectbox("Choose your curree ownershius!", ['OWN', 'RENT', 'MORTGAGE', 'OTHER'])
        person_cred_hist_length = st.number_input('In your credit history length!')
        person_emp_length = st.number_input('Input your emngth!')
        person_default_on_file = st.selectbox("Choose 'Y' ce", ['Y', 'N'])
        
        if st.button("Run Predict Model", key = 1):
           st.write('미구현!')
     
    with tab2:
        st.title("Risk Prediction")

        st.header("Please Eit Info!")
        person_income_val = st.number_input('Input yl income!')
        person_home_ownership_val = st.selectbox("Choose your come ownership status!", ['OWN', 'RENT', 'MORTGAGE', 'OTHER'])
        person_cred_hist_length = st.number_input('Input your creditgth!')
        person_emp_length = st.number_input('Inpuength!')
        person_default_on_file = st.selectbox("Choose 'Y' if yoalut experiance", ['Y', 'N'])
        
        if st.button("Run Predict Model", key = 2):
           st.write('미구현!')
         
    with tab3:
        st.title("Credit Prediction :dollar:")

        st.header("Please Enter Your Credit Info!")
        person_income_val = st.number_input('Input your annual income!')
        person_home_ownership_val = st.selectbox("Choose your current home ownership status!", ['OWN', 'RENT', 'MORTGAGE', 'OTHER'])
        person_cred_hist_length = st.number_input('Input your credit history length!')
        person_emp_length = st.number_input('Input your employeed length!')
        person_default_on_file = st.selectbox("Choose 'Y' if you have Defalut experiance", ['Y', 'N'])
        
        if st.button("Run Predict Model", key = 3):
           st.write('미구현!')
           '''
           # Personal_data
            person_Cred = pd.DataFrame({
            'person_income': [person_income_val],
            'person_home_ownership': [person_home_ownership_val],
            'cb_person_cred_hist_length': [person_cred_hist_length],
            'person_emp_length': [person_emp_length],
            'cb_person_default_on_file': [person_default_on_file]
            })

           # new_data encoded & handled
            person_Credit = pd.get_dummies(person_Cred, drop_first=True)
            scaler = StandardScaler()
            person_Credit[['person_income', 'cb_person_cred_hist_length', 'person_emp_length']] = scaler.fit_transform(person_Credit[['person_income', 'cb_person_cred_hist_length', 'person_emp_length']])

           # model fitted
            C_model =  Credit_Logit_Reg(data_loan)
            pred_res = C_model.predict_proba(person_Credit)

            st.write(f"Credit Grade Predicted! :")
            st.write(f"  - A: {pred_res[0] * 100:.2f}%")
            st.write(f"  - B: {pred_res[1] * 100:.2f}%")
            st.write(f"  - C: {pred_res[2] * 100:.2f}%")
            st.write(f"  - D: {pred_res[3] * 100:.2f}%")
            st.write(f"  - E: {pred_res[4] * 100:.2f}%")
            st.write(f"  - F: {pred_res[5] * 100:.2f}%")
            st.write(f"  - G: {pred_res[6] * 100:.2f}%")
            '''
        
     
    with tab4:
        st.title("ANOVA Tool :bar_chart:")

        st.header("Choose Multiple Columns")
        cols = st.multiselect('columns', data_loan.columns, key = '4')
        
        st.header("Chosen Columns: ")
        st.dataframe(cols, use_container_width = True)

        st.header("ANOVA Results: ")
        ANOVA_res(data_loan, cols)


        st.header("Data Sample: ")
        sample_nums = st.slider("Sample Numbers", min_value = 10, max_value = 300)
        st.dataframe(data_loan.loc[:, cols].head(sample_nums), use_container_width = True)


         
    with tab5:
        st.title("Variance Visualization :chart:")

        st.header("Choose your Target:")
        target = st.selectbox('Choose One target!', data_loan.columns)

        st.header("Choose 2 Columns")
        cols = st.multiselect('columns', data_loan.columns, max_selections = 2 ,key = '5')

        if st.button("push me!"):
            vis_barchart(data_loan, target ,cols)
            vis_piechart(data_loan, target, cols)





if __name__ == "__main__":
    main()
