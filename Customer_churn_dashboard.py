import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title='Customer Churn Dashboard', layout='wide')
st.title('🏦 Customer Churn Analysis Dashboard')

@st.cache_data
def load_data():
    df = pd.read_csv('Churn_Modelling.csv')
    df = df.drop(['RowNumber','CustomerId','Surname'], axis=1)
    df['Gender'] = df['Gender'].map({'Male':1,'Female':0})
    df = pd.get_dummies(df, columns=['Geography'], drop_first=True)
    df['AgeGroup'] = pd.cut(df['Age'], bins=[18,30,45,60,100], labels=['Young','Adult','MidAge','Senior'])
    df['BalancePerProduct'] = df['Balance']/(df['NumOfProducts']+1)
    df['TenureGroup'] = pd.cut(df['Tenure'], bins=[0,3,7,10], labels=['New','Mid','Old'])
    model_df = pd.get_dummies(df, columns=['AgeGroup','TenureGroup'], drop_first=True)
    X = model_df.drop('Exited', axis=1)
    y = model_df['Exited']
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train,y_train)
    preds = rf.predict(X_test)
    acc = (preds==y_test).mean()
    full_scaled = scaler.transform(X)
    df['ChurnRisk'] = rf.predict_proba(full_scaled)[:,1]
    df['RiskCategory'] = pd.cut(df['ChurnRisk'], bins=[0,0.3,0.7,1], labels=['Low','Medium','High'])
    return df, acc

df, acc = load_data()

# Sidebar filters
st.sidebar.header('Filters')
gender = st.sidebar.multiselect('Gender', options=df['Gender'].unique(), default=df['Gender'].unique())
risk = st.sidebar.multiselect('Risk Category', options=df['RiskCategory'].dropna().unique(), default=df['RiskCategory'].dropna().unique())
filtered = df[df['Gender'].isin(gender) & df['RiskCategory'].isin(risk)]

# KPIs
c1,c2,c3,c4 = st.columns(4)
c1.metric('Total Customers', len(filtered))
c2.metric('Churn Rate', f"{filtered['Exited'].mean()*100:.2f}%")
c3.metric('High Risk Customers', int((filtered['RiskCategory']=='High').sum()))
c4.metric('Best Model Accuracy', f'{acc*100:.2f}%')

# Charts
col1,col2 = st.columns(2)
with col1:
    st.subheader('Churn Distribution')
    fig, ax = plt.subplots()
    sns.countplot(x='Exited', data=filtered, ax=ax)
    st.pyplot(fig)
with col2:
    st.subheader('Risk Segmentation')
    fig, ax = plt.subplots()
    sns.countplot(x='RiskCategory', data=filtered, order=['Low','Medium','High'], ax=ax)
    st.pyplot(fig)

col3,col4 = st.columns(2)
with col3:
    st.subheader('Age vs Churn')
    fig, ax = plt.subplots()
    sns.histplot(data=filtered, x='Age', hue='Exited', bins=25, ax=ax)
    st.pyplot(fig)
with col4:
    st.subheader('Balance by Churn')
    fig, ax = plt.subplots()
    sns.boxplot(x='Exited', y='Balance', data=filtered, ax=ax)
    st.pyplot(fig)

st.subheader('High Risk Customers Preview')
st.dataframe(filtered[filtered['RiskCategory']=='High'][['CreditScore','Age','Balance','EstimatedSalary','ChurnRisk']].head(20))

st.caption('Built with Streamlit | Customer Churn Analysis in Banking')
